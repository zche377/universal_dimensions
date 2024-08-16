"""
    Adapted code from Raj Magesh
"""
from collections.abc import Sequence
import functools

import more_itertools
from tqdm.auto import tqdm
import numpy as np
import torch
import xarray as xr
from bonner.computation.decomposition import PLSSVD
from bonner.computation.cuda import try_devices
from bonner.computation.regression._utilities import create_splits
from bonner.computation.metrics._corrcoef import _helper
from bonner.computation.xarray import align_source_to_target


uncentered_covariance = functools.partial(
    _helper,
    center=False,
    scale=False,
    correction=1,
    return_diagonal=True,
    copy=True,
)

uncentered_correlation = functools.partial(
    _helper,
    center=False,
    scale=True,
    correction=1,
    return_diagonal=True,
    copy=True,
)


class PLSSVDScorer:
    def __init__(
        self,
        *,
        n_folds: int = 0,
        center: bool = True,
        scale: bool = False,
        n_permutations: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        batch_size: int = 10,
    ) -> None:
        self.n_folds = n_folds
        self.center = center
        self.scale = scale
        self.n_permutations = n_permutations
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size

        self.identifier = (
            "plssvd"
            f".n_folds={self.n_folds}"
            f".center={self.center}"
            f".scale={self.scale}"
            f".n_permutations={self.n_permutations}"
            f".shuffle={self.shuffle}"
            f".seed={self.seed}"
        )

    def __call__(
        self,
        *,
        x_train: xr.DataArray,
        y_train: xr.DataArray,
        x_test: xr.DataArray,
        y_test: xr.DataArray,
    ) -> xr.Dataset:
        outputs = []

        if self.n_folds == 0:
            splits = [None]
        else:
            splits = create_splits(
                n=y_test.shape[-2],
                n_folds=self.n_folds,
                shuffle=self.shuffle,
                seed=self.seed,
            )

        for fold, indices_test in enumerate(splits):
            output = _plssvd(
                x_train=torch.from_numpy(x_train.values),
                y_train=torch.from_numpy(y_train.values),
                x_test=torch.from_numpy(x_test.values),
                y_test=torch.from_numpy(y_test.values),
                indices_test=indices_test,
                seed=self.seed,
                n_components=None,
                center=self.center,
                scale=self.scale,
                n_permutations=self.n_permutations,
                batch_size=self.batch_size,
            )
            outputs.append(output.expand_dims({"fold": [np.array(fold).astype(np.uint8)]}))

        return xr.merge(outputs)


def _plssvd(
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    indices_test: Sequence[int] | None,
    n_components: int | None,
    seed: int,
    center: bool,
    scale: bool,
    n_permutations: int,
    batch_size: int,
) -> xr.Dataset:
    if indices_test is None:
        indices_test = list(set(range(x_train.shape[-2])))
        indices_train = indices_test
    else:
        indices_train = list(set(range(x_train.shape[-2])) - set(indices_test))

    plssvd = PLSSVD(n_components=n_components, seed=seed, center=center, scale=scale)
    try_devices(plssvd.fit)(x_train[..., indices_train, :], y_train[..., indices_train, :])

    s1 = try_devices(plssvd.transform)(x_test[..., indices_test, :], direction="left")
    s2 = try_devices(plssvd.transform)(y_test[..., indices_test, :], direction="right")
    
    output = xr.Dataset(
        data_vars={
            "covariance": xr.DataArray(
                data=uncentered_covariance(s1, s2).cpu().numpy(),
                dims=("rank",),
            ),
            "correlation": xr.DataArray(
                data=uncentered_correlation(s1, s2).cpu().numpy(),
                dims=("rank",),
            ),
            "rank": xr.DataArray(
                data=1 + torch.arange(plssvd.n_components).numpy().astype(np.uint32),
                dims=("rank",),
            ),
        }
    )

    if n_permutations > 0:
        rng = np.random.default_rng(seed=seed)
        permutations = np.stack(
            [
                rng.permutation(len(indices_test))
                for _ in range(n_permutations)
            ]
        )

        output = output.assign(
            {
                "covariance (permuted)": xr.DataArray(
                    data=np.empty((n_permutations, plssvd.n_components)),
                    dims=("permutation", "rank"),
                ),
                "correlation (permuted)": xr.DataArray(
                    data=np.empty((n_permutations, plssvd.n_components)),
                    dims=("permutation", "rank"),
                ),
            }
        )

        start = 0
        for batch in enumerate(tqdm(more_itertools.chunked(permutations, n=batch_size), desc="permutation", leave=False)):

            s1 = try_devices(plssvd.transform)(x_test[np.stack(batch), :], direction="left")

            output["covariance (permuted)"][start:start + len(batch), :] = uncentered_covariance(s1, s2).cpu().numpy()
            output["correlation (permuted)"][start:start + len(batch), :] = uncentered_correlation(s1, s2).cpu().numpy()
            start += len(batch)

    return output


class CrossPredictorPLSSVDScorer:
    def __init__(
        self,
        *,
        n_folds: int = 5,
        center: bool = True,
        scale: bool = False,
        shuffle: bool = True,
        seed: int = 0,
        batch_size: int = 10,
    ) -> None:
        self.n_folds = n_folds
        self.center = center
        self.scale = scale
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size

        self.identifier = (
            "cross_predictor_plssvd"
            f".n_folds={self.n_folds}"
            f".center={self.center}"
            f".scale={self.scale}"
            f".shuffle={self.shuffle}"
            f".seed={self.seed}"
        )


    def __call__(
        self,
        *,
        x_train_rep: xr.DataArray,
        x_test_rep: xr.DataArray,
        y: xr.DataArray,
    ) -> xr.Dataset:
        x_train_rep, y = self._align_predictor_and_target(
            predictor=x_train_rep, target=y, target_dim="neuroid",
        )
        x_test_rep, y = self._align_predictor_and_target(
            predictor=x_test_rep, target=y, target_dim="neuroid",
        )
        
        outputs = []

        splits = create_splits(
            n=y.shape[-2],
            n_folds=self.n_folds,
            shuffle=self.shuffle,
            seed=self.seed,
        )

        for fold, indices_test in enumerate(splits):
            output = _cp_plssvd(
                x_train_rep=torch.from_numpy(x_train_rep.values),
                x_test_rep=torch.from_numpy(x_test_rep.values),
                y=torch.from_numpy(y.values),
                indices_test=indices_test,
                seed=self.seed,
                n_components=None,
                center=self.center,
                scale=self.scale,
                batch_size=self.batch_size,
            )
            outputs.append(output.expand_dims({"fold": [np.array(fold).astype(np.uint8)]}))

        return xr.merge(outputs)
    
    def _align_predictor_and_target(
        self,
        *,
        predictor: xr.DataArray,
        target: xr.DataArray,
        target_dim: str,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        predictor = align_source_to_target(
            source=predictor,
            target=target,
            sample_coord="stimulus_id",
            sample_dim="presentation",
        ).transpose("presentation", target_dim)
        target = target.transpose("presentation", target_dim)
        return predictor, target

# notice that test for indices is different from test for x
def _cp_plssvd(
    *,
    x_train_rep: torch.Tensor,
    x_test_rep: torch.Tensor,
    y: torch.Tensor,
    indices_test: Sequence[int] | None,
    n_components: int | None,
    seed: int,
    center: bool,
    scale: bool,
    batch_size: int,
) -> xr.Dataset:
    if indices_test is None:
        indices_test = list(set(range(y.shape[-2])))
        indices_train = indices_test
    else:
        indices_train = list(set(range(y.shape[-2])) - set(indices_test))

    plssvd = PLSSVD(n_components=n_components, seed=seed, center=center, scale=scale)
    try_devices(plssvd.fit)(x=x_train_rep[..., indices_train, :],y= y[..., indices_train, :])

    # s_train_rep_train =  try_devices(plssvd.transform)(z=x_train_rep[..., indices_train, :], direction="left")
    # s_test_rep_train =  try_devices(plssvd.transform)(z=x_test_rep[..., indices_train, :], direction="left")
    # s_train_rep_test =  try_devices(plssvd.transform)(z=x_train_rep[..., indices_test, :], direction="left")
    # s_test_rep_test =  try_devices(plssvd.transform)(z=x_test_rep[..., indices_test, :], direction="left")
    # s_y_test = try_devices(plssvd.transform)(z=y[..., indices_test, :], direction="right")
    
    x_y_test = try_devices(plssvd.inverse_transform)(
        z=try_devices(plssvd.transform)(z=y[..., indices_test, :], direction="right"),
        direction="left",
    ).cpu()
    
    
    output = xr.Dataset(
        data_vars={
            # "reliability_train.s": xr.DataArray(
            #     data=uncentered_correlation(s_train_rep_train, s_test_rep_train).cpu().numpy(),
            #     dims=("rank",)
            # ),
            # "reliability_test.s": xr.DataArray(
            #     data=uncentered_correlation(s_train_rep_test, s_test_rep_test).cpu().numpy(),
            #     dims=("rank",)
            # ),
            # "alignment_score.s": xr.DataArray(
            #     data=uncentered_correlation(s_train_rep_test, s_y_test).cpu().numpy(),
            #     dims=("rank",),
            # ),
            # "rank": xr.DataArray(
            #     data=1 + torch.arange(plssvd.n_components).numpy().astype(np.uint32),
            #     dims=("rank",),
            # ),
            "reliability_train.x": xr.DataArray(
                data=uncentered_correlation(x_train_rep[..., indices_train, :], x_test_rep[..., indices_train, :]).cpu().numpy(),
                dims=("neuroid",)
            ),
            "reliability_test.x": xr.DataArray(
                data=uncentered_correlation(x_train_rep[..., indices_test, :], x_test_rep[..., indices_test, :]).cpu().numpy(),
                dims=("neuroid",)
            ),
            "alignment_score.x": xr.DataArray(
                data=uncentered_correlation(x_train_rep[..., indices_test, :], x_y_test).cpu().numpy(),
                dims=("neuroid",),
            ),
            "neuroid": xr.DataArray(
                data=1 + torch.arange(x_train_rep.size(-1)).numpy().astype(np.uint32),
                dims=("neuroid",),
            ),
        }
    )

    return output
