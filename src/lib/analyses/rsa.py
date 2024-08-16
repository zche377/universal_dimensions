import sys
from loguru import logger
import logging

#  to adjust raj's engineering preference
logger.remove()
logger.add(sys.stderr, level="INFO")

logging.basicConfig(level=logging.INFO)

from copy import deepcopy
from tqdm.auto import tqdm

import torch
import numpy as np
import xarray as xr
import pandas as pd

from bonner.caching import cache, BONNER_CACHING_HOME
from bonner.datasets.allen2021_natural_scenes import N_SUBJECTS
from bonner.computation.metrics import pearson_r, spearman_r
from bonner.computation.xarray import align_source_to_target_coord

from lib.analyses.target_pca_model import target_pca_model
from lib.datasets.allen2021_natural_scenes import get_shared_stimulus_ids, load_activations_from_all
from lib.models import Model
from lib.datasets import load_dataset


def _stimulus_ids(mode: str) -> list[str]:
    ssid = list(get_shared_stimulus_ids())
    ssid.sort()
    match mode:
        case "shared":
            return ssid
        case "shared.train":
            return ssid[::2]
        case "shared.test":
            return ssid[1::2]
        case _:
            raise ValueError(f"Unknown mode: {mode}")

def _stats_to_sort(
    yid: str,
    mode: str,
    roi: str,
    node: str,
    model: str,
    sort_by: str,
    n_top: int,
    basis: str = "activation_pc",
    regression_type: str = "ridgecv",
) -> pd.DataFrame:
    match mode:
        case "shared" | "shared.train" | "shared.test":
            mode = "NSD"
    mode_dict = {
        "NSD": "shared",
        "NSD.shared": "same"
    }
    root = BONNER_CACHING_HOME / "summary_results"
    x = [
        xr.open_dataarray(root / f"brain_similarity/brain_similarity.regression={regression_type}.basis={basis}.roi={roi}.mode={mode_dict[mode]}.yielder={yid}.nc").to_dataset(name="brain_similarity"),
        xr.open_dataarray(root / f"universality_index/universality_index.basis={basis}.type=within_basis.regression={regression_type}.mode={mode}.yielder={yid}.nc").to_dataset(name="within_basis_ui"),
    ]
    x = xr.merge(x)
    df = x.to_dataframe()
    if basis == "activation_pc":
        df["pc"] = df.index.values
    df = df.reset_index()
    df["model"] = [str(s) for s in df["model"]]
    df["node"] = [str(s) for s in df["node"]]
    df["brain_similarity"] = [float(s) for s in df["brain_similarity"]]
    df["within_basis_ui"] = [float(s) for s in df["within_basis_ui"]]
    df = df[df.node == node]
    df = df[df.model == model]
    
    idx = np.array(np.argsort(np.argsort(df[sort_by])[::-1]))
    if isinstance(n_top, float):
        n_top = max(1, int(len(idx) * n_top))
    if n_top > 0:
        df["excluded"] = idx >= n_top
    else:
        df["excluded"] = idx < (len(df) + n_top)
    return df

def _compute_model_rdm(
    mode: str, 
    model: Model, 
    node: str,
    sort_by: str = None,
    n_top: (int | float) = .01,
    yielder_id: str = None,
    roi: str = None,
) -> xr.DataArray:
    target_model = deepcopy(model)
    if node != "all":
        target_model.update_nodes([node])
    target = load_activations_from_all(target_model, mode="shared")
    sid = _stimulus_ids(mode)
    target = align_source_to_target_coord(
        source=target.sel(presentation=target["stimulus_id"].isin(sid)),
        target_coord=sid,
        sample_coord="stimulus_id",
        sample_dim="presentation",
    ).transpose("presentation", "neuroid")
    neuroid = target.neuroid.values
    
    if sort_by is not None:
        df = _stats_to_sort(yielder_id, mode, roi, node, model.identifier, sort_by, n_top)
        pca_model = target_pca_model(target_model, identifier="NSD")
        target = xr.DataArray(
            pca_model.transform(torch.from_numpy(target.values)).cpu().numpy(),
            dims=["presentation", "neuroid"],
            coords={
                "stimulus_id": ("presentation", target["stimulus_id"].values),
                "neuroid": [f"pc={i}" for i in np.arange(pca_model.n_components)+1],
            },
        )
        
        target.loc[{"neuroid": np.array(df.excluded)}] = 0
        target = xr.DataArray(
            pca_model.inverse_transform(torch.from_numpy(target.values)).cpu().numpy(),
            dims=["presentation", "neuroid"],
            coords={
                "stimulus_id": ("presentation", target["stimulus_id"].values),
                "neuroid": neuroid,
            },
        )
        
    target = torch.from_numpy(target.values).T
    
    return xr.DataArray(
        1 - pearson_r(target, return_diagonal=False).cpu().numpy(),
        dims=("stimulus1", "stimulus2"),
        coords={"stimulus1": sid, "stimulus2": sid},
    )

@cache(
    "rdm/domain=brain/dataset=allen2021.natural_scenes/mode={mode}/roi={roi}.subject={subject}.nc",
)
def _compute_brain_rdm(mode: str, subject: int, roi: str = "general",) -> xr.DataArray:
    sid = _stimulus_ids(mode)
    beta = load_dataset(
        dataset="NSD",
        subjects=subject, 
        roi=roi,
        stimulus_ids=sid,
    )[subject]
    
    beta = align_source_to_target_coord(
        source=beta,
        target_coord=sid,
        sample_coord="stimulus_id",
        sample_dim="presentation",
    ).transpose("presentation", "neuroid")
    
    beta = torch.from_numpy(beta.values).T
    return xr.DataArray(
        1 - pearson_r(beta, return_diagonal=False).cpu().numpy(),
        dims=("stimulus1", "stimulus2"),
        coords={"stimulus1": sid, "stimulus2": sid},
    )

def _flattend_tril(x: torch.Tensor) -> torch.Tensor:
    lower_tri_indices = torch.tril_indices(x.size(0), x.size(1), offset=-1)
    return x[lower_tri_indices[0], lower_tri_indices[1]]
    
def _compute_rsa_score(
    mode,
    model,
    nodes,
    subjects,
    sort_id: str,
    **kwargs,
) -> xr.DataArray:
    subject_scores = []
    for subject in subjects:
        node_scores = []
        for node in nodes:
            cacher = cache(
                "rdm"
                "/domain=model"
                f"/mode={mode}"
                f"/sort_by={sort_id}"
                f"/model={model.identifier}"
                f"/node={node}.nc",
            )
            model_rdm = cacher(_compute_model_rdm)(mode, model, node, **kwargs)
            brain_rdm = _compute_brain_rdm(mode, subject=subject)
            score = spearman_r(
                _flattend_tril(torch.from_numpy(model_rdm.values)), 
                _flattend_tril(torch.from_numpy(brain_rdm.values))
            ).cpu().numpy()
            node_scores.append(xr.DataArray(score).assign_coords({"node": node}))
        subject_scores.append(xr.concat(node_scores, dim="node").assign_coords({"subject": subject}))
    return xr.concat(subject_scores, dim="subject")

def rsa(
    model: Model, 
    mode: str = "shared",
    roi: str | list[str] = "general",
    sort_by: str = None,
    n_top: (int | float) = .01,
    yielder_id: str = None,
) -> xr.DataArray:
    rois = tqdm(roi, desc="roi", leave=False) if isinstance(roi, list) else [roi]
    sort_id = sort_by
    if sort_by is not None:
        sort_id += f".n_top={n_top}"
        
    for roi in rois:
        train_scores = _compute_rsa_score(f"{mode}.train", model, model.nodes, range(N_SUBJECTS), sort_id="None",).mean(dim="subject")
        best_node = train_scores.node.isel(node=train_scores.argmax()).item()
        cacher = cache(
            "rsa_score"
            f"/roi={roi}"
            f"/mode={mode}.test"
            f"/sort_by={sort_id}"
            f"/model={model.identifier}"
            f"/node=best.nc",
        )
        cacher(_compute_rsa_score)(f"{mode}.test", model, [best_node], range(N_SUBJECTS), sort_id=sort_id, sort_by=sort_by, n_top=n_top, yielder_id=yielder_id, roi=roi)
        