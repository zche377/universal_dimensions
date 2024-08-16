import sys
from loguru import logger
import logging

#  to adjust raj's engineering preference
logger.remove()
logger.add(sys.stderr, level="INFO")

logging.basicConfig(level=logging.INFO)

from copy import deepcopy
from tqdm.auto import tqdm

import os
import torch
import numpy as np
import xarray as xr

from bonner.computation.metrics import pearson_r
from bonner.caching import cache, BONNER_CACHING_HOME
from bonner.datasets.allen2021_natural_scenes import N_SUBJECTS
from bonner.computation.regression import LinearRegression, create_splits, Regression
from bonner.computation.xarray import align_source_to_target
from bonner.models.hooks import Hook, GlobalMaxpool, GlobalAveragePool, RandomProjection

from lib.analyses.brain_similarity import ROIS
from lib.analyses.target_pca_model import target_pca_model
from lib.datasets.allen2021_natural_scenes import get_shared_stimulus_ids, load_activations_from_all
from lib.models import Model
from lib.datasets import load_dataset
from lib.scorers import RidgeGCVRegression


def _to_tensor(x: xr.DataArray) -> torch.Tensor:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return torch.tensor(x.values).to(device)


def _stimulus_ids(mode: str) -> list[str]:
    match mode:
        case "all":
            return None
        case "shared" | "same" | "RP":
            return list(get_shared_stimulus_ids())
        case _:
            raise ValueError(f"Unknown mode: {mode}")


def _model_space_score(
    subject_target: xr.DataArray,
    avg_beta: xr.Dataset,
    betas: dict[str, xr.Dataset],
    regression: Regression,
    splits: list[np.ndarray],
) -> xr.DataArray:
    br = []
    avg_beta = _to_tensor(align_source_to_target(
        source=avg_beta, target=subject_target,
        sample_coord="stimulus_id", sample_dim="presentation",
    ).transpose("presentation", "neuroid"))
    subject_target_cuda = _to_tensor(subject_target)
    
    for indices_test in splits:
        indices_train = np.setdiff1d(
            np.arange(avg_beta.shape[-2]),
            np.array(indices_test)
        )
        regressor = regression()
        regressor.fit(
            x=avg_beta[..., indices_train, :],
            y=subject_target_cuda[..., indices_train, :],
        )
        
        fold_target = subject_target[..., indices_test, :]
        temp_br = []
        reps = [(0, 1), (0, 2), (1, 2)]
        for rep in reps:
            rep_target = fold_target.sel(presentation=fold_target["stimulus_id"].isin(betas[np.max(rep)]["stimulus_id"].values))
            beta_0 = align_source_to_target(
                source=betas[rep[0]], target=rep_target,
                sample_coord="stimulus_id", sample_dim="presentation",
            ).transpose("presentation", "neuroid")
            beta_1 = align_source_to_target(
                source=betas[rep[1]], target=rep_target,
                sample_coord="stimulus_id", sample_dim="presentation",
            ).transpose("presentation", "neuroid")
            
            temp_br.append(pearson_r(
                regressor.predict(_to_tensor(beta_0)),
                regressor.predict(_to_tensor(beta_1))
            ).cpu())
        br.append(torch.stack(temp_br))
    reps = [f"{i}".replace(" ", "") for i in reps]
    br = torch.stack(br)
    if br.dim() == 2:
        br = br.unsqueeze(-1)
    return xr.DataArray(
        name="score",
        data=br.numpy(),
        dims=["fold", "rep", "neuroid"],
        coords={
            "fold": list(range(5)),
            "rep": reps,
            "neuroid": subject_target["neuroid"].values,
        },
        attrs={
            "regression": "ols",
            "n_folds": 5,
            "metric": "correlation",
            "shuffle": 1,
            "seed": 11,
        }
    )


def _model_space(
    target_model: Model, 
    roi: str,
    regression: callable,
    n_folds: int,
    mode: str,
    basis: str,
    regression_type: str
) -> xr.DataArray:
    BR = []
    node = target_model.nodes[0] if len(target_model.nodes) == 1 else "all"
    
    match mode:
        case "shared" | "same" | "RP":
            lmode = "shared"
        case _:
            lmode = mode
            
    target = load_activations_from_all(target_model, mode=lmode)
    stimulus_ids = target["stimulus_id"].values
            
    if basis == "activation_pc":      
        match mode:
            case "all":
                pca_identifier = "ImageNet"
            case "shared" | "RP":
                pca_identifier = "NSD"
            case "same":
                pca_identifier = "NSD.shared"  
        pca_model = target_pca_model(target_model, identifier=pca_identifier)
        target = pca_model.transform(torch.from_numpy(target.values)).cpu().numpy()
        neuroids = np.arange(pca_model.n_components)+1
        
        target = xr.DataArray(
            target,
            dims=["presentation", "neuroid"],
            coords={
                "stimulus_id": ("presentation", stimulus_ids),
                "neuroid": neuroids,
            },
        )
        target.name = f"{target_model.identifier}.node={node}"
    
    for subject in range(N_SUBJECTS):  
        avg_beta = load_dataset(
            dataset="NSD",
            subjects=subject, 
            roi=roi,
            stimulus_ids=_stimulus_ids(mode),
        )[subject]
        subject_target = target.sel(presentation=target["stimulus_id"].isin(avg_beta["stimulus_id"].values))
        betas = {
            rep: load_dataset(
                    dataset="NSD",
                    subjects=subject, 
                    roi=roi,
                    stimulus_ids=_stimulus_ids(mode),
                    average_across_reps=False,
                    repetition=rep,
                )[subject]
            for rep in range(3)
        } 
        splits = create_splits(
            n=avg_beta.shape[-2], 
            n_folds=n_folds,
            shuffle=True, 
            seed=11,
        )
        cacher = cache(
            "brain_reliability"
            f"/basis={basis}"
            f"/scorer={regression_type}"
            f"/roi={roi}"
            f"/mode={mode}"
            f"/model={target_model.identifier}"
            f"/node={node}"
            f"/subject={subject}.nc",
        )
        BR.append(cacher(_model_space_score)(
            subject_target,
            avg_beta,
            betas,
            regression,
            splits,
        ).mean("fold").mean("rep"))
    BR = xr.concat(BR, dim="subject").mean("subject")
    return BR 


@cache("brain_reliability/basis=voxel/roi={roi}/mode=same.nc", )
def _voxel_space(
    roi: str,
) -> xr.DataArray:
    reps = [(0, 1), (0, 2), (1, 2)]
    str_reps = [f"{i}".replace(" ", "") for i in reps]
    br, subject_coord = [], []
    for subject in range(N_SUBJECTS):
        betas = {
            rep: load_dataset(
                    dataset="NSD",
                    subjects=subject, 
                    roi=roi,
                    stimulus_ids=_stimulus_ids("same"),
                    average_across_reps=False,
                    repetition=rep,
                )[subject]
            for rep in range(3)
        }
        temp_br = []
        for rep in reps:
            beta_0 = betas[rep[0]].sel(presentation=betas[rep[0]]["stimulus_id"].isin(betas[np.max(rep)]["stimulus_id"].values))
            beta_1 = betas[rep[1]].sel(presentation=betas[rep[1]]["stimulus_id"].isin(betas[np.max(rep)]["stimulus_id"].values))
            beta_1 = align_source_to_target(
                source=beta_1, target=beta_0,
                sample_coord="stimulus_id", sample_dim="presentation",
            ).transpose("presentation", "neuroid")
            temp_br.append(pearson_r(
                _to_tensor(beta_0),
                _to_tensor(beta_1),
            ))
        neuroid = betas[0]["neuroid"].values
        temp_br = xr.DataArray(
            name="score",
            data=torch.stack(temp_br).cpu().numpy(),
            dims=["rep", "neuroid"],
            coords={
                "rep": str_reps,
                "neuroid": neuroid,
                "subject": ("neuroid", [subject] * len(neuroid)),
            },
            attrs={
                "n_folds": 5,
                "metric": "correlation",
                "shuffle": 1,
                "seed": 11,
            }
        )
        br.append(temp_br)
        subject_coord.extend([subject] * len(temp_br["neuroid"])) 
    reps = [f"{i}".replace(" ", "") for i in reps]
    br = xr.concat(br, dim="neuroid").assign_coords({"subject": ("neuroid", subject_coord)})
    return br


def brain_reliability(
    model: Model, 
    roi: str | list[str] = "general",
    mode: str = "all",
    basis: str = "activation_pc",
    regression_type: str = "ols",
) -> xr.DataArray: 
    assert mode in {"all", "shared", "same", "RP"}
    BR, node_list, roi_list = [], [], []
    
    match regression_type:
        case "ols":
            regression = LinearRegression
        case "ridgecv":
            regression = RidgeGCVRegression
    n_folds = 5
    
    if isinstance(roi, list):
        rois = tqdm(roi, desc="roi", leave=False)
    elif roi == "default_list":
        rois = tqdm(ROIS, desc="roi", leave=False)
    else:
        rois = [roi]
    
    nodes = model.nodes
    if basis == "voxel":
        for roi in rois:
            temp_br = _voxel_space(roi).mean("rep")
            BR.append(temp_br)
            roi_list.extend([roi] * len(temp_br))
        BR = xr.concat(BR, dim="neuroid")
        BR = BR.assign_coords({"roi": ("neuroid", roi_list),})
    else:
        for roi in rois:
            for node in tqdm(nodes, desc="node", leave=False):
                logging.info(node)
                
                cache_path = (
                    BONNER_CACHING_HOME
                    / "brain_reliability"
                    / f"basis={basis}"
                    / f"scorer={regression_type}"
                    / f"roi={roi}"
                    / f"mode={mode}"
                    / f"model={model.identifier}"
                    / f"node={node}"
                )
                num_predictor = N_SUBJECTS  
                if cache_path.exists() and sum(1 for entry in os.listdir(cache_path) if os.path.isfile(os.path.join(cache_path, entry))) == num_predictor:
                    continue
                
                target_model = deepcopy(model)
                if node != "all":
                    target_model.update_nodes([node])
                temp_br = _model_space(target_model, roi, regression, n_folds, mode, basis, regression_type,)
    
    return None      

