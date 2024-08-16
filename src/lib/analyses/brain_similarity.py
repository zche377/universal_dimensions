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

from bonner.caching import cache
from bonner.datasets.allen2021_natural_scenes import N_SUBJECTS
from bonner.models.hooks import Hook, GlobalMaxpool, GlobalAveragePool, RandomProjection

from lib.analyses.target_pca_model import target_pca_model
from lib.datasets.allen2021_natural_scenes import get_shared_stimulus_ids, load_activations_from_all
from lib.models import Model
from lib.scorers import RegressionScorer
from lib.datasets import load_dataset, load_stimulus_set


ROIS = [
    f"streams-{stream}" for stream in [
        "lateral",
        "parietal",
        "ventral",
]]


def _stimulus_ids(mode: str) -> list[str]:
    match mode:
        case "all":
            return None
        case "shared" | "same" | "spatial" | "RP" |"shared.nonfiltered" | "shared.z":
            return list(get_shared_stimulus_ids())
        case _:
            raise ValueError(f"Unknown mode: {mode}")


def _score(
    scorer: RegressionScorer, target: xr.DataArray, predictor: xr.DataArray
) -> xr.DataArray:
    return scorer(predictor=predictor, target=target)


def _brain_similarity(
    target_model: Model, 
    roi: str,
    scorer: RegressionScorer,
    mode: str,
    basis: str,
    subjects: list[int],
) -> xr.DataArray:
    BS = []
    node = target_model.nodes[0] if len(target_model.nodes) == 1 else "all"
    
    if basis == "activation_pc":
        match mode:
            case "all":
                pca_identifier = "ImageNet"
            case "shared" | "spatial" | "RP":
                pca_identifier = "NSD"
            case "shared.nonfiltered":
                pca_identifier = "NSD.nonfiltered"
            case "shared.z":
                pca_identifier = "NSD.z"
            case "same":
                pca_identifier = "NSD.shared"
        pca_model = target_pca_model(target_model, identifier=pca_identifier)
    
    match mode:
        case "shared" | "same" | "spatial" | "RP" | "shared.nonfiltered" | "shared.z":
            lmode = "shared"
        case _:
            lmode = mode
    target = load_activations_from_all(target_model, mode=lmode)
    
    if mode == "spatial":
        target_model.update_hooks({})
        n_presentation, n_channel = target.shape
        stimulus_set = load_stimulus_set(identifier="NSD", mode="shared")
        target = target_model(stimulus_set)
        
    stimulus_ids = target["stimulus_id"].values
    
    if basis == "activation_pc":
        if mode == "spatial":
            temp = torch.from_numpy(target.values)
            temp = temp.view(n_presentation, n_channel, -1).swapaxes(-2, -1)
            target = xr.DataArray(temp.numpy())
        target = pca_model.transform(torch.from_numpy(target.values)).cpu()
        if mode == "spatial":
            target = target.flatten(1).numpy()
            target = xr.DataArray(
                target,
                dims=["presentation", "neuroid"],
                coords={
                    "stimulus_id": ("presentation", stimulus_ids),
                    "neuroid": np.arange(target.shape[-1]),
                },
            )
        else:
            neuroids = np.arange(pca_model.n_components)+1
            target = xr.DataArray(
                target.numpy(),
                dims=["presentation", "neuroid"],
                coords={
                    "stimulus_id": ("presentation", stimulus_ids),
                    "neuroid": neuroids,
                },
            )
        target.name = f"{target_model.identifier}.node={node}"
    elif basis == "activation_channel_demeaned":
        target = target - target.mean("presentation")
    
    for subject in subjects:
        beta = load_dataset(
            dataset="NSD",
            subjects=subject, 
            roi=roi,
            stimulus_ids=_stimulus_ids(mode),
        )[subject]
        subject_target = target.sel(presentation=target["stimulus_id"].isin(beta["stimulus_id"].values))
        cacher = cache(
            f"brain_similarity"
            f"/basis={basis}"
            f"/scorer={scorer.identifier}"
            f"/roi={roi}"
            f"/mode={mode}"
            f"/model={target_model.identifier}"
            f"/node={node}"
            f"/subject={subject}.nc",
        )
        
        match basis:
            case "activation_channel" | "activation_pc" | "activation_channel_demeaned":
                cacher(_score)(scorer, target=subject_target, predictor=beta)
            case "voxel":
                cacher(_score)(scorer, target=beta, predictor=subject_target)
                
    return None


def brain_similarity(
    model: Model, 
    roi: str | list[str] = "general",
    mode: str = "same",
    basis: str = "activation_pc",
    regression_type: str = "ols",
    subjects: (int | list[int]) = list(range(N_SUBJECTS)),
) -> xr.DataArray: 
    assert mode in {"all", "shared", "same", "spatial", "RP", "shared.z", "shared.nonfiltered"}
    BS, node_list, roi_list = [], [], []
    if isinstance(subjects, int):
        subjects = [subjects]
    match regression_type:
        case "ols":
            scorer = RegressionScorer(regression="ols", n_folds=5)
        case "ridgecv":
            scorer = RegressionScorer(regression="ridgecv", n_folds=5)
    nodes = model.nodes
    
    if isinstance(roi, list):
        rois = tqdm(roi, desc="roi", leave=False)
    elif roi == "default_list":
        rois = tqdm(ROIS, desc="roi", leave=False)
    else:
        rois = [roi]
    
    for roi in rois:
        for node in tqdm(nodes, desc="node", leave=False):
            logging.info(node)
            target_model = deepcopy(model)
            if node != "all":
                target_model.update_nodes([node])
            temp_bs = _brain_similarity(target_model, roi, scorer, mode, basis, subjects)
    
    return None

