import sys
from loguru import logger
import logging

#  to adjust raj's engineering preference
logger.remove()
logger.add(sys.stderr, level="INFO")

logging.basicConfig(level=logging.INFO)

import os
import re
from copy import deepcopy
from tqdm.auto import tqdm

import torch
import numpy as np
import xarray as xr

from bonner.caching import cache, BONNER_CACHING_HOME
from bonner.datasets.allen2021_natural_scenes import N_SUBJECTS
from bonner.computation.metrics import pearson_r
from bonner.computation.regression import LinearRegression, create_splits
from bonner.computation.xarray import align_source_to_target

from lib.analyses.brain_similarity import ROIS
from lib.analyses.target_pca_model import target_pca_model
from lib.datasets.allen2021_natural_scenes import get_shared_stimulus_ids, load_activations_from_all
from lib.models import Model
from lib.datasets import load_dataset
from lib.analyses.yield_models import yield_models
from lib.scorers import RegressionScorer, RidgeGCVRegression



N_FOLDS = 5


def _model_yielder(yielder: callable,) -> callable:
    def _yielder(target_identifier: str = None,):
        for model in yield_models(yielder):
            if target_identifier is not None:
                match yielder.__name__:
                    case "resnet18_classification_imagenet1k_varied_seeds":
                        target_seed = re.search(".*seed=(.*)", target_identifier)[1]
                        predictor_seed = re.search(".*seed=(.*)", model.identifier)[1]
                        if target_seed == predictor_seed:
                            continue
                    case _:
                        if model.identifier == target_identifier:
                            continue
            yield model.identifier, load_activations_from_all(model, "shared")
    return _yielder


def _brain_yielder(roi: str,) -> callable:
    def _yielder(target_identifier: int = None,):
        for subject in range(N_SUBJECTS):
            if target_identifier is not None and subject == target_identifier:
                continue
            yield subject, load_dataset(
                dataset="NSD",
                subjects=subject, 
                roi=roi,
                stimulus_ids=list(get_shared_stimulus_ids()),
            )[subject]
    return _yielder


def _to_tensor(x: xr.DataArray) -> torch.Tensor:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return torch.tensor(x.values).to(device)


def _within_basis_score(
    target: xr.DataArray,
    predictor: xr.DataArray,
    scorer: RegressionScorer,
) -> xr.DataArray:
    return scorer(predictor=predictor, target=target)


def within_basis_score(
    target: xr.DataArray,
    target_identifier: str | int,
    predictor_yielder: callable,
    scorer: RegressionScorer,
    basis: str,
    mode: str,
    roi: str = None,
    sub_identifiers: dict[str, str] = {},
) -> xr.DataArray:
    scores = []
    for predictor_identifier, predictor in predictor_yielder(target_identifier):
        path = (
            f"universality_index"
            f"/basis={basis}"
            f"/type=within_basis"
            f"/scorer={scorer.identifier}"
        )
        if basis == "voxel":
            path += f"/roi={roi}"
        path += (
            f"/mode={mode}"
            f"/target_identifier={target_identifier}"
        )
        for key, value in sub_identifiers.items():
            path += f"/{key}={value}"
        path += f"/predictor_identifier={predictor_identifier}.nc"
        cacher = cache(
            path,
        )
        
        cacher(_within_basis_score)(target, predictor, scorer)
        
    return None
    

def _cross_basis_score(
    target: xr.DataArray,
    predictor_yielder: callable,
    regression: callable,
    regression_type: str,
    general_basis: str,
):
    splits = create_splits(
        n=target.shape[-2], 
        n_folds=N_FOLDS,
        shuffle=True, 
        seed=11,
    )
    neuroid = target["neuroid"].values
    scores = []
    for fold, indices_test in enumerate(splits):
        indices_train = np.setdiff1d(
            np.arange(target.shape[-2]),
            np.array(indices_test)
        )
        predicteds = []
        for _, predictor in predictor_yielder():
            # to use cuda linear regression with model features as predictor
            non_zero_std_index = np.std(predictor.values, axis=0) != 0
            predictor = predictor[:, non_zero_std_index]
            sub_target = align_source_to_target(
                source=target.sel(presentation=target["stimulus_id"].isin(predictor["stimulus_id"].values)),
                target=predictor,
                sample_coord="stimulus_id",
                sample_dim="presentation",
            ).transpose("presentation", "neuroid")
            
            regressor = regression()
            regressor.fit(
                x=_to_tensor(predictor[..., indices_train,:]),
                y=_to_tensor(sub_target[..., indices_train, :]),
            )
            predicted = regressor.predict(_to_tensor(predictor[..., indices_test, :])).cpu()
            predicteds.append(predicted)
        temp_scores = []
        p1, p2 = np.tril_indices(len(predicteds), k=-1)
        for i in range(len(predicteds)):
            temp_scores.append(pearson_r(
                predicteds[p1[i]], predicteds[p2[i]],
            ))
        temp_scores = torch.stack(temp_scores)
        if temp_scores.dim() == 1:
            temp_scores = temp_scores.unsqueeze(-1)
        scores.append(temp_scores.mean(dim=0))
    scores = xr.DataArray(
        name="score",
        data=torch.stack(scores).numpy(),
        dims=("fold", "neuroid"),
        coords={
            "neuroid": neuroid,
        },
        attrs={
            "regression": regression_type,
            "n_folds": 5,
            "metric": "correlation",
            "shuffle": 1,
            "seed": 11,
        }
    )
    return scores
    

def cross_basis_score(
    target: xr.DataArray,
    target_identifier: str | int,
    predictor_yielder: callable,
    regression: callable,
    basis: str,
    mode: str,
    node: str,
    roi: str,
    regression_type: str,
    yielder_id: str = None,
):
    match basis:
        case "voxel":
            cacher = cache(
                f"universality_index"
                f"/basis={basis}"
                f"/type=cross_basis"
                f"/regression={regression_type}"
                f"/roi={roi}"
                f"/yielder={yielder_id}"
                f"/target_identifier={target_identifier}.nc"
            )
            cacher(_cross_basis_score)(target, predictor_yielder, regression, regression_type, "brain")
        case _:
            cacher = cache(
                f"universality_index"
                f"/basis={basis}"
                f"/type=cross_basis"
                f"/regression={regression_type}"
                f"/roi={roi}"
                f"/mode={mode}"
                f"/target_identifier={target_identifier}"
                f"/node={node}.nc"
            )
            cacher(_cross_basis_score)(target, predictor_yielder, regression, regression_type, "model")
        

def universality_index(
    basis: str, # "activation_channel", "activation_pc", "voxel"
    mode: str, 
    cross_basis: bool = False,
    model: Model = None,
    model_predictor_yielder: callable = None,
    subject: int = None,
    roi: str | list[str] = "general",
    regression_type: str = "ols",
) -> xr.DataArray:
    """
    metrics in the class of universality index:
    
    universality - basis model, cross basis false
    subject reliability (of brain on model space) - basis model, cross basis true
    universality of model on voxel space - basis voxel, cross basis true
    subject reliability of voxel  - basis voxel, cross basis false
    
    """
    assert basis in {"activation_channel", "activation_pc", "voxel", "activation_channel_demeaned"}
    assert mode in {"NSD.shared", "NSD", "RP", "NSD.z", "NSD.nonfiltered"}
    
    general_basis = "brain" if basis == "voxel" else "model"
    
    if isinstance(roi, list):
        rois = tqdm(roi, desc="roi", leave=False)
    elif roi == "default_list":
        rois = tqdm(ROIS, desc="roi", leave=False)
    else:
        rois = [roi]
        
    for roi in rois:
        match (general_basis, cross_basis):
            case ("model", False): # canonical strength
                assert model is not None
                within_basis_yielder = _model_yielder(model_predictor_yielder)
            case ("model", True):
                assert model is not None
                assert roi is not None
                brain_yielder = _brain_yielder(roi,)
            case ("brain", True):
                assert roi is not None
                model_yielder = _model_yielder(model_predictor_yielder)
            case ("brain", False): # subject reliability of voxel
                assert roi is not None
                within_basis_yielder = _brain_yielder(roi,)
        
        scorer = RegressionScorer(regression=regression_type, n_folds=N_FOLDS)
        match regression_type:
            case "ols":
                regression = LinearRegression
            case "ridgecv":
                regression = RidgeGCVRegression
        
        scores = []
        
        if general_basis == "brain" and not cross_basis:
            coords = {"subject": []}
            for target_identifier, brain_target in tqdm(within_basis_yielder(roi), desc="subject", leave=False):
                logging.info(target_identifier)
                temp_score = within_basis_score(
                    brain_target,
                    target_identifier,
                    within_basis_yielder,
                    scorer,
                    basis,
                    mode,
                    roi=roi
                )
                scores.append(temp_score)
                coords["subject"].extend([target_identifier] * len(temp_score))
        elif general_basis == "brain" and cross_basis:
            for subj in tqdm([subject], desc="subject", leave=False):
                brain_target = load_dataset(
                    dataset="NSD",
                    subjects=subj, 
                    roi=roi,
                    stimulus_ids=list(get_shared_stimulus_ids()),
                )[subj]
                temp_score = cross_basis_score(
                    brain_target,
                    subj,
                    model_yielder,
                    regression,
                    basis,
                    None,
                    None,
                    roi,
                    regression_type,
                    yielder_id=model_predictor_yielder.__name__
                )
        else:
            coords = {"node": []}
            for node in tqdm(model.nodes, desc="node", leave=False):
                logging.info(node)
                
                if not cross_basis:
                    cache_path = (
                        BONNER_CACHING_HOME
                        / "universality_index"
                        / f"basis={basis}"
                        / f"type=within_basis"
                        / f"scorer={scorer.identifier}"
                        / f"mode={mode}"
                        / f"target_identifier={model.identifier}"
                        / f"node={node}"
                    )
                    match model_predictor_yielder.__name__:
                        case "resnet50_imagenet1k_varied_tasks":
                            num_predictor = 8
                        case "classification_imagenet1k_varied_architectures":
                            num_predictor = 18
                        case _:
                            num_predictor = 19
                        
                    if cache_path.exists() and sum(1 for entry in os.listdir(cache_path) if os.path.isfile(os.path.join(cache_path, entry))) == num_predictor:
                        continue
                
                target_model = deepcopy(model)
                target_model.update_nodes([node])
                model_target = load_activations_from_all(target_model, "shared")
                
                if basis == "activation_pc":
                    pmode = mode if mode != "RP" else "NSD"
                    pca_model = target_pca_model(target_model, identifier=pmode)
                    model_target = xr.DataArray(
                        pca_model.transform(torch.from_numpy(model_target.values)).cpu().numpy(),
                        dims=["presentation", "neuroid"],
                        coords={
                            "stimulus_id": ("presentation", model_target["stimulus_id"].values),
                            "neuroid": np.arange(pca_model.n_components)+1,
                        },
                    )
                    model_target.name = f"{target_model.identifier}.node={node}"
                elif basis == "activation_channel_demeaned":
                    model_target = model_target - model_target.mean("presentation")
                
                if not cross_basis:
                    temp_score = within_basis_score(
                        model_target,
                        target_model.identifier,
                        within_basis_yielder,
                        scorer,
                        basis,
                        mode,
                        sub_identifiers={"node": node},
                    )
                else:
                    temp_score = cross_basis_score(
                        model_target,
                        target_model.identifier,
                        brain_yielder,
                        regression,
                        basis,
                        mode,
                        node,
                        roi,
                        regression_type,
                    )
    return None   
            