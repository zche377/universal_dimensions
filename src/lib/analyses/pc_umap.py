import sys
from loguru import logger
import logging

#  to adjust raj's engineering preference
logger.remove()
logger.add(sys.stderr, level="INFO")

logging.basicConfig(level=logging.INFO)

import os
import io
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import xarray as xr
import umap
from nilearn import plotting, datasets, surface
from copy import deepcopy
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
sns.set(rc={"figure.dpi":600, "savefig.dpi": 600}, font_scale=1)
sns.set_style("ticks")
sns.set_palette("Set2")

from bonner.caching import cache, BONNER_CACHING_HOME
from bonner.datasets.allen2021_natural_scenes import load_stimuli

from lib.analyses.yield_models import yield_models
from lib.datasets.allen2021_natural_scenes import  load_activations_from_all
from lib.analyses.target_pca_model import target_pca_model
from lib.datasets import load_dataset
from lib.computation._pca import RankFilteredPCA



SEED = 11
N_COMPONENTS_EXTRA = 10
CS_THRESHOLD = 0.6
NSD_STIMULI_PATH = (
    Path(os.getenv("BONNER_DATASETS_HOME"))
    / "allen2021.natural_scenes"
    / "images"
)
IMAGENET_TRAIN_PATH = Path(
    "/home/zchen160/scratch4-mbonner5/shared/brainscore/brainio/image_russakovsky2014_ilsvrc2012/train/"
)


def _stats_to_sort(
    yid: str,
    basis: str,
    mode: str,
    roi: str,
    regression_type: str,
    node: str,
    model: str,
    sort_by: str,
    threshold_col: str,
    threshold: float,
    n_top: int,
) -> pd.DataFrame:
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
    if node is not None:
        df = df[df.node == node]
    if model is not None:
        df = df[df.model == model]
    
    if threshold_col is not None:
        df["above_threshold"] = df[threshold_col] > threshold
    else:
        df["above_threshold"] = True
    idx = np.array(np.argsort(np.argsort(df[sort_by])[::-1]))
    if n_top > 0:
        df["within_n_top"] = idx < n_top
    else:
        df["within_n_top"] = idx >= (len(df) + n_top)
    df["included"] = df["above_threshold"] & df["within_n_top"]
    return df


def _X(yielder, model_str, node_str, stim, df, pca):
    X = []
    for model in yield_models(yielder):
        if model_str is not None and model_str != model.identifier:
            continue
        for node in model.nodes:
            if node_str is not None and node_str != node:
                continue
            target_model = deepcopy(model)
            target_model.update_nodes([node])
            if stim == "shared":
                target = load_activations_from_all(target_model, mode="shared")
            elif stim == "all":
                target = load_activations_from_all(target_model, "all")
            elif isinstance(stim, int):
                target_subject_stimulus_ids = list(load_dataset(
                    dataset="NSD",
                    subjects=stim, 
                    roi="general",
                    stimulus_ids=None,
                )[stim]["stimulus_id"].values)
                target = load_activations_from_all(target_model, "all")
                target = target.sel(presentation=target["stimulus_id"].isin(target_subject_stimulus_ids))
            
            pca_model = target_pca_model(target_model, identifier="NSD")
            target = xr.DataArray(
                pca_model.transform(torch.from_numpy(target.values)).cpu().numpy(),
                dims=["presentation", "neuroid"],
                coords={
                    "stimulus_id": ("presentation", target["stimulus_id"].values),
                    "neuroid": [f"pc={i}" for i in np.arange(pca_model.n_components)+1],
                },
            )
            target.name = f"{target_model.identifier}.node={node}"
            
            temp_df = df[df.model == model.identifier]
            temp_df = temp_df[temp_df.node == node]
            target = target.sel(neuroid=np.array(temp_df.included))
            X.append(target.values)
            
    X = np.concatenate(X, axis=-1)
    if pca:
        X = torch.from_numpy(X)
        pca_model = RankFilteredPCA()
        pca_model.fit(X)
        X = pca_model.transform(X).cpu().numpy()
    
    return xr.DataArray(
        X,
        dims=["presentation", "neuroid"],
        coords={
            "stimulus_id": ("presentation", target.stimulus_id.values),
        },
    )


def _X_umap(yielder, yielder_name, basis, mode, roi, regression_type, node_str, model_str, sort_by, threshold_col, threshold, n_top, stim, pca,):
    df = _stats_to_sort(yielder_name, basis, mode, roi, regression_type, node_str, model_str, sort_by, threshold_col, threshold, n_top)
    X = _X(yielder, model_str, node_str, stim, df, pca)
    mapper = umap.UMAP(random_state=SEED)  
    X_umap = mapper.fit_transform(X.values)
    X_umap = xr.DataArray(
        X_umap,
        dims=["presentation", "neuroid"],
        coords={
            "stimulus_id": ("presentation", X.stimulus_id.values),
        },
    )
      
    return X_umap

def _plot_X_umap(X_umap, n_top, dir_path,):
    x, y = X_umap[:, 0], X_umap[:, 1]
    stimuli = load_stimuli()
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, stim_id in enumerate(X_umap.stimulus_id.values):
        stim_val = stimuli.sel(stimulus_id=stim_id).values
        image_box = OffsetImage(stim_val, zoom=0.05)
        image_box.image.axes = ax
        ab = AnnotationBbox(
            image_box,
            xy=(x[i], y[i]),
            xycoords="data",
            frameon=False,
            pad=0,
        )
        ax.add_artist(ab)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.axis("off")
    
    fig_path = BONNER_CACHING_HOME / dir_path / f"ntop={n_top}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path) 
    
    return None

def pc_umap(
    yielder: callable,
    yielder_name: str,
    basis: str,
    mode: str,
    roi: str,
    regression_type: str,
    sort_by: str,
    n_top: int = 100,
    stim: (int | str) = 0,
    threshold_col: str = None,
    threshold: int = None,
    node_str: str = None,
    model_str: str = None,
    pca: bool = False,
) -> None:
    dir_path = f"pc_umap/basis={basis}/regression={regression_type}/roi={roi}/mode={mode}/stim={stim}/yielder={yielder_name}/model={model_str}/node={node_str}/sort_by={sort_by}.threshold={threshold_col}={threshold}"
   
    if pca:
        dir_path += f".pca={pca}"
    cacher = cache(f"{dir_path}/ntop={n_top}.nc")
    X_umap = cacher(_X_umap)(yielder, yielder_name, basis, mode, roi, regression_type, node_str, model_str, sort_by, threshold_col, threshold, n_top, stim, pca)
    _plot_X_umap(X_umap, n_top, dir_path,)
    
    