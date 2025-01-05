import sys
from loguru import logger
import logging

#  to adjust raj's engineering preference
logger.remove()
logger.add(sys.stderr, level="INFO")

logging.basicConfig(level=logging.INFO)

from tqdm.auto import tqdm
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
from PIL import Image, ImageFilter
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

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

def _plot_X_umap(X_umap, n_top, dir_path, batch_size=32):
    x, y = X_umap[:, 0], X_umap[:, 1]
    stimuli = load_stimuli()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()
    
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Process stimuli in batches
    for batch_start in tqdm(range(0, len(X_umap.stimulus_id), batch_size), desc="Batch processing"):
        batch_end = min(batch_start + batch_size, len(X_umap.stimulus_id))
        batch_ids = X_umap.stimulus_id.values[batch_start:batch_end]
        
        # Load batch of stimuli
        batch_images = []
        batch_positions = []
        original_images = []
        
        for i, stim_id in enumerate(batch_ids):
            stim_val = stimuli.sel(stimulus_id=stim_id).values
            img = Image.fromarray(stim_val)
            batch_images.append(F.to_tensor(stim_val))
            batch_positions.append((x[batch_start + i], y[batch_start + i]))
            original_images.append(img)
        
        # Stack and move to device
        img_tensors = torch.stack(batch_images).to(device)
        
        # Perform inference
        with torch.no_grad():
            batch_outputs = model(img_tensors)
        
        # Process each image in the batch
        for i, (img, output) in enumerate(zip(original_images, batch_outputs)):
            for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                if label == 1 and score > 0.8:  # "person" category and confidence > 0.8
                    x1, y1, x2, y2 = map(int, box.tolist())
                    
                    # Crop and blur the face
                    face_region = img.crop((x1, y1, x2, y2))
                    blurred_face = face_region.filter(ImageFilter.GaussianBlur(10))
                    
                    # Paste the blurred face back into the image
                    img.paste(blurred_face, (x1, y1))
            
            # Convert the image back to a numpy array
            stim_val = np.array(img)
            
            # Add the image to the plot
            image_box = OffsetImage(stim_val, zoom=0.05)
            image_box.image.axes = ax
            ab = AnnotationBbox(
                image_box,
                xy=batch_positions[i],
                xycoords="data",
                frameon=False,
                pad=0,
            )
            ax.add_artist(ab)
    
    # Finalize the plot
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.axis("off")
    
    # Save the figure
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
    plot: bool = True,
) -> None:
    try:
        stim = int(stim)
    except:
        pass
    
    if n_top is None:
        n_top = 100
    else:
        try:
            n_top = int(n_top)
        except:
            pass
        
    if node_str is None or node_str != "node_loop":
        node_str = [node_str]
    else:
        match yielder_name:
            case "resnet50_imagenet1k_varied_tasks":
                node_str = (
                    ["relu"]
                    + [f"layer1.{i}.relu_2" for i in range(3)]
                    + [f"layer2.{i}.relu_2" for i in range(4)]
                    + [f"layer3.{i}.relu_2" for i in range(6)]
                    + [f"layer4.{i}.relu_2" for i in range(3)]
                )
            case  "untrained_resnet18_varied_seeds" | "resnet18_classification_imagenet1k_varied_seeds":
                node_str = (
                    ["relu"]
                    + [f"layer{i}.{j}.relu" for i in range(1, 5) for j in range(2) ]
                )
            case _:
                raise ValueError(
                    "not implemented"
                )
    
    for ns in node_str:
        logging.info(ns)
        dir_path = f"pc_umap/basis={basis}/regression={regression_type}/roi={roi}/mode={mode}/stim={stim}/yielder={yielder_name}/model={model_str}/node={ns}/sort_by={sort_by}.threshold={threshold_col}={threshold}"
    
        if pca:
            dir_path += f".pca={pca}"
        cacher = cache(f"{dir_path}/ntop={n_top}.nc")
        X_umap = cacher(_X_umap)(yielder, yielder_name, basis, mode, roi, regression_type, ns, model_str, sort_by, threshold_col, threshold, n_top, stim, pca)
        if plot:
            _plot_X_umap(X_umap, n_top, dir_path,)
    