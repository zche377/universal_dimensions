import logging

logging.basicConfig(level=logging.INFO)

import torch
import numpy as np

from bonner.computation.decomposition import PCA
from bonner.caching import cache

from lib.models import Model
from lib.computation._pca import RankFilteredPCA
from lib.datasets import load_stimulus_set
from lib.datasets.allen2021_natural_scenes import load_activations_from_all



@cache(
    "pca_model/fit={stimulus_id}/model={model_id}/hash={hash}.pkl",
    helper=lambda kwargs: {
        "stimulus_id": kwargs["identifier"],
        "model_id": kwargs["model"].identifier,
        "hash": kwargs["model"].hash,
    },
)
def target_pca_model(model: Model, identifier: str = "NSD") -> PCA:
    pca_model = RankFilteredPCA()
    
    match identifier:
        case "ImageNet":
            stimulus_set = load_stimulus_set(identifier="ImageNet", n_stimuli=[30000, 150000])
            pca_model.fit(torch.from_numpy(model(stimulus_set[0]).values))
        case "NSD":
            x = load_activations_from_all(model, "unshared")
            pca_model.fit(torch.from_numpy(x.values))
        case "NSD.z":
            x = load_activations_from_all(model, "unshared")
            pca_model = RankFilteredPCA(z_score=True)
            pca_model.fit(torch.from_numpy(x.values))
        case "NSD.nonfiltered":
            x = load_activations_from_all(model, "unshared")
            pca_model = RankFilteredPCA(filtered=False)
            pca_model.fit(torch.from_numpy(x.values))
        case "NSD.shared":
            x = load_activations_from_all(model, "shared")
            pca_model.fit(torch.from_numpy(x.values))
        case "NSD.all":
            x = load_activations_from_all(model, "all")
            pca_model.fit(torch.from_numpy(x.values))
    return pca_model
