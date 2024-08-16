import torch.nn as nn
import xarray as xr

from bonner.models.datapipes import create_image_datapipe
from bonner.models.features import (
    extract_features,
    concatenate_features,
    flatten_features,
)
from bonner.models.hooks import Hook

from lib.utils import hash_string
from lib.datasets import StimulusSet


class Model:
    def __init__(
        self,
        model: nn.Module,
        preprocess: callable,
        identifier: str,
        nodes: list[str],
        hooks: dict[str, Hook] = {},
    ) -> None:
        self.model = model
        self.preprocess = preprocess
        self.identifier = identifier
        self.nodes = nodes
        self.hooks = hooks
        self.hash = self._hash_node_list()

    def __call__(
        self,
        stimulus_set: StimulusSet,
        *,
        batch_size: int = 256,
    ) -> xr.DataArray:
        datapipe = create_image_datapipe(
            datapipe=stimulus_set,
            indices=stimulus_set.metadata.index.to_list(),
            batch_size=batch_size,
            preprocess_fn=self.preprocess,
        )
        features =  extract_features(
            model=self.model,
            model_identifier=self.identifier,
            nodes=self.nodes,
            hooks=self.hooks,
            datapipe=datapipe,
            datapipe_identifier=stimulus_set.identifier,
        )
        
        features = flatten_features(features)
        features = concatenate_features(features)

        features = features.rename(
            f"{self.identifier}.nodes={self.hash}.stimulus_set={stimulus_set.identifier}"
        )

        return features

    def extract_features(
        self, stimulus_set: StimulusSet, *, batch_size: int = 256
    ) -> dict[str, xr.DataArray]:
        datapipe = create_image_datapipe(
            datapipe=stimulus_set,
            indices=stimulus_set.metadata.index.to_list(),
            batch_size=batch_size,
            preprocess_fn=self.preprocess,
        )
        return extract_features(
            model=self.model,
            model_identifier=self.identifier,
            nodes=self.nodes,
            hooks=self.hooks,
            datapipe=datapipe,
            datapipe_identifier=stimulus_set.identifier,
        )

    def _hash_node_list(self) -> str:
        node_concat = ".".join(sorted(self.nodes))
        hook_concat = ".".join(
            sorted([f"{node}:{hook.identifier}" for node, hook in self.hooks.items()])
        )
        return hash_string("_".join([node_concat, hook_concat]))
    
    def update_nodes(self, nodes: list[str]) -> None:
        self.nodes = nodes
        new_hooks = {}
        for k in self.hooks.keys():
            if k in nodes:
                new_hooks[k] = self.hooks[k]
        self.hooks = new_hooks
        self.hash = self._hash_node_list()
        
    def update_hooks(self, hooks: dict[str, Hook]) -> None:
        self.hooks = hooks
        self.hash = self._hash_node_list()

