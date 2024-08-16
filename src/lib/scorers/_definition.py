"""
    Adapted code from Raj Magesh
"""
import logging

logging.basicConfig(level=logging.INFO)
from abc import ABC, abstractmethod

import xarray as xr
from bonner.computation.xarray import align_source_to_target


class Scorer(ABC):
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def __call__(
        self, 
        *, 
        predictor: xr.DataArray, 
        target: xr.DataArray,
        target_dim: str = "neuroid",
    ) -> xr.Dataset:
        predictor, target = self._align_predictor_and_target(
            predictor=predictor, target=target, target_dim=target_dim
        )
        
        score = self._score(
            predictor=predictor, target=target, target_dim=target_dim
        )
        score = score.assign_attrs(
            {
                "predictor": predictor.name,
                "target": target.name,
            }
        )
        return score

    @abstractmethod
    def _score(self, *, predictor: xr.DataArray, target: xr.DataArray) -> xr.Dataset:
        pass

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
    

class CrossPredictorScorer(ABC):
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def __call__(
        self, 
        *, 
        predictor_train_rep: xr.DataArray, 
        predictor_test_rep: xr.DataArray, 
        target: xr.DataArray,
        target_dim: str = "neuroid",
    ) -> xr.Dataset:
        predictor_train_rep, target = self._align_predictor_and_target(
            predictor=predictor_train_rep, target=target, target_dim=target_dim
        )
        predictor_test_rep, target = self._align_predictor_and_target(
            predictor=predictor_test_rep, target=target, target_dim=target_dim
        )
        
        score = self._score(
            predictor_train_rep=predictor_train_rep, 
            predictor_test_rep=predictor_test_rep, 
            target=target, 
            target_dim=target_dim
        )
        score = score.assign_attrs(
            {
                "predictor_train": predictor_train_rep.name,
                "predictor_test": predictor_test_rep.name,
                "target": target.name,
            }
        )
        return score

    @abstractmethod
    def _score(
        self,
        *, 
        predictor_train_rep: xr.DataArray, 
        predictor_test_rep: xr.DataArray, 
        target: xr.DataArray
    ) -> xr.Dataset:
        pass

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