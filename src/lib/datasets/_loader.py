from typing import Any

import xarray as xr

from lib.datasets._definition import StimulusSet


def load_stimulus_set(identifier: str, **kwargs: Any) -> (StimulusSet | list[StimulusSet]):
    match identifier:
        case "ImageNet":
            from lib.datasets.russakovsky2014_imagenet import load_stimulus_set

            return load_stimulus_set(**kwargs)
        case "NSD":
            from lib.datasets.allen2021_natural_scenes import load_stimulus_set

            return load_stimulus_set(**kwargs)
        case _:
            raise ValueError(f"stimulus set {identifier} not found")
        
        
def load_dataset(dataset: str, **kwargs: Any) -> dict[int, xr.DataArray]:
    match dataset:
        case "NSD":
            from lib.datasets.allen2021_natural_scenes import load_dataset
            return load_dataset(**kwargs)
