"""
    Adapted code from Raj Magesh
"""
from PIL import Image
import copy

import numpy as np
import pandas as pd
from bonner.caching import cache
from torchdata.datapipes.map import MapDataPipe
from lib.utils import IMAGENET_TRAIN_STIM_HOME

IDENTIFIER = "russakovsky2014_imagenet.train"

@cache(
    "stimulus_sets/{identifier}.pkl",
    helper=lambda kwargs: {
        "identifier": IDENTIFIER,
    }
)
def load_stimulus_metedata() -> pd.DataFrame:
    root = IMAGENET_TRAIN_STIM_HOME
    stimulus_paths = [path.relative_to(root) for path in root.rglob(f"*.*")]
    stimuli = [
        (
            stimulus_path.stem.split(".")[0],
            str(stimulus_path.parent),
            str(stimulus_path)
        )
        for stimulus_path in stimulus_paths
    ]
    return (
        pd.DataFrame(data=stimuli, columns=["stimulus_id", "class_id", "path"])
        .set_index("stimulus_id")
    )


class StimulusSet(MapDataPipe):
    def __init__(self) -> None:
        self.identifier = IDENTIFIER
        self.root = IMAGENET_TRAIN_STIM_HOME
        self.metadata = load_stimulus_metedata()
        
    def __getitem__(self, stimulus_id: str) -> Image.Image:
        return Image.open(
            self.root / self.metadata.loc[stimulus_id, "path"]
        ).convert("RGB")
    
    def __len__(self) -> int:
        return len(self.metadata.path)
    

def sample_idx(
    metadata: pd.DataFrame,
    n_stimuli: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)

    n_classes = len(metadata.class_id.unique())
    n_stimuli_per_class = [n_stimuli // n_classes] * n_classes
    for i in range(n_stimuli % n_classes):
        n_stimuli_per_class[i] += 1
    rng.shuffle(n_stimuli_per_class)

    indices = metadata.groupby("class_id").indices
    sample_indices = []
    for i_class, class_ in enumerate(sorted(list(indices.keys()))):
        sample_indices.extend(
            rng.choice(
                indices[class_], 
                size=n_stimuli_per_class[i_class],
                replace=False
            )
            .tolist()
        )
    rng.shuffle(sample_indices)
    return sample_indices


@cache(
    "stimulus_sets/{identifier}.n_stimuli={n_stimuli}.seed={seed}.pkl",
    helper=lambda kwargs: {
        "identifier": IDENTIFIER,
        "n_stimuli": kwargs["n_stimuli"],
        "seed": kwargs["seed"],
    }
)
def load_stimulus_set(
    *,
    n_stimuli: (int | list[int]),
    seed: int = 11,
):
    stimulus_set = StimulusSet()
    if isinstance(n_stimuli, int):
        stimulus_set.identifier += f".n_stimuli={n_stimuli}.seed={seed}"
        idx = sample_idx(stimulus_set.metadata, n_stimuli, seed)
        stimulus_set.metadata = stimulus_set.metadata.iloc[idx]
        return stimulus_set
    else:
        n_stimuli = np.array(n_stimuli)
        sum_ = np.sum(n_stimuli)
        idx = sample_idx(stimulus_set.metadata, sum_, seed)
        stimulus_set.metadata = stimulus_set.metadata.iloc[idx]
        stimulus_sets = {}
        for i, n_stimuli_ in enumerate(n_stimuli):
            temp = copy.deepcopy(stimulus_set)
            temp.identifier += f".n_stimuli={n_stimuli_}_of_{sum_}.seed={seed}"
            sl = slice(np.sum(n_stimuli[:i]), np.sum(n_stimuli[:i + 1]))
            temp.metadata = temp.metadata.iloc[sl]
            stimulus_sets[i] = temp
        return stimulus_sets
        
   
