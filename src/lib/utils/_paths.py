from pathlib import Path
import os

IMAGENET_TRAIN_STIM_HOME = Path("/scratch4/mbonner5/shared/brainscore/brainio/image_russakovsky2014_ilsvrc2012/train")
NSD_STIM_HOME = Path(os.path.join(
    os.environ["BONNER_DATASETS_HOME"],
    "allen2021.natural_scenes",
    "images",
))
                     