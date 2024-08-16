from dotenv import load_dotenv
load_dotenv()

import sys
from loguru import logger
import logging
logger.remove()
logger.add(sys.stderr, level="INFO")

logging.disable(logging.WARNING)
import argparse
from lib.datasets import load_dataset
from lib.analyses.brain_similarity import ROIS

from bonner.datasets.allen2021_natural_scenes import N_SUBJECTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", type=str, default="default_list")
    parser.add_argument("--subj", type=int, default=None)
    args = parser.parse_args()

    if args.roi == "default_list":
        rois = ROIS
    else:
        rois = [args.roi]
        
    subjects = list(range(N_SUBJECTS)) if args.subj is None else args.subj

    for roi in rois:
        load_dataset(dataset="NSD", roi=roi, subjects=subjects)
