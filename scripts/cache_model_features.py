from dotenv import load_dotenv
load_dotenv()

import sys
import argparse
from loguru import logger
import logging
logger.remove()
logger.add(sys.stderr, level="INFO")

logging.basicConfig(level=logging.INFO)

from lib.analyses.yield_models import (
    yield_models,
    load_yielder,
)
from lib.datasets import load_stimulus_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidx", type=int, default=0)
    parser.add_argument("--nyield", type=int, default=9999)
    parser.add_argument(
        "--yielder", type=str, default="resnet50_imagenet1k_varied_tasks"
    )
    args = parser.parse_args()

    yielder = load_yielder(
        args.yielder,
    )

    stimulus_set = load_stimulus_set(identifier="NSD", mode="all")

    for model in yield_models(yielder, num_yield=args.nyield, start_idx=args.sidx):
        logging.info(f"Caching {model.identifier}")
        model(stimulus_set)
