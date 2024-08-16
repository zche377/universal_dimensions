from dotenv import load_dotenv
load_dotenv()

import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

import logging

logging.basicConfig(level=logging.INFO)

import argparse
import numpy as np
from lib.analyses.yield_models import (
    yield_models,
    load_yielder,
)

from lib.analyses.brain_similarity import brain_similarity
from lib.analyses.brain_reliability import brain_reliability
from lib.analyses.universality_index import universality_index
from lib.analyses.rsa import rsa
from bonner.datasets.allen2021_natural_scenes import N_SUBJECTS

from lib.analyses.pc_umap import pc_umap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidx", type=int, default=0)
    parser.add_argument("--nyield", type=int, default=9999)
    parser.add_argument("--roi", type=str, default="general")
    parser.add_argument("--score", type=str, default="universality_index")
    parser.add_argument("--mode", type=str, default="NSD")
    parser.add_argument(
        "--yielder", type=str, default="resnet50_imagenet1k_varied_tasks"
    )
    parser.add_argument("--subj", type=int, default=None)
    parser.add_argument("--bychannel", dest="by_channel", action="store_true")
    parser.set_defaults(by_channel=False)

    parser.add_argument("--basis", type=str, default="activation_pc")
    parser.add_argument("--cb", dest="cross_basis", action="store_true")
    parser.set_defaults(cross_basis=False)
    parser.add_argument("--pca", dest="pca", action="store_true")
    parser.set_defaults(pca=False)

    parser.add_argument("--reg", type=str, default="ridgecv")
    parser.add_argument("--sortby", type=str, default=None)
    parser.add_argument("--ntop", type=str, default=None)
    parser.add_argument("--stim", type=str, default="0")
    parser.add_argument("--threshcol", type=str, default=None)
    parser.add_argument("--node", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--thresh", type=float, default=None)
    parser.add_argument("--roia", type=str, default=None)
    parser.add_argument("--roib", type=str, default=None)

    args = parser.parse_args()

    yielder = load_yielder(args.yielder)
    predictor_yielder = yielder

    mode_dict = {
        "ImageNet": "all",
        "NSD": "shared",
        "Hybrid": "shared",
        "NSD.shared": "same",
    }

    if args.ntop is None:
        ntop = None
    elif args.ntop.isdigit():
        ntop = int(args.ntop)
    else:
        ntop = float(args.ntop)    
    
    match args.score:
        case "pc_umap":
            pc_umap(
                yielder,
                args.yielder,
                args.basis,
                args.mode,
                args.roi,
                args.reg,
                args.sortby,
                n_top=ntop,
                stim=int(args.stim) if args.stim.isdigit() else args.stim,
                threshold_col=args.threshcol,
                threshold=args.thresh,
                node_str=args.node,
                model_str=args.model,
                pca=args.pca,
            )
        case "universality_index":
            if args.basis == "voxel":
                subjects = (
                    np.arange(args.subj, args.subj + args.nyield).tolist()
                    if args.subj is not None
                    else range(N_SUBJECTS)
                )
                for subject in subjects:
                    universality_index(
                        args.basis,
                        args.mode,
                        cross_basis=args.cross_basis,
                        model_predictor_yielder=predictor_yielder,
                        roi=args.roi,
                        regression_type=args.reg,
                        subject=subject,
                    )
            else:
                for model in yield_models(
                    yielder,
                    num_yield=args.nyield,
                    start_idx=args.sidx,
                ):
                    logging.info(f"Computing {args.score} of {model.identifier}")
                    universality_index(
                        args.basis,
                        args.mode,
                        cross_basis=args.cross_basis,
                        model=model,
                        model_predictor_yielder=predictor_yielder,
                        roi=args.roi,
                        regression_type=args.reg,
                    )
        case "brain_reliability":
            if args.basis == "voxel":
                brain_reliability(
                    None,
                    roi=args.roi,
                    mode=mode_dict[args.mode],
                    basis=args.basis,
                    regression_type=args.reg,
                )
            else:
                for model in yield_models(
                    yielder,
                    num_yield=args.nyield,
                    start_idx=args.sidx,
                ):
                    logging.info(f"Computing {args.score} of {model.identifier}")
                    brain_reliability(
                        model,
                        roi=args.roi,
                        mode=mode_dict[args.mode],
                        basis=args.basis,
                        regression_type=args.reg,
                    )
        case "brain_similarity":
            for model in yield_models(
                yielder,
                num_yield=args.nyield,
                start_idx=args.sidx,
            ):
                logging.info(f"Computing {args.score} of {model.identifier}")
                brain_similarity(
                    model,
                    roi=args.roi,
                    mode=mode_dict[args.mode],
                    basis=args.basis,
                    regression_type=args.reg,
                    subjects=list(range(N_SUBJECTS)) if args.subj is None else args.subj,  
                )
        case "rsa":
            for model in yield_models(
                    yielder,
                    num_yield=args.nyield,
                    start_idx=args.sidx,
                ):
                logging.info(f"Computing {args.score} of {model.identifier}")
                rsa(
                    model, 
                    roi=args.roi,
                    mode=mode_dict[args.mode],
                    sort_by=args.sortby,
                    n_top=ntop,
                    yielder_id=args.yielder,
                )
        case _:
            raise ValueError(f"Unknown score type {args.score}")
