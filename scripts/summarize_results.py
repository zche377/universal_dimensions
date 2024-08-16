from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

import argparse

import re
import xarray as xr
import numpy as np

from bonner.caching import cache, BONNER_CACHING_HOME
from bonner.datasets.allen2021_natural_scenes import N_SUBJECTS

from lib.analyses.yield_models import (
    yield_models,
    load_yielder,
)
from lib.analyses.brain_similarity import ROIS


def _set_rois(roi):
    if isinstance(roi, list):
        return roi
    elif roi == "default_list":
        return _set_rois
    else:
        return [roi]


@cache(
    "summary_results/universality_index/universality_index.basis=voxel.type=within_basis.regression={regression}.roi={roi}.mode={mode}.nc",
    helper=lambda kwargs: {
        "roi": kwargs["roi"],
        "mode": kwargs["mode"],
        "regression": kwargs["regression"],
    },
)
def _voxel_within_basis(
    roi: str,
    regression: str,
    mode: str,
) -> xr.DataArray:
    UI, subject_list = [], []
    for subject in range(N_SUBJECTS):
        scores = []
        for predictor_subject in range(N_SUBJECTS):
            if subject == predictor_subject:
                continue
            path = (
                f"universality_index"
                f"/basis=voxel"
                f"/type=within_basis"
                f"/scorer=regression.{regression}.n_folds=5.metric=correlation.shuffle=True.seed=11"
                f"/roi={roi}"
                f"/mode={mode}"
                f"/target_identifier={subject}"
                f"/predictor_identifier={predictor_subject}.nc"
            )
            scores.append(cache()._load(path).mean("fold"))
        UI.append(xr.concat(scores, dim="predictor_subject").mean("predictor_subject"))
        subject_list.extend([subject] * len(UI[-1]["neuroid"]))
    return xr.concat(UI, dim="neuroid").assign_coords(
        {"subject": ("neuroid", subject_list)}
    )


@cache(
    "summary_results/universality_index/universality_index.basis={basis}.type=within_basis.regression={regression}.mode={mode}.yielder={yielder}.nc",
    helper=lambda kwargs: {
        "mode": kwargs["mode"],
        "regression": kwargs["regression"],
        "basis": kwargs["basis"],
        "yielder": kwargs["identifier"],
    },
)
def _model_within_basis(
    model_yielder: callable,
    id_yielder: callable,
    basis: str,
    regression: str,
    mode: str,
    identifier: str,
) -> xr.DataArray:
    logging.info("")
    UI, model_list = [], []
    for model in yield_models(model_yielder):
        logging.info(f"Collecting universality index of {model.identifier}")
        scores, node_list = [], []
        for node in model.nodes:
            temp_scores = []
            for predictor_model_identifier in yield_models(id_yielder):
                if model.identifier == predictor_model_identifier:
                    continue
                path = (
                    f"universality_index"
                    f"/basis={basis}"
                    f"/type=within_basis"
                    f"/scorer=regression.{regression}.n_folds=5.metric=correlation.shuffle=True.seed=11"
                    f"/mode={mode}"
                    f"/target_identifier={model.identifier}"
                    f"/node={node}"
                    f"/predictor_identifier={predictor_model_identifier}.nc"
                )
                temp_scores.append(cache()._load(path).mean("fold"))
            scores.append(xr.concat(temp_scores, dim="predictor").median("predictor"))
            node_list.extend([node] * len(scores[-1]["neuroid"]))
        UI.append(
            xr.concat(scores, dim="neuroid").assign_coords(
                {"node": ("neuroid", node_list)}
            )
        )
        model_list.extend([model.identifier] * len(UI[-1]["neuroid"]))
    return xr.concat(UI, dim="neuroid").assign_coords(
        {"model": ("neuroid", model_list)}
    )


@cache(
    "summary_results/universality_index/universality_index.basis=voxel.type=cross_basis.regression={regression}.roi={roi}.yielder={yielder}.nc",
    helper=lambda kwargs: {
        "regression": kwargs["regression"],
        "yielder": kwargs["identifier"],
        "roi": kwargs["roi"],
    },
)
def _voxel_cross_basis(
    regression: str,
    roi: str,
    identifier: str,
) -> xr.DataArray:
    scores, subject_list = [], []
    for subject in range(N_SUBJECTS):
        path = (
            f"universality_index"
            f"/basis=voxel"
            f"/type=cross_basis"
            f"/regression={regression}"
            f"/roi={roi}"
            f"/yielder={identifier}"
            f"/target_identifier={subject}.nc"
        )
        scores.append(cache()._load(path).mean("fold"))
        subject_list.extend([subject] * len(scores[-1]["neuroid"]))
    return xr.concat(scores, dim="neuroid").assign_coords(
        {"subject": ("neuroid", subject_list)}
    )


@cache(
    "summary_results/universality_index/universality_index.basis={basis}.type=cross_basis.regression={regression}.roi={roi}.mode={mode}.yielder={yielder}.nc",
    helper=lambda kwargs: {
        "mode": kwargs["mode"],
        "regression": kwargs["regression"],
        "basis": kwargs["basis"],
        "yielder": kwargs["identifier"],
        "roi": kwargs["roi"],
    },
)
def _model_cross_basis(
    model_yielder: callable,
    basis: str,
    regression: str,
    mode: str,
    roi: str,
    identifier: str,
) -> xr.DataArray:
    scores, model_list, node_list, roi_list = [], [], [], []
    rois = _set_rois(roi)
    for model in yield_models(model_yielder):
        logging.info(f"Collecting universality index of {model.identifier}")
        for node in model.nodes:
            for val in rois:
                path = (
                    f"universality_index"
                    f"/basis={basis}"
                    f"/type=cross_basis"
                    f"/regression={regression}"
                    f"/roi={val}"
                    f"/mode={mode}"
                    f"/target_identifier={model.identifier}"
                    f"/node={node}.nc"
                )
                # logging.info(path)
                scores.append(cache()._load(path).mean("fold"))
                l = len(scores[-1]["neuroid"])
                l = len(scores[-1]["neuroid"])
                roi_list.extend([val] * l)
                node_list.extend([node] * l)
                model_list.extend([model.identifier] * l)

    scores = xr.concat(scores, dim="neuroid")
    scores = scores.assign_coords(
        {
            "model": ("neuroid", model_list),
            "node": ("neuroid", node_list),
            "roi": ("neuroid", roi_list),
        }
    )
    return scores


def universality_index_result(
    model_yielder: callable,
    id_yielder: callable,
    roi: str,
    mode: str,
    basis: str,
    regression: str,
    cross_basis: bool,
    identifier: str,
):
    if basis == "voxel":
        if not cross_basis:
            return _voxel_within_basis(roi, regression, mode)
        else:
            return _voxel_cross_basis(regression, roi, identifier)
    else:
        if not cross_basis:
            return _model_within_basis(
                model_yielder, id_yielder, basis, regression, mode, identifier
            )
        else:
            return _model_cross_basis(
                model_yielder, basis, regression, mode, roi, identifier
            )


@cache(
    "summary_results/brain_similarity/brain_similarity.regression={regression}.basis={basis}.roi={roi}.mode={mode}.yielder={yielder}.nc",
    helper=lambda kwargs: {
        "roi": kwargs["roi"],
        "mode": kwargs["mode"],
        "yielder": kwargs["identifier"],
        "basis": kwargs["basis"],
        "regression": kwargs["regression"],
    },
)
def brain_similarity_results(
    model_yielder: callable,
    roi: str,
    mode: str,
    basis: str,
    regression: str,
    identifier: str,
    subjects: (int | list[int])
) -> xr.DataArray:
    BS, model_list, node_list, roi_list = [], [], [], []
    rois = _set_rois(roi)
    if isinstance(subjects, int):
        subjects = [subjects]
    for model in yield_models(
        model_yielder,
    ):
        logging.info(f"Collecting brain similarity of {model.identifier}")
        nodes = model.nodes
        for node in nodes:
            for val in rois:
                bs = []
                for subject in subjects:
                    path = (
                        f"brain_similarity"
                        f"/basis={basis}"
                        f"/scorer=regression.{regression}.n_folds=5.metric=correlation.shuffle=True.seed=11"
                        f"/roi={val}"
                        f"/mode={mode}"
                        f"/model={model.identifier}"
                        f"/node={node}"
                        f"/subject={subject}.nc"
                    )
                    bs.append(cache()._load(path).mean("fold"))
                if basis == "voxel":
                    subj_list = np.concatenate(
                        [[i] * len(bs[i]) for i in range(N_SUBJECTS)]
                    )
                    bs = xr.concat(bs, dim="neuroid")
                    bs = bs.assign_coords(
                        {
                            "subject": ("neuroid", subj_list),
                        }
                    )
                else:
                    bs = xr.concat(bs, dim="subject").mean(dim="subject")
                l = bs.sizes["neuroid"]
                roi_list.extend([val] * l)
                node_list.extend([str(node)] * l)
                model_list.extend([str(model.identifier)] * l)
                BS.append(bs)
    match basis:
        case (
            "activation_channel"
            | "activation_pc"
            | "voxel"
            | "activation_channel_demeaned"
        ):
            BS = xr.concat(BS, dim="neuroid")
            BS = BS.assign_coords(
                {
                    "model": ("neuroid", model_list),
                    "node": ("neuroid", node_list),
                    "roi": ("neuroid", roi_list),
                }
            )
    return BS


@cache(
    "summary_results/brain_reliability/brain_reliability.basis=voxel.roi={roi}.mode={mode}.nc",
    helper=lambda kwargs: {
        "roi": kwargs["roi"],
        "mode": kwargs["mode"],
    },
)
def _brr_voxel(
    roi: str,
    mode: str,
    regression: str,
) -> xr.DataArray:
    BR = []
    path = f"brain_reliability" f"/basis=voxel" f"/roi={roi}" f"/mode={mode}.nc"
    return cache()._load(path).mean("rep")


@cache(
    "summary_results/brain_reliability/brain_reliability.regression={regression}.basis={basis}.roi={roi}.mode={mode}.yielder={yielder}.nc",
    helper=lambda kwargs: {
        "roi": kwargs["roi"],
        "mode": kwargs["mode"],
        "yielder": kwargs["model_yielder"].__name__,
        "basis": kwargs["basis"],
        "regression": kwargs["regression"],
    },
)
def _brr_model(
    model_yielder: callable,
    roi: str,
    mode: str,
    regression: str,
    basis: str,
) -> xr.DataArray:
    BR, model_list, node_list, roi_list = [], [], [], []
    rois = _set_rois(roi)
    for model in yield_models(model_yielder):
        logging.info(f"Collecting brain reliability of {model.identifier}")
        nodes = model.nodes
        for node in nodes:
            for val in rois:
                br = []
                for subject in range(N_SUBJECTS):
                    path = (
                        f"brain_reliability"
                        f"/basis={basis}"
                        f"/scorer={regression}"
                        f"/roi={val}"
                        f"/mode={mode}"
                        f"/model={model.identifier}"
                        f"/node={node}"
                        f"/subject={subject}.nc"
                    )
                    br.append(cache()._load(path).mean("fold").mean("rep"))
                br = xr.concat(br, dim="subject").mean("subject")

                l = br.sizes["neuroid"]
                node_list.extend([str(node)] * l)
                model_list.extend([str(model.identifier)] * l)
                roi_list.extend([str(val)] * l)
                BR.append(br)
    BR = xr.concat(BR, dim="neuroid")
    BR = BR.assign_coords(
        {
            "model": ("neuroid", model_list),
            "node": ("neuroid", node_list),
            "roi": ("neuroid", roi_list),
        }
    )
    return BR


def brain_reliability_results(
    model_yielder: callable,
    roi: str,
    mode: str,
    basis: str,
    regression: str,
) -> xr.DataArray:
    if basis == "voxel":
        return _brr_voxel(roi, mode, regression)
    else:
        return _brr_model(model_yielder, roi, mode, regression, basis)


@cache(
    "summary_results/rsa_score/rsa_score.roi={roi}.mode={mode}.yielder={yielder}.sort_by={sid}.nc",
    helper=lambda kwargs: {
        "roi": kwargs["roi"],
        "mode": kwargs["mode"],
        "yielder": kwargs["identifier"],
        "sid": kwargs["sid"],
    },
)
def rsa_results(
    model_yielder: callable,
    roi: str,
    mode: str,
    identifier: str,
    sid: str,
) -> xr.DataArray:
    rois = _set_rois(roi)
    RS = []
    for model in yield_models(model_yielder, ):
        logging.info(f"Collecting rsa score of {model.identifier}")
        roi_scores = []
        for roi in rois:
            path = (
                "rsa_score"
                f"/roi={roi}"
                f"/mode={mode}.test"
                f"/sort_by={sid}"
                f"/model={model.identifier}"
                f"/node=best.nc"
            )
            roi_scores.append(cache()._load(path).mean("node").assign_coords({"roi": roi}))
        RS.append(xr.concat(roi_scores, dim="roi").assign_coords({"model": model.identifier}))
    return xr.concat(RS, dim="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", type=str, default="general")
    parser.add_argument("--score", type=str, default="universality_index")
    parser.add_argument(
        "--yielder", type=str, default="resnet50_imagenet1k_varied_tasks"
    )
    parser.add_argument("--mode", type=str, default="NSD")
    parser.add_argument("--bychannel", dest="by_channel", action="store_true")
    parser.set_defaults(by_channel=False)

    parser.add_argument("--reg", type=str, default="ridgecv")
    parser.add_argument("--basis", type=str, default="activation_pc")
    parser.add_argument("--cb", dest="cross_basis", action="store_true")
    parser.set_defaults(cross_basis=False)
    parser.add_argument("--subj", type=int, default=None)
    parser.add_argument("--sortby", type=str, default=None)
    parser.add_argument("--ntop", type=str, default=None)

    args = parser.parse_args()

    model_yielder = load_yielder(args.yielder)
    id_yielder = load_yielder(args.yielder, identifier_only=True)
    identifier = model_yielder.__name__

    mode_dict = {
        "ImageNet": "all",
        "NSD": "shared",
        "Hybrid": "shared",
        "NSD.shared": "same",
        "NSD.z": "shared.z",
        "NSD.nonfiltered": "shared.nonfiltered",
        "spatial": "spatial",
        "RP": "RP",
    }

    match args.score:
        case "universality_index":
            universality_index_result(
                model_yielder,
                id_yielder,
                args.roi,
                mode=args.mode,
                basis=args.basis,
                regression=args.reg,
                cross_basis=args.cross_basis,
                identifier=identifier,
            )
        case "brain_similarity":
            brain_similarity_results(
                model_yielder,
                args.roi,
                mode=mode_dict[args.mode],
                basis=args.basis,
                regression=args.reg,
                identifier=identifier,
                subjects=list(range(N_SUBJECTS)) if args.subj is None else args.subj,
            )
        case "brain_reliability":
            brain_reliability_results(
                model_yielder,
                args.roi,
                mode=mode_dict[args.mode],
                basis=args.basis,
                regression=args.reg,
            )
        case "rsa":
            if args.sortby is None:
                sid = "None"
            else:
                sid = f"{args.sortby}.n_top={args.ntop}"
            rsa_results(
                model_yielder,
                args.roi,
                mode=mode_dict[args.mode],
                identifier=args.yielder,
                sid=sid,
            )
        case _:
            raise ValueError(f"Unknown result type {args.result}")
