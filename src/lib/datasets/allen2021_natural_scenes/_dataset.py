"""
    Adapted code from Raj Magesh
"""
import typing

import numpy as np
import xarray as xr
import nilearn.plotting
import matplotlib as mpl

from bonner.computation.xarray import filter_dataarray
from bonner.datasets.allen2021_natural_scenes import (
    IDENTIFIER,
    N_SUBJECTS,
    compute_shared_stimulus_ids,
    create_roi_selector,
    z_score_betas_within_runs,
    z_score_betas_within_sessions,
    average_betas_across_reps,
    load_betas,
    load_rois,
    load_validity,
    convert_dataarray_to_nifti1image,
    StimulusSet
)
from bonner.caching import cache

from lib.utils import hash_string
from lib.models import Model


ROIS = {
    "general": ({"source": "nsdgeneral", "label": "nsdgeneral"},),
    "V1-4": ({"source": "prf-visualrois"},),
    "faces": ({"source": "floc-faces"},),
    "words": ({"source": "floc-words"},),
    "places": ({"source": "floc-places"},),
    "bodies": ({"source": "floc-bodies"},),
    "frontal": (
        {"source": "corticalsulc", "label": "IFG"},
        {"source": "corticalsulc", "label": "IFRS"},
        {"source": "corticalsulc", "label": "MFG"},
        {"source": "corticalsulc", "label": "OS"},
        {"source": "corticalsulc", "label": "PrCS"},
        {"source": "corticalsulc", "label": "SFRS"},
        {"source": "corticalsulc", "label": "SRS"},
        {"source": "HCP_MMP1", "label": "1"},
        {"source": "HCP_MMP1", "label": "2"},
        {"source": "HCP_MMP1", "label": "3a"},
        {"source": "HCP_MMP1", "label": "3b"},
    ),
    "streams": ({"source": "streams"},),
}

SUBSET_ROIS: dict[str, dict[str, tuple[dict[str, str], ...]]] = {
    "general": {
        "general.lh": (
            {"source": "nsdgeneral", "label": "nsdgeneral", "hemisphere": "l"},
        ),
        "general.rh": (
            {"source": "nsdgeneral", "label": "nsdgeneral", "hemisphere": "r"},
        ),
    },
    "frontal": {
        "IFG": ({"source": "corticalsulc", "label": "IFG"},),
        "IFRS": ({"source": "corticalsulc", "label": "IFRS"},),
        "MFG": ({"source": "corticalsulc", "label": "MFG"},),
        "OS": ({"source": "corticalsulc", "label": "OS"},),
        "PrCS": ({"source": "corticalsulc", "label": "PrCS"},),
        "SFRS": ({"source": "corticalsulc", "label": "SFRS"},),
        "SRS": ({"source": "corticalsulc", "label": "SRS"},),
        "HCP_MMP1_1": ({"source": "HCP_MMP1", "label": "1"},),
        "HCP_MMP1_2": ({"source": "HCP_MMP1", "label": "2"},),
        "HCP_MMP1_3a": ({"source": "HCP_MMP1", "label": "3a"},),
        "HCP_MMP1_3b": ({"source": "HCP_MMP1", "label": "3b"},),
        "frontal.lh": (
            {"source": "corticalsulc", "label": "IFG", "hemisphere": "l"},
            {"source": "corticalsulc", "label": "IFRS", "hemisphere": "l"},
            {"source": "corticalsulc", "label": "MFG", "hemisphere": "l"},
            {"source": "corticalsulc", "label": "OS", "hemisphere": "l"},
            {"source": "corticalsulc", "label": "PrCS", "hemisphere": "l"},
            {"source": "corticalsulc", "label": "SFRS", "hemisphere": "l"},
            {"source": "corticalsulc", "label": "SRS", "hemisphere": "l"},
            {"source": "HCP_MMP1", "label": "1", "hemisphere": "l"},
            {"source": "HCP_MMP1", "label": "2", "hemisphere": "l"},
            {"source": "HCP_MMP1", "label": "3a", "hemisphere": "l"},
            {"source": "HCP_MMP1", "label": "3b", "hemisphere": "l"},
        ),
        "frontal.rh": (
            {"source": "corticalsulc", "label": "IFG", "hemisphere": "r"},
            {"source": "corticalsulc", "label": "IFRS", "hemisphere": "r"},
            {"source": "corticalsulc", "label": "MFG", "hemisphere": "r"},
            {"source": "corticalsulc", "label": "OS", "hemisphere": "r"},
            {"source": "corticalsulc", "label": "PrCS", "hemisphere": "r"},
            {"source": "corticalsulc", "label": "SFRS", "hemisphere": "r"},
            {"source": "corticalsulc", "label": "SRS", "hemisphere": "r"},
            {"source": "HCP_MMP1", "label": "1", "hemisphere": "r"},
            {"source": "HCP_MMP1", "label": "2", "hemisphere": "r"},
            {"source": "HCP_MMP1", "label": "3a", "hemisphere": "r"},
            {"source": "HCP_MMP1", "label": "3b", "hemisphere": "r"},
        ),
    },
    "V1-4": {
        "V4": ({"source": "prf-visualrois", "label": "hV4"},),
        "V3": (
            {"source": "prf-visualrois", "label": "V3v"},
            {"source": "prf-visualrois", "label": "V3d"},
        ),
        "V2": (
            {"source": "prf-visualrois", "label": "V2v"},
            {"source": "prf-visualrois", "label": "V2d"},
        ),
        "V1": (
            {"source": "prf-visualrois", "label": "V1v"},
            {"source": "prf-visualrois", "label": "V1d"},
        ),
    },
    "V1-4.individual": {
        x: ({"source": "prf-visualrois", "label": x},)
        for x in (
            "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"
        )
    },
    
    "streams": {
        f"streams-{stream}": ({"source": "streams", "label": stream},)
        for stream in (
            "early",
            "lateral",
            "parietal",
            "ventral",
            "midlateral",
            "midparietal",
            "midventral",
        )
    },
}



@cache(
    f"miscellaneous/{IDENTIFIER}.shared_stimulus_ids/n_repetition={{n_repetitions}}.pkl"
)
def get_shared_stimulus_ids(n_repetitions: int = 1) -> set[str]:
    return compute_shared_stimulus_ids(
        load_dataset(average_across_reps=False).values(), n_repetitions=n_repetitions
    )


def open_betas_by_roi(
    *,
    subject: int,
    resolution: str,
    preprocessing: str,
    roi: str,
) -> xr.DataArray:
    def _find_parent_roi(roi: str) -> str:
        for parent_roi, child_rois in SUBSET_ROIS.items():
            if roi in child_rois.keys():
                return parent_roi
        raise ValueError("missing ROI")

    rois = load_rois(subject=subject, resolution=resolution).load()
    validity = load_validity(subject=subject, resolution=resolution)
    validity = np.all(
        validity.stack({"neuroid": ("x", "y", "z")}, create_index=True).values[:-1, :],
        axis=0,
    )

    if roi in ROIS:
        selector = create_roi_selector(rois=rois, selectors=ROIS[roi])

        cacher = cache(
            "data"
            f"/dataset={IDENTIFIER}"
            f"/resolution={resolution}.preprocessing={preprocessing}"
            f"/roi={roi}"
            "/raw"
            f"/subject={subject}.nc"
        )

        betas = cacher(load_betas)(
            subject=subject,
            resolution=resolution,
            preprocessing=preprocessing,
            neuroid_filter=np.logical_and(validity, selector),
        )
    else:
        parent_roi = _find_parent_roi(roi)
        betas = (
            open_betas_by_roi(
                subject=subject,
                resolution=resolution,
                preprocessing=preprocessing,
                roi=parent_roi,
            )
            .load()
            .set_index({"neuroid": ("x", "y", "z")})
        )
        rois = load_rois(subject=subject, resolution=resolution).load()
        selector = create_roi_selector(
            rois=rois, selectors=SUBSET_ROIS[parent_roi][roi]
        )
        selector = np.logical_and(validity, selector)
        selector = (
            rois.isel({"neuroid": selector})
            .set_index({"neuroid": ("x", "y", "z")})
            .indexes["neuroid"]
        )

        betas = betas.sel(neuroid=selector).reset_index("neuroid")

    return betas


@cache(
    "data"
    f"/dataset={IDENTIFIER}"
    "/resolution={resolution}.preprocessing={preprocessing}"
    "/roi={roi}"
    "/preprocessed"
    "/z_score={z_score}.average_across_reps={average_across_reps}"
    "/subject={subject}.nc"
)
def preprocess_betas(
    *,
    resolution: str,
    preprocessing: str,
    subject: int,
    roi: str,
    z_score: str,
    average_across_reps: bool,
) -> xr.DataArray:
    betas = open_betas_by_roi(
        resolution=resolution,
        preprocessing=preprocessing,
        subject=subject,
        roi=roi,
    ).load()

    match z_score:
        case "session":
            betas = z_score_betas_within_sessions(betas)
        case "run":
            betas = z_score_betas_within_runs(betas)
        case None:
            pass
        case _:
            raise ValueError("z_score must be 'session', 'run', or None")

    if average_across_reps:
        betas = average_betas_across_reps(betas)
    else:
        reps: dict[str, int] = {}
        rep_id: list[int] = []
        for stimulus_id in betas["stimulus_id"].values:
            if stimulus_id in reps:
                reps[stimulus_id] += 1
            else:
                reps[stimulus_id] = 0
            rep_id.append(reps[stimulus_id])
        betas = betas.assign_coords({"rep_id": ("presentation", rep_id)})

    attrs = {
        "preprocessing": preprocessing,
        "roi": roi,
        "z_score": z_score,
        "average_across_reps": average_across_reps,
        "subject": subject,
    }
    identifier = ".".join([f"{key}={value}" for key, value in attrs.items()])
    return betas.transpose("presentation", "neuroid").rename(
        f"{IDENTIFIER}.{identifier}"
    )


def plot_brain_map(
    data: xr.DataArray,
    *,
    subject: int,
    resolution: str = "1pt8mm",
    **kwargs: typing.Any,
) -> mpl.figure.Figure:
    volume = convert_dataarray_to_nifti1image(
        data, subject=subject, resolution=resolution
    )
    fig, _ = nilearn.plotting.plot_img_on_surf(
        volume,
        views=["lateral", "medial", "ventral"],
        hemispheres=["left", "right"],
        colorbar=True,
        inflate=True,
        threshold=np.finfo(np.float32).resolution,
        **kwargs,
    )
    return fig


def load_dataset(
    *,
    subjects: (int | list[int]) = list(range(N_SUBJECTS)),
    resolution: str = "1pt8mm",
    preprocessing: str = "fithrf_GLMdenoise_RR",
    roi: str = "general",
    z_score: str = "session",
    average_across_reps: bool = True,
    repetition: int = None,
    stimulus_ids: list[str] = None,
    exclude: bool = False,
) -> dict[int, xr.DataArray]:
    if isinstance(subjects, int):
        subjects = [subjects]
    if repetition is not None:
        average_across_reps = False
    dataset = {
        subject: preprocess_betas(
            subject=subject,
            resolution=resolution,
            preprocessing=preprocessing,
            roi=roi,
            z_score=z_score,
            average_across_reps=average_across_reps,
        )
        for subject in subjects
    }
    if stimulus_ids is not None:
        dataset = {
            subject: filter_by_stimulus_id(
                dataset_.load(),
                stimulus_ids=stimulus_ids,
                exclude=exclude,
            )
            for subject, dataset_ in dataset.items()
        }
    if repetition is not None:
        dataset = {
            subject: filter_by_repetition(dataset_.load(), repetition=repetition)
            for subject, dataset_ in dataset.items()
        }
    return dataset


def filter_by_stimulus_id(
    data: xr.DataArray, *, stimulus_ids: list[str], exclude: bool = False
) -> xr.DataArray:
    hash_ = hash_string(".".join(sorted(list(set(stimulus_ids)))))
    return filter_dataarray(
        data,
        coord="stimulus_id",
        values=stimulus_ids,
        exclude=exclude,
    ).rename(f"{data.name}.stimulus_ids={hash_}.exclude={exclude}")


def filter_by_repetition(data: xr.DataArray, *, repetition: int) -> xr.DataArray:
    assert repetition in {0, 1, 2}
    data = data.load().isel({"presentation": data["rep_id"] == repetition})
    return data.sortby(data["stimulus_id"]).rename(f"{data.name}.rep={repetition}")


def load_stimulus_set(
    *,
    mode: str = "all",
    n_stimuli: int = None,
    seed: int = 11,
) -> StimulusSet:
    assert mode in {"all", "shared", "unshared"}
    # recommend to only use mode="all" now that cache is available
    # alternatively, see load_acitivations_from_all
    identifier = f"{IDENTIFIER}.{mode}"
    if n_stimuli is not None:
        identifier += f".n_stimuli={n_stimuli}.seed={seed}"
    return _load_stimulus_set(
        mode=mode,
        n_stimuli=n_stimuli,
        seed=seed,
        identifier=identifier,   
    )


@cache(
    "stimulus_sets/{identifier}.pkl"
)
def _load_stimulus_set(
    mode: str,
    n_stimuli: int,
    seed: int,
    identifier: str,
) -> StimulusSet:
    stimulus_set = StimulusSet()
    stimulus_set.identifier = identifier
    if mode == "shared":
        stimulus_set.metadata = stimulus_set.metadata.loc[
            list(get_shared_stimulus_ids())
        ]
    elif mode == "unshared":
        stimulus_set.metadata = stimulus_set.metadata.drop(
            list(get_shared_stimulus_ids())
        )
    if n_stimuli is not None:
        assert n_stimuli <= len(stimulus_set)
        stimulus_set.metadata = stimulus_set.metadata.sample(
            n=n_stimuli, random_state=seed, axis=0
        )
    stimulus_set.metadata = stimulus_set.metadata.sort_index()
    return stimulus_set


def load_activations_from_all(model: Model, mode: str) -> xr.DataArray:
    stimulus_set = load_stimulus_set()
    x = model(stimulus_set)
    match mode:
        case "all":
            return x
        case "shared":
            return x.sel(presentation=x["stimulus_id"].isin(list(get_shared_stimulus_ids())))
        case "unshared":
            return x.sel(presentation=~(x["stimulus_id"].isin(list(get_shared_stimulus_ids())).values))
        case _:
            raise ValueError(f"Invalid mode: {mode}")    
    

