import functools
from pathlib import Path

import nibabel as nib
import nilearn.plotting
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from nibabel.nifti1 import Nifti1Header, Nifti1Image
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_data, load_surf_mesh, vol_to_surf
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

from bonner.datasets._utilities import BONNER_DATASETS_HOME

from typing import Any

import seaborn as sns

DEFAULT_RC = {
    "font.family": ["serif"],
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "pdf.fonttype": 42,
    "figure.figsize": (3, 3),
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.format": "svg",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.edgecolor": "None",
    "figure.titlesize": "medium",
    "figure.labelsize": "medium",
    "legend.title_fontsize": "medium",
    "legend.fontsize": "medium",
    "lines.markeredgewidth": 0,
}


def set_plotting_defaults(
    context: str = "paper",
    style: str = "white",
    rc: dict[str, Any] = DEFAULT_RC,
) -> None:
    sns.set_theme(
        context=context,
        style=style,
        rc=rc,
    )



MNI_SHAPE = (182, 218, 182)
MNI_ORIGIN = np.asarray([183 - 91, 127, 73]) - 1
MNI_RESOLUTION = 1
DATASET_DIRECTORY = BONNER_DATASETS_HOME / "allen2021.natural_scenes"

BRAIN_SHAPES = {
    0: (81, 104, 83),
    1: (82, 106, 84),
    2: (81, 106, 82),
    3: (85, 99, 80),
    4: (79, 97, 78),
    5: (85, 113, 83),
    6: (78, 95, 81),
    7: (80, 103, 78),
}



def load_transformation(
    subject: int,
    *,
    source_space: str,
    target_space: str,
    suffix: str,
    dataset_directory: Path = DATASET_DIRECTORY,
) -> np.ndarray:
    filepath = (
        dataset_directory
        / "nsddata"
        / "ppdata"
        / f"subj{subject + 1:02}"
        / "transforms"
        / f"{source_space}-to-{target_space}{suffix}"
    )
    
    return nib.load(filepath).get_fdata()


def reshape_dataarray_to_brain(
    data: xr.DataArray,
    *,
    subject: int,
) -> np.ndarray:
    brain_shape = BRAIN_SHAPES[subject]
    if data.ndim == 2:
        output_shape = (data.shape[0], *brain_shape)
    else:
        output_shape = brain_shape

    output = np.full(output_shape, fill_value=np.nan)
    output[..., data["x"].data, data["y"].data, data["z"].data] = data.data
    return output


def convert_array_to_mni(
    data: xr.DataArray,
    *,
    subject: int,
    order: int = 0,
) -> xr.DataArray:
    data_ = reshape_dataarray_to_brain(
        data=data,
        subject=subject,
    )

    transformation = load_transformation(
        subject=subject,
        source_space="func1pt8",
        target_space="MNI",
        suffix=".nii.gz",
    )
    transformation -= 1
    transformation = np.flip(transformation, axis=0)

    good_voxels = np.all(
        np.stack(
            [transformation[..., dim] < data_.shape[dim] for dim in (-1, -2, -3)]
            + [np.all(np.isfinite(transformation), axis=-1)],
            axis=-1,
        ),
        axis=-1,
    )
    neuroids = xr.DataArray(
        data=good_voxels,
        dims=("x", "y", "z"),
    ).stack({"neuroid": ("x", "y", "z")})
    neuroids = neuroids[neuroids]

    transformation_ = transformation[good_voxels, :].transpose()

    return xr.DataArray(
        name=f"{data.name}.mni",
        data=map_coordinates(
            np.nan_to_num(data_.astype(np.float64), nan=0),
            transformation_,
            order=order,
            mode="nearest",
            output=np.float32,
        ),
        dims=("neuroid",),
        coords={dim: ("neuroid", neuroids[dim].data) for dim in ("x", "y", "z")},
    )


def convert_ndarray_to_nifti1image(
    data: np.ndarray,
    *,
    resolution: float = MNI_RESOLUTION,
    origin: np.ndarray = MNI_ORIGIN,
) -> Nifti1Image:
    header = Nifti1Header()
    header.set_data_dtype(data.dtype)

    affine = np.diag([resolution] * 3 + [1])
    if origin is None:
        origin = (([1, 1, 1] + np.asarray(data.shape)) / 2) - 1
    affine[0, -1] = -origin[0] * resolution
    affine[1, -1] = -origin[1] * resolution
    affine[2, -1] = -origin[2] * resolution

    return Nifti1Image(data, affine, header)


def _normalize_curv_map(
    curv_map,
    /,
    *,
    low: float = 0.25,
    high: float = 0.5,
) -> np.ndarray:
    negative = curv_map < 0
    positive = curv_map >= 0
    curv_map[negative] = low
    curv_map[positive] = high
    return curv_map


def plot_brain_map(
    volume: Nifti1Image,
    vmax: float,
    *,
    ax: Axes,
    hemisphere: str,
    surface_type: str = "infl",
    mesh: str = "fsaverage",
    low: float = 0.25,
    high: float = 0.5,
    **kwargs,
) -> None:
    fsaverage = fetch_surf_fsaverage(mesh=mesh)

    nilearn.plotting.plot_surf_stat_map(
        axes=ax,
        stat_map=vol_to_surf(
            volume,
            fsaverage[f"pial_{hemisphere}"],
        ),
        surf_mesh=fsaverage[f"{surface_type}_{hemisphere}"],
        threshold=np.finfo(np.float32).resolution,
        colorbar=False,
        # bg_map=_normalize_curv_map(
        #     load_surf_data(fsaverage[f"curv_{hemisphere}"]),
        #     low=low,
        #     high=high,
        # ),
        bg_map=fsaverage[f"sulc_{hemisphere}"],
        engine="matplotlib",
        vmax=vmax,
        **kwargs,
    )


def fill_transparent_background(
    input_: Image.Image,
    /,
    *,
    color: str = "WHITE",
) -> Image.Image:
    output = Image.new("RGBA", input_.size, color)
    output.paste(input_, mask=input_)
    return output


def concatenate_lateral_and_ventral(
    *,
    lateral: Image.Image,
    ventral: Image.Image,
) -> Image.Image:
    cropbox = {
        "ventral": {
            "left": 0.05,
            "right": 0.05,
            "top": 0.33,
            "bottom": 0.3,
        },
        "lateral": {
            "left": 0.2,
            "right": 0.1,
            "top": 0.1,
            "bottom": 0.2,
        },
    }
    images = {
        "lateral": lateral,
        "ventral": ventral,
    }
    images = {
        view: crop_fraction(image, **cropbox[view]) for view, image in images.items()
    }
    images["ventral"] = rescale_image(
        images["ventral"],
        size=images["lateral"].size,
        direction="width",
    )
    return concatenate_images(
        images["lateral"],
        images["ventral"],
        direction="vertical",
        color=None,
    )


def rescale_image(
    image: Image.Image,
    /,
    *,
    size: tuple[int, int],
    direction: str,
) -> Image.Image:
    w_reference, h_reference = size
    w, h = image.size
    match direction:
        case "height":
            return image.resize(
                (int(w * h_reference / h), h_reference),
            )
        case "width":
            return image.resize(
                (w_reference, int(h * w_reference / w)),
            )
        case _:
            raise ValueError


def concatenate_images(
    first: Image.Image,
    second: Image.Image,
    /,
    *,
    direction: str,
    overlap: float = 0,
    reverse_zorder: bool = False,
    color: str | None = "white",
) -> Image.Image:
    match direction:
        case "horizontal":
            size = (
                first.width + int((1 - overlap) * second.width),
                first.height,
            )
            location = (size[0] - second.width, 0)

        case "vertical":
            size = (
                first.width,
                first.height + int((1 - overlap) * second.height),
            )
            location = (0, size[1] - second.height)
        case _:
            raise ValueError

    if color is not None:
        concatenated_image = Image.new(mode="RGBA", size=size, color=color)
    else:
        concatenated_image = Image.new(mode="RGBA", size=size)

    paste_second_image = functools.partial(
        concatenated_image.alpha_composite,
        second,
        location,
    )

    if reverse_zorder:
        concatenated_image.alpha_composite(first, (0, 0))
        paste_second_image()
    else:
        paste_second_image()
        concatenated_image.alpha_composite(first, (0, 0))

    return concatenated_image


def crop_fraction(
    image: Image.Image,
    *,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> Image.Image:
    height, width = image.size
    return image.crop(
        (
            int(left * width),
            int(top * height),
            int(width - right * width),
            int(height - bottom * height),
        ),
    )


def _plot_single_brain_map(
    mni: xr.DataArray,
    vmax: float,
    *,
    view: str | tuple[int, int],
    hemisphere: str = "left",
) -> Figure:
    set_plotting_defaults()
    volume = np.full(MNI_SHAPE, fill_value=np.nan)
    volume[..., mni["x"].data, mni["y"].data, mni["z"].data] = mni.data
    volume = convert_ndarray_to_nifti1image(volume)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plot_brain_map(
        volume,
        vmax=vmax,
        ax=ax,
        hemisphere=hemisphere,
        view=view,
    )
    return fig


def plot_brain_baguette(
    mni_brain_maps: dict[str, xr.DataArray],
    *,
    cache_directory: Path,
    label_location: tuple[float, float] = (0.55, 0.075),
    vmax=None,
) -> Image.Image:
    set_plotting_defaults()
    x, y = label_location
    cache_directory.mkdir(exist_ok=True, parents=True)

    concatenated_images = []

    for label, mni_brain_map in mni_brain_maps.items():
        filepath = cache_directory / "single_brain" / f"{label}.lateral.png"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if not filepath.exists():
            fig = _plot_single_brain_map(mni_brain_map, vmax=vmax, view=(0, 210),)
            fig.savefig(filepath, transparent=True)
        lateral = Image.open(filepath)

        filepath = cache_directory / "single_brain" / f"{label}.ventral.png"
        if not filepath.exists():
            fig = _plot_single_brain_map(mni_brain_map, vmax=vmax, view="ventral",)
            fig.savefig(filepath, transparent=True)
        ventral = Image.open(filepath)

        plt.close("all")

        image = concatenate_lateral_and_ventral(lateral=lateral, ventral=ventral)
        draw = ImageDraw.Draw(image)
        draw.text(
            (image.width * x, image.height * y),
            f"{label}",
            fill="black",
            font_size=720,
            anchor="ms",
        )
        concatenated_images.append(image)

    concatenate = functools.partial(
        concatenate_images,
        overlap=0.6,
        direction="horizontal",
        color=None,
    )
    concatenated_image = functools.reduce(concatenate, concatenated_images)
    return fill_transparent_background(concatenated_image)