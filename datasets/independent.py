"""
Utilities to read the independent data set used for external validation.
"""
from logging import info
from pathlib import Path

import numpy as np
import utils
from skimage import img_as_float32
from skimage.transform import rescale

from .labeled import extract_tissue_fg as _labeled_extract_tissue_fg
from .labeled import import_images  # Re-export
from .labeled import pixel_size as _labeled_pixel_size

pixel_size = 0.25  # Microns


def extract_he(
    image_path: Path,
    save_path: Path,
    pyramid_level: int = 0,
    channel_dimension: int = 0,
    dry_run: bool = False,
) -> None:
    info(f"Extracting H&E channels from level {pyramid_level} of image {image_path}.")
    he = img_as_float32(
        utils.read_tiff_channel(image_path, 2 + pyramid_level)
    )  # The first two pages are thumbnails.
    scale = pixel_size / _labeled_pixel_size
    # Corresponds to skimage's default value.
    anti_aliasing_sigma = (_labeled_pixel_size / pixel_size - 1) / 2
    info(
        (
            f"Rescaling by factor {scale:.4f}.. to match physical pixel size of labeled data set. Gaussian sigma for "
            f"anti-aliasing is {anti_aliasing_sigma:.4f}."
        )
    )
    he = rescale(
        he,
        scale,
        anti_aliasing=True,
        anti_aliasing_sigma=anti_aliasing_sigma,
        multichannel=True,
    )
    info("Rotating by 90° CCW to roughly match orientation of labeled counterpart.")
    he = np.rot90(he)
    if channel_dimension != 2:
        # The read image has its channels in the last dimension.
        he = np.moveaxis(he, 2, channel_dimension)
    info(f"The overall shape of the extracted channels is {he.shape}.")
    if not dry_run:
        info(f"Saving channels to {save_path}.")
        np.save(save_path, he)


def extract_tissue_fg(img: np.ndarray) -> np.ndarray:
    # TODO: This is a temporary workaround because registration introduced black BG fill color. Change registration,
    # re-register images, and remove this.
    return _labeled_extract_tissue_fg(img) & (
        (img[0] != 0.0) | (img[1] != 0.0) | (img[2] != 0.0)
    )
