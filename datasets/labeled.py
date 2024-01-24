"""
Utilities to read the IF-labeled data set used for training.
"""
import os
from logging import error, info, warn
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile
import xmltodict
from skimage import img_as_float32
from tifffile import imread
from utils import num_digits

bg_threshold = 55000 / 65535
pixel_size = 0.325  # Microns


##### Importing #####


def import_images(
    source_dir: Path,
    save_dir: Path,
    image_names: Optional[List[str]] = None,
    dry_run: bool = False,
) -> None:
    if dry_run:
        warn("NOTE: The following is a dry run!")
    if image_names is None:
        image_names = os.listdir(source_dir)
    info(f"Importing {len(image_names)} images from {source_dir} into {save_dir}.")
    if not dry_run:
        save_dir.mkdir(parents=True)
    for i, image_name in enumerate(image_names):
        info(f"Importing image #{i+1:0{num_digits(len(image_names))}d} {image_name}.")
        image_source_path = source_dir / image_name
        if not dry_run:
            copy2(image_source_path, save_dir)
    info("Importing done.")


def import_tissue_geometry(
    dataset_info: Dict[str, Any], geometries_save_dir: Path, dry_run: bool = False
) -> None:
    if dry_run:
        warn("NOTE: The following is a dry run!")
    info(
        f"Importing {len(dataset_info)} tissue geometries from Omero into {geometries_save_dir}."
    )
    from omero import Omero

    omero = Omero()
    for i, image_info in enumerate(dataset_info):
        image_id = image_info["@id"]
        image_name = image_info["Name"]
        info(
            f"Importing tissue geometry for #{i+1:0{num_digits(len(dataset_info))}d} {image_name} (OMERO id: {image_id})."
        )
        rois = omero.get_image_rois(omero.get_image_info(image_id))
        # E.g. the ROI's name of "H2021-192_exp2_s01_HP-224363BR.ome.tiff" is "HP-224363BR".
        roi_name = image_name.split(".")[0].split("_")[-1][:-2]
        tissue_geometry = rois[roi_name]
        geometry_save_path = geometries_save_dir / (image_name + ".wkt")
        if not dry_run:
            with open(geometry_save_path, "w") as f:
                f.write(tissue_geometry.wkt)
    info("Importing done.")


##### Extraction #####

_sentinel_idx = "channel_index_not_detected"


def _get_image_channel_metadata(image_path: Path) -> Dict:
    with tifffile.TiffFile(image_path) as f:
        image_metadata = xmltodict.parse(f.ome_metadata)["OME"]["Image"]
    return image_metadata["Pixels"]["Channel"]


def _find_channel_index(names, channel_metadata):
    matches = [i for i, c in enumerate(channel_metadata) if c["@Name"] in names]
    assert len(matches) < 2, f"Multiple channels match {names} in {channel_metadata}."
    return matches[0] if len(matches) == 1 else _sentinel_idx


def extract_he(
    image_path: Path,
    save_path: Path,
    pyramid_level: int = 0,
    channel_dimension: int = 0,
    dry_run: bool = False,
) -> None:
    info(f"Extracting H&E channels from level {pyramid_level} of image {image_path}.")
    channel_metadata = _get_image_channel_metadata(image_path)
    he_r_idx = _find_channel_index(["RED"], channel_metadata)
    he_g_idx = _find_channel_index(["GREEN"], channel_metadata)
    he_b_idx = _find_channel_index(["BLUE"], channel_metadata)
    if any(idx == _sentinel_idx for idx in [he_r_idx, he_g_idx, he_b_idx]):
        error(
            f"Could not find all H&E channel indices: r={he_r_idx}, g={he_g_idx}, b={he_b_idx}."
        )
    else:
        info(f"Found H&E channels at indices r={he_r_idx}, g={he_g_idx}, b={he_b_idx}.")
        he = np.stack(
            [
                img_as_float32(
                    imread(image_path, series=0, level=pyramid_level, key=he_r_idx)
                ),
                img_as_float32(
                    imread(image_path, series=0, level=pyramid_level, key=he_g_idx)
                ),
                img_as_float32(
                    imread(image_path, series=0, level=pyramid_level, key=he_b_idx)
                ),
            ],
            axis=channel_dimension,
        )
        info(f"The overall shape of the extracted channels is {he.shape}.")
        if not dry_run:
            info(f"Saving channels to {save_path}.")
            np.save(save_path, he)


def extract_cksox(image_path: Path, save_path: Path, dry_run: bool = False):
    info(f"Extracting CKSOX channel from image {image_path}.")
    channel_metadata = _get_image_channel_metadata(image_path)
    cksox_idx = _find_channel_index(["CKSOX10", "cCKSOX10"], channel_metadata)
    if cksox_idx == _sentinel_idx:
        error(f"Could not find CKSOX channel index.")
    else:
        info(f"Found CKSOX channel at index {cksox_idx}.")
        cksox = imread(image_path, series=0, key=cksox_idx)
        info(f"The shape of the extracted channel is {cksox.shape}.")
        if not dry_run:
            info(f"Saving channel to {save_path}.")
            np.save(save_path, cksox)


def extract_cd(
    image_path: Path, save_path: Path, channel_dimension: int = 0, dry_run: bool = False
) -> None:
    info(f"Extracting CD3, CD4, and CD8 channels from image {image_path}.")
    channel_metadata = _get_image_channel_metadata(image_path)
    cd3_idx = _find_channel_index(["CD3", "cCD3"], channel_metadata)
    cd4_idx = _find_channel_index(["CD4", "cCD4"], channel_metadata)
    cd8_idx = _find_channel_index(["CD8", "cCD8"], channel_metadata)
    if any(idx == _sentinel_idx for idx in [cd3_idx, cd4_idx, cd8_idx]):
        error(
            f"Could not find all CD channel indices: CD3={cd3_idx}, CD4={cd4_idx}, CD8={cd8_idx}."
        )
    else:
        info(
            f"Found CD channels at indices: CD3={cd3_idx}, CD4={cd4_idx}, CD8={cd8_idx}."
        )
        cd = np.stack(
            [
                imread(image_path, series=0, key=cd3_idx),
                imread(image_path, series=0, key=cd4_idx),
                imread(image_path, series=0, key=cd8_idx),
            ],
            axis=channel_dimension,
        )
        info(f"The overall shape of the extracted channels is {cd.shape}.")
        if not dry_run:
            info(f"Saving channels to {save_path}.")
            np.save(save_path, cd)


def extract_tissue_fg(img: np.ndarray) -> np.ndarray:
    assert np.issubdtype(img.dtype, np.floating)
    return (img[0] < bg_threshold) | (img[1] < bg_threshold) | (img[2] < bg_threshold)
