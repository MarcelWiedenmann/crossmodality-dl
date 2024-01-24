"""
Utilities for data preprocessing and tiling.
"""
import json
import math
from logging import info, warning
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes

import tiling
from utils import compute_metrics, confusion_matrix, list_files, num_digits

#### Preprocessing ####


def threshold_images(
    images_dir: Path,
    masks_save_dir: Path,
    threshold: Union[int, str],
    dry_run: bool = False,
) -> None:
    if dry_run:
        warning("NOTE: The following is a dry run!")

    image_paths = list_files(images_dir, ".npy")

    info(f"Thresholding {len(image_paths)} images in directory {images_dir}.")
    info(f"Thresholded masks will be saved to {masks_save_dir}.")

    if not dry_run:
        masks_save_dir.mkdir(parents=True)

    for i, image_path in enumerate(image_paths):
        info(
            f"Thresholding image #{i+1:0{num_digits(len(image_paths))}d} {image_path.stem}."
        )
        image = np.load(image_path)
        mask_save_path = masks_save_dir / image_path.name
        _threshold_image(image, mask_save_path, threshold, dry_run)
    info("Thresholding done.")


def _threshold_image(
    image, mask_save_path: Path, threshold: Union[int, str], dry_run: bool
) -> None:
    if image.dtype != "uint16":
        raise ValueError(f"Unsupported pixel type: {image.dtype}")
    if isinstance(threshold, str):
        if threshold == "otsu":
            thresh = threshold_otsu(image, nbins=2**16)
            info(f"Computed mask threshold: {thresh}.")
        else:
            raise ValueError(f"Unrecognized thresholding method: {threshold}.")
        thresholding_type = threshold
    else:
        thresh = threshold
        info(f"Set mask threshold: {thresh}.")
        thresholding_type = "manual"
    if not dry_run and thresholding_type != "manual":
        with open(mask_save_path.parent / (mask_save_path.name + ".json"), "w") as f:
            json.dump(
                {"threshold": {"type": thresholding_type, "value": int(thresh)}},
                f,
                indent=4,
            )
    mask = image > thresh
    del image
    if not dry_run:
        np.save(mask_save_path, mask)


def fill_holes_in_masks(
    masks_dir: Path,
    filled_masks_save_dir: Path,
    diameters: Dict[str, int],
    dry_run: bool = False,
) -> None:
    if dry_run:
        warning("NOTE: The following is a dry run!")

    mask_paths = list_files(masks_dir, ".npy")
    if len(mask_paths) != len(diameters):
        raise ValueError(
            "Number of input masks and configured diameters does not match."
        )

    info(f"Filling holes in {len(mask_paths)} masks in directory {masks_dir}.")
    info(f"Masks with their holes filled will be saved to {filled_masks_save_dir}.")

    if not dry_run:
        filled_masks_save_dir.mkdir(parents=True)

    for i, mask_path in enumerate(mask_paths):
        diameter = diameters[mask_path.stem]
        area_threshold = round(math.pi * (diameter / 2) ** 2)
        info(
            f"Filling holes in mask #{i+1:0{num_digits(len(mask_paths))}d} {mask_path}."
        )
        info(
            f"The given diameter of {diameter} px results in an area threshold of {area_threshold} px^2."
        )
        mask = np.load(mask_path)
        filled_mask_save_path = filled_masks_save_dir / mask_path.name
        filled_mask = remove_small_holes(
            mask, area_threshold=area_threshold, connectivity=2
        )
        if not dry_run:
            np.save(filled_mask_save_path, filled_mask)
    info("Hole filling done.")


##### Tiling #####


def tile_images(
    images_dir: Path,
    images_save_dir: Path,
    tile_shape: Tuple[int, int],
    anchor: Tuple[int, int],
    stride: Tuple[int, int],
    dry_run: bool = False,
    image_names: Optional[List[str]] = None,
) -> None:
    if dry_run:
        warning("NOTE: The following is a dry run!")

    image_paths = list_files(images_dir, ".npy")
    if image_names is not None:
        image_paths = [p for p in image_paths if p.name in image_names]

    info(f"Tiling {len(image_paths)} images in directory {images_dir}.")
    info(
        f"Tiled images will be saved to their respective subdirectories under {images_save_dir}."
    )

    if not dry_run:
        images_save_dir.mkdir(parents=True, exist_ok=True)

    info(
        f"The tile shape is {tile_shape}, the anchor is {anchor}, the stride is {stride}."
    )

    for i, image_path in enumerate(image_paths):
        image_name = image_path.stem
        info(f"Tiling image #{i+1:0{num_digits(len(image_paths))}d} {image_name}.")
        image_save_dir = images_save_dir / image_name
        if not dry_run:
            image_save_dir.mkdir()
        tile_image(image_path, image_save_dir, tile_shape, anchor, stride, dry_run)
    info("Tiling of images done.")


def tile_image(
    image_or_path: Union[np.ndarray, Path],
    save_dir: Path,
    tile_shape: Tuple[int, int],
    anchor: Tuple[int, int],
    stride: Tuple[int, int],
    dry_run: bool,
) -> None:
    if isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        image = np.load(image_or_path)
    original_image_shape = image.shape
    assert (
        len(original_image_shape) <= 3
    ), "Tiling is only supported for 2D and 3D images."

    tiles = _Tiles(image, tile_shape, anchor, stride)
    num_tiles_total = tiles.num_tiles_y * tiles.num_tiles_x
    info(
        (
            f"Tiling image with shape {original_image_shape} into {num_tiles_total} "
            f"(#y: {tiles.num_tiles_y}, #x: {tiles.num_tiles_x}) tiles."
        )
    )
    info(f"Tiles will be saved to directory {save_dir}.")

    tiling.write_tiling_info(
        save_dir, original_image_shape, tiles.pad_y, tiles.pad_x, tiles.pad_mode
    )

    num_tiles_y_num_digits = num_digits(tiles.num_tiles_y)
    num_tiles_x_num_digits = num_digits(tiles.num_tiles_x)
    num_tiles_total_num_digits = num_digits(num_tiles_total)

    tile_index = 0
    for y in range(tiles.num_tiles_y):
        for x in range(tiles.num_tiles_x):
            tile_save_path = save_dir / (
                save_dir.name
                + f"_tile_{tile_index:0{num_tiles_total_num_digits}d}_y_{y:0{num_tiles_y_num_digits}d}_x_{x:0{num_tiles_x_num_digits}d}"
            )
            tile = tiles[y, x]
            if not dry_run:
                np.save(tile_save_path, tile)
            tile_index += 1
            del tile
    del tiles
    del image


class _Tiles:
    def __init__(
        self,
        image: np.ndarray,
        tile_shape: Tuple[int, int],
        anchor: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        img_size_y, img_size_x = image.shape[-2:]
        tile_size_y, tile_size_x = tile_shape
        half_tile_size_y, half_tile_size_x = tile_size_y // 2, tile_size_x // 2
        anchor_y, anchor_x = anchor
        stride_y, stride_x = stride

        num_tiles_y = math.ceil((img_size_y - anchor_y) / stride_y)
        num_tiles_x = math.ceil((img_size_x - anchor_x) / stride_x)

        pad_before_y = max(0, half_tile_size_y - anchor_y)
        pad_after_y = max(
            0, (anchor_y + (num_tiles_y - 1) * stride_y + half_tile_size_y) - img_size_y
        )
        pad_before_x = max(0, half_tile_size_x - anchor_x)
        pad_after_x = max(
            0, (anchor_x + (num_tiles_x - 1) * stride_x + half_tile_size_x) - img_size_x
        )
        pad_mode = "reflect"

        if sum([pad_before_y, pad_after_y, pad_before_x, pad_after_x]) > 0:
            pad_width = (((0, 0),) if image.ndim == 3 else ()) + (
                (pad_before_y, pad_after_y),
                (pad_before_x, pad_after_x),
            )
            image = np.pad(image, pad_width, pad_mode)
            anchor_y += pad_before_y
            anchor_x += pad_before_x

        self.image = image
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.half_tile_size_y = half_tile_size_y
        self.half_tile_size_x = half_tile_size_x
        self.anchor_y = anchor_y
        self.anchor_x = anchor_x
        self.stride_y = stride_y
        self.stride_x = stride_x
        self.num_tiles_y = num_tiles_y
        self.num_tiles_x = num_tiles_x
        self.pad_y = (pad_before_y, pad_after_y)
        self.pad_x = (pad_before_x, pad_after_x)
        self.pad_mode = pad_mode

    def __getitem__(self, idx: Tuple[int, int]) -> np.ndarray:
        y, x = idx
        c_y = self.anchor_y + y * self.stride_y
        c_x = self.anchor_x + x * self.stride_x
        min_y = c_y - self.half_tile_size_y
        max_y = min_y + self.tile_size_y - 1
        min_x = c_x - self.half_tile_size_x
        max_x = min_x + self.tile_size_x - 1
        return self.image[..., min_y : max_y + 1, min_x : max_x + 1]


##### Statistics computation to speed up later tile sampling #####


def compute_tile_statistics(
    tilings_dir: Path,
    statistics_name: str,
    statistics_compute_fn: Callable[[np.ndarray], float],
    image_names: Optional[List[str]] = None,
) -> None:
    tiling_dirs = list_files(tilings_dir, file_pattern="*/")
    if image_names is not None:
        tiling_dirs = [d for d in tiling_dirs if d.name in image_names]
    info(
        f'Computing tile statistics "{statistics_name}" for {len(tiling_dirs)} tilings in directory {tilings_dir}.'
    )
    for tiling_dir in tiling_dirs:
        _compute_tile_statistics(tiling_dir, statistics_name, statistics_compute_fn)
    info("Computing tile statistics done.")


def _compute_tile_statistics(
    tiling_dir: Path,
    statistics_name: str,
    statistics_compute_fn: Callable[[np.ndarray], float],
) -> None:
    tile_paths = list_files(tiling_dir, file_extension=".npy")
    info(
        f"Computing tile statistics for tiling {tiling_dir.name} consisting of {len(tile_paths)} tiles."
    )
    lut = np.zeros(len(tile_paths))
    for p, tile_path in enumerate(tile_paths):
        flat_tile_idx, _ = tiling.get_tile_indices(tile_path)
        assert p == flat_tile_idx
        tile = np.load(tile_path)
        lut[p] = statistics_compute_fn(tile)
    info(f"Min value: {lut.min()}, max value: {lut.max()}.")
    lut_save_path = tiling_dir / (statistics_name + ".lut")
    info(f"Saving statistics to lookup table {lut_save_path}.")
    _save_lut(lut_save_path, lut)


def _save_lut(lut_save_path, lut):
    # Open file explicitly. Otherwise numpy will still add its default file extension to the file name. We only want
    # tiles to have that extension to simplify file handling.
    with open(lut_save_path, "wb") as f:
        np.save(f, lut)


def compute_tile_tissue_fg_overlap(
    serial_tilings_dir: Path,
    terminal_tilings_dir: Path,
    serial2terminal: Dict[str, str],
    serial_extract_tissue_fg_fn: Callable[[np.ndarray], np.ndarray],
    terminal_extract_tissue_fg_fn: Callable[[np.ndarray], np.ndarray],
) -> None:
    serial_tiling_dirs = list_files(serial_tilings_dir, file_pattern="*/")
    terminal_tiling_dirs = list_files(terminal_tilings_dir, file_pattern="*/")
    terminal_tiling_dirs = {d.name: d for d in terminal_tiling_dirs}
    info(
        (
            f"Computing tissue foreground overlap ratios for {len(serial_tiling_dirs)} pairs of independent and "
            f"labeled tilings in directories {serial_tilings_dir} and {terminal_tilings_dir}, respectively."
        )
    )
    for serial_tiling_dir in serial_tiling_dirs:
        terminal_image_name = serial2terminal[serial_tiling_dir.name]
        terminal_tiling_dir = terminal_tiling_dirs[terminal_image_name]
        _compute_tile_tissue_fg_overlap(
            serial_tiling_dir,
            terminal_tiling_dir,
            serial2terminal,
            serial_extract_tissue_fg_fn,
            terminal_extract_tissue_fg_fn,
        )
    info("Computing tissue overlap ratios done.")


def _compute_tile_tissue_fg_overlap(
    serial_tiling_dir: Path,
    terminal_tiling_dir: Path,
    serial2terminal: Dict[str, str],
    serial_extract_tissue_fg_fn: Callable[[np.ndarray], np.ndarray],
    terminal_extract_tissue_fg_fn: Callable[[np.ndarray], np.ndarray],
) -> None:
    serial_tile_paths = list_files(serial_tiling_dir, file_extension=".npy")
    terminal_tile_paths = list_files(terminal_tiling_dir, file_extension=".npy")
    info(
        (
            f"Computing overlap ratios for tilings {serial_tiling_dir.name} and {terminal_tiling_dir.name} consisting "
            f"of {len(serial_tile_paths)} tiles, respectively."
        )
    )
    lut = np.zeros(len(serial_tile_paths))
    for p, (serial_tile_path, terminal_tile_path) in enumerate(
        zip(serial_tile_paths, terminal_tile_paths)
    ):
        _check_are_counterparts(
            p, serial_tile_path, terminal_tile_path, serial2terminal
        )
        serial_tile = np.load(serial_tile_path)
        terminal_tile = np.load(terminal_tile_path)
        serial_mask = serial_extract_tissue_fg_fn(serial_tile)
        terminal_mask = terminal_extract_tissue_fg_fn(terminal_tile)
        jaccard = compute_metrics(*confusion_matrix(terminal_mask, serial_mask))[
            "Jaccard"
        ]
        lut[p] = 0.0 if np.isnan(jaccard) else jaccard
    info(f"Min value: {lut.min()}, max value: {lut.max()}.")
    lut_save_path = serial_tiling_dir / "serial_terminal_tissue_fg_overlaps.lut"
    info(f"Saving overlaps to lookup table {lut_save_path}.")
    _save_lut(lut_save_path, lut)


def _check_are_counterparts(
    flat_idx: int,
    serial_tile_path: Path,
    terminal_tile_path: Path,
    serial2terminal: Dict[str, str],
):
    assert (
        serial2terminal[serial_tile_path.parent.name] == terminal_tile_path.parent.name
    )

    (
        serial_valid_tile_shape,
        serial_overlap_shape,
    ) = tiling.get_valid_tile_shape_and_overlap(serial_tile_path.parent.parent)
    (
        terminal_valid_tile_shape,
        terminal_overlap_shape,
    ) = tiling.get_valid_tile_shape_and_overlap(terminal_tile_path.parent.parent)
    assert serial_valid_tile_shape == terminal_valid_tile_shape
    assert serial_overlap_shape == terminal_overlap_shape

    serial_flat_idx, serial_idx_yx = tiling.get_tile_indices(serial_tile_path)
    terminal_flat_idx, terminal_idx_yx = tiling.get_tile_indices(terminal_tile_path)
    assert serial_flat_idx == flat_idx
    assert terminal_flat_idx == flat_idx
    assert serial_idx_yx == terminal_idx_yx


def compute_tile_histograms(tilings_dir: Path) -> None:
    tiling_dirs = list_files(tilings_dir, file_pattern="*/")
    info(
        f"Computing RGB and HSV color histograms for {len(tiling_dirs)} tilings in directory {tilings_dir}."
    )
    for tiling_dir in tiling_dirs:
        _compute_tile_histograms(tiling_dir)
    info("Computing histograms done.")


def _compute_tile_histograms(tiling_dir: Path) -> None:
    tile_paths = list_files(tiling_dir, file_extension=".npy")
    info(
        f"Computing histograms for tiling {tiling_dir.name} consisting of {len(tile_paths)} tiles."
    )
    # #tiles x #channels x width of byte type. Computing histograms at what corresponds to byte precision should be
    # sufficient (and most intuitive).
    rgb = np.zeros((len(tile_paths), 3, 256), dtype="int32")
    hsv = np.zeros((len(tile_paths), 3, 256), dtype="int32")
    for p, tile_path in enumerate(tile_paths):
        flat_tile_idx, _ = tiling.get_tile_indices(tile_path)
        assert p == flat_tile_idx
        rgb_tile = np.load(tile_path)
        hsv_tile = np.moveaxis(rgb2hsv(np.moveaxis(rgb_tile, 0, -1)), -1, 0)
        # Note that we compute the histograms on the entire tile and not just on tissue FG. This is because the
        # existence of tissue gaps around nuclei is an important difference between serial vs terminal data. Thus, it
        # should be incorporated in the tile curation.
        for c in range(3):
            rgb[p, c], _ = np.histogram(rgb_tile[c], bins=256, range=(0, 1))
            hsv[p, c], _ = np.histogram(hsv_tile[c], bins=256, range=(0, 1))
    rgb_save_path = tiling_dir / "rgb_hist.lut"
    hsv_save_path = tiling_dir / "hsv_hist.lut"
    info(
        f"Saving histograms to lookup tables {rgb_save_path} and {hsv_save_path}, respectively."
    )
    _save_lut(rgb_save_path, rgb)
    _save_lut(hsv_save_path, hsv)
