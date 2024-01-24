"""
Utilities to deal with (the logical structure of) tiled data sets and images. The expected structure of a tiled data set's directory is as follows:

Base directory: named "<dataset_name>_tiled"
    |-> Modality directory: e.g. "cksox", "he", etc.
    ... Can include nested preprocessing directories, e.g. "cksox/masks_thr_1000/holes_filled_dia_BR_15_LU_50".
        |-> Level directory: named "level_<level>", where <level> denotes the level in the original image's pyramid
            |-> Tilings directory: named "shape_<valid_tile_size_y>_<valid_tile_size_x>_overlap_<overlap_y>_<overlap_x>"
                |-> Tiling directory: named after the original image file
                    |-> Tiles: named "<original_image_file>_tile_<flat_tile_idx>_y_<tile_idx_y>_x_<tile_idx_x>.npy"*
                    |-> info.json: contains information about the original image, e.g. its shape
                    |-> Optionally: lookup tables containing tile statistics, e.g. "label_fg_ratios.lut"

*At some point, it could make sense to switch to a chunked file format like HDF5 instead of having each tile in its own
file.
"""
import json
import re
from pathlib import Path
from typing import Tuple

import numpy as np

import utils

_tile_indices_pattern = re.compile(".*_tile_(.*)_y_(.*)_x_(.*)\.npy")


def get_valid_tile_shape_and_overlap(
    tilings_dir: Path,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    m = re.match("shape_(.*)_(.*)_overlap_(.*)_(.*)", tilings_dir.name)
    valid_tile_shape = int(m.group(1)), int(m.group(2))
    overlap_shape = int(m.group(3)), int(m.group(4))
    return valid_tile_shape, overlap_shape


def get_num_tiles(tiling_dir: Path) -> Tuple[int, int]:
    max_tile_path = utils.list_files(tiling_dir, file_extension=".npy")[-1]
    _, (max_tile_idx_y, max_tile_idx_x) = get_tile_indices(max_tile_path)
    num_tiles_y = int(max_tile_idx_y) + 1
    num_tiles_x = int(max_tile_idx_x) + 1
    return num_tiles_y, num_tiles_x


def write_tiling_info(
    tiling_dir: Path,
    original_image_shape: Tuple[int, int],
    pad_y: Tuple[int, int],
    pad_x: Tuple[int, int],
    pad_mode: str,
) -> None:
    if pad_y[0] < 0 or pad_y[1] < 0 or pad_x[0] < 0 or pad_x[1] < 0:
        raise ValueError(f"Negative padding encountered: y={pad_y}, x={pad_x}.")
    with open(tiling_dir / "info.json", "w") as f:
        json.dump(
            {
                "original_image_shape": {
                    "y": original_image_shape[-2],
                    "x": original_image_shape[-1],
                },
                "padding": {
                    "y": {"before": pad_y[0], "after": pad_y[1]},
                    "x": {"before": pad_x[0], "after": pad_x[1]},
                    "mode": pad_mode,
                },
            },
            f,
            indent=4,
        )


def get_tiling_info(
    tiling_dir: Path,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], str]:
    with open(tiling_dir / "info.json") as f:
        tiling_info = json.load(f)
    original_image_shape_entry = tiling_info["original_image_shape"]
    original_image_shape = (
        original_image_shape_entry["y"],
        original_image_shape_entry["x"],
    )
    pad_entry = tiling_info["padding"]
    pad_y_entry = pad_entry["y"]
    pad_y = pad_y_entry["before"], pad_y_entry["after"]
    pad_x_entry = pad_entry["x"]
    pad_x = pad_x_entry["before"], pad_x_entry["after"]
    pad_mode = pad_entry["mode"]
    if pad_y[0] < 0 or pad_y[1] < 0 or pad_x[0] < 0 or pad_x[1] < 0:
        raise ValueError(f"Negative padding encountered: y={pad_y}, x={pad_x}.")
    return original_image_shape, pad_y, pad_x, pad_mode


def read_tiling(tiling_dir: Path) -> np.ndarray:
    _, (overlap_y, overlap_x) = get_valid_tile_shape_and_overlap(tiling_dir.parent)
    tile_paths = utils.list_files(tiling_dir, file_extension=".npy")
    tiles = []
    tile_indices = []
    for tile_path in tile_paths:
        tiles.append(np.load(tile_path))
        tile_indices.append(get_tile_indices(tile_path)[1])
    tiles = np.stack(tiles)
    original_image_shape, pad_y, pad_x, _ = get_tiling_info(tiling_dir)
    return utils.fuse_sparse_tiles(
        tiles,
        tile_indices,
        original_image_shape,
        (overlap_y, overlap_x),
        pad_y,
        pad_x,
    )


def get_tile_indices(tile_path: Path) -> Tuple[int, Tuple[int, int]]:
    m = _tile_indices_pattern.match(tile_path.name)
    flat_idx = int(m.group(1))
    idx_y, idx_x = int(m.group(2)), int(m.group(3))
    return flat_idx, (idx_y, idx_x)


class TiledImage:
    def __init__(self, tiling_dir: Path, in_memory: bool = False):
        self._tile_paths = utils.list_files(tiling_dir, file_extension=".npy")
        self._num_tiles_y, self._num_tiles_x = get_num_tiles(tiling_dir)
        if in_memory:
            self._tiles = [np.load(p) for p in self._tile_paths]
        else:
            self._tiles = None
        self.path = tiling_dir

    def __len__(self) -> int:
        return len(self._tile_paths)

    @property
    def num_tiles(self) -> Tuple[int, int]:
        return self._num_tiles_y, self._num_tiles_x

    def get_flat_index(self, *idx: Tuple[int, ...]) -> int:
        if len(idx) > 1:
            if (
                0 > idx[0]
                or idx[0] >= self._num_tiles_y
                or 0 > idx[1]
                or idx[1] >= self._num_tiles_x
            ):
                raise IndexError(
                    f"Index {idx} exceeds bounds {0, 0} to {self._num_tiles_y, self._num_tiles_x}."
                )
            return idx[0] * self._num_tiles_x + idx[1]
        else:
            return idx

    def get_tile_path(self, *idx: Tuple[int, ...]) -> Path:
        return self._tile_paths[self.get_flat_index(*idx)]

    def __getitem__(self, idx: Tuple[int, ...]) -> np.ndarray:
        if self._tiles is not None:
            return self._tiles[self.get_flat_index(*idx)]
        else:
            return np.load(self.get_tile_path(*idx))

    def __repr__(self):
        return f"Image {self.path.name} made up of {len(self._tile_paths)} tiles"
