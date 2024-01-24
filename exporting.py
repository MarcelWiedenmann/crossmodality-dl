import csv
from logging import info
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tifffile
from skimage import img_as_uint
from skimage.transform import pyramid_reduce

import constants as c
import tiling
import utils


class ImageExport:
    def __init__(self):
        self._metadata = None
        self._channel_paths = []
        self._channels = {}

    def add_metadata(self, metadata: Union[Path, Dict[str, Any]]):
        if self._metadata is not None:
            raise RuntimeError("Metadata already set.")
        metadata = (
            utils.read_omero_metadata(metadata)
            if isinstance(metadata, Path)
            else metadata
        )
        self._metadata = metadata
        return self

    def add_channel(self, name: str, source: Union[Path, np.ndarray]):
        source = self._read_image(source) if isinstance(source, Path) else source
        self._add_channel(name, source)
        return self

    def add_multichannel(self, names: List[str], source: Union[Path, np.ndarray]):
        mc = self._read_image(source) if isinstance(source, Path) else source
        if mc.shape[-1] == len(names):
            mc = np.moveaxis(mc, -1, 0)
        for i, c in enumerate(mc):
            self._add_channel(names[i], c)
        return self

    def add_rgb(self, source: Union[Path, np.ndarray]):
        self.add_multichannel(["RED", "GREEN", "BLUE"], source)
        return self

    def add_prediction(self, source: Union[Path, np.ndarray], name: str = "Prediction"):
        prediction = self._read_image(source) if isinstance(source, Path) else source
        if not np.issubdtype(prediction.dtype, np.floating):
            raise TypeError("Prediction is expected to be floating point.")
        self._add_channel(f"{name}_(probabilities)", prediction)
        self._add_channel(f"{name}_(threshold=0.5)", prediction > 0.5)
        return self

    def _read_image(self, path: Path) -> np.ndarray:
        self._channel_paths.append(path)
        return tiling.read_tiling(path) if path.is_dir() else np.load(path)

    def _add_channel(self, name: str, channel: np.ndarray):
        if name in self._channels:
            raise RuntimeError(f'Channel "{name}" already exists.')
        self._channels[name] = img_as_uint(channel)

    def export(self, export_path: Path, dry_run: bool = False):
        info(
            f"The channels in the exported image are derived from the following sources:\n{pformat(self._channel_paths)}."
        )
        channels = list(self._channels.values())
        channel_names = list(self._channels.keys())
        if not dry_run:
            export_tiff(channels, channel_names, self._metadata, export_path)


def export_tiff(
    channels: List[np.ndarray],
    channel_names: List[str],
    base_metadata: Dict[str, Any],
    export_path: Path,
):
    if "OME" in base_metadata:
        base_metadata = base_metadata["OME"]
    base_image_metadata = base_metadata["Image"]
    base_pixels_metadata = base_image_metadata["Pixels"]
    try:
        # Only exists if pyramidal.
        base_pyramid_metadata = base_metadata["StructuredAnnotations"]["MapAnnotation"][
            "Value"
        ]["M"]
    except KeyError:
        base_pyramid_metadata = None
    metadata = {
        "OME": {
            # tiffile does not support "Instrument" and "StructuredAnnotations" at the moment. So the following two
            # lines will not have any effect. But it does not hurt keeping them.
            "Instrument": base_metadata.get("Instrument"),
            "StructuredAnnotations": base_metadata.get(
                "StructuredAnnotations"
            ),  # Only exists if pyramidal.
            "Image": {
                "Name": base_image_metadata.get("@Name"),
                # "ObjectiveSettings" is also not supported.
                "ObjectiveSettings": base_image_metadata.get("ObjectiveSettings"),
                "Pixels": {
                    "PhysicalSizeX": base_pixels_metadata["@PhysicalSizeX"],
                    "PhysicalSizeXUnit": base_pixels_metadata["@PhysicalSizeXUnit"],
                    "PhysicalSizeY": base_pixels_metadata["@PhysicalSizeY"],
                    "PhysicalSizeYUnit": base_pixels_metadata["@PhysicalSizeYUnit"],
                    "Channel": [{"Name": name} for name in channel_names],
                },
            },
        }
    }
    pyramid = []
    if base_pyramid_metadata is not None:
        for level_index in range(len(base_pyramid_metadata)):
            size_x, size_y = tuple(
                int(dim)
                for dim in base_pyramid_metadata[level_index]["#text"].split(" ")
            )
            pyramid.append((size_y, size_x))
    else:
        full_size_y, full_size_x = channels[0].shape
        for i in range(1, 8):
            pyramid.append((full_size_y // 2**i, full_size_x // 2**i))

    info(f"Exporting TIFF containing channels ({channel_names}) to {export_path}.")
    with tifffile.TiffWriter(export_path, bigtiff=True, ome=True) as tif:
        tile_shape = (1024, 1024)
        options = dict(photometric="minisblack", tile=tile_shape)
        if metadata is not None:
            options["metadata"] = metadata
        info(
            (
                f"Writing {len(channels)} channels at {len(pyramid) + 1} pyramid levels and in tiles "
                f"of shape {tile_shape}."
            )
        )
        for i, level in enumerate(_generate_pyramid_levels(channels, pyramid)):
            if i == 0:
                options["subifds"] = len(pyramid)
            else:
                if i == 1:
                    del options["subifds"]
                options["subfiletype"] = 1
            info(f"Writing pyramid level {i}.")
            _write_level(tif, level, options)
    info("Exporting TIFF done.")


def _generate_pyramid_levels(
    channels: List[np.ndarray], pyramid: List[Tuple[int, int]]
):
    level = channels
    yield level
    for (size_y, size_x) in pyramid:
        info("Generating next pyramid level.")
        # We need to manually resize the individual levels of the pyramid since pyramid_reduce seems to use a
        # different rounding mechanism for the shape dimensions than was used in the original image (in case the pyramid
        # is derived from the original image metadata).
        level = [pyramid_reduce(c)[:size_y, :size_x] for c in level]
        # pyramid_reduce produces float outputs, convert back into uints but retain references to floats to avoid
        # amplyfing any rounding errors through the levels.
        # Note that we do not treat masks separately, i.e., their boolean values will also be interpolated to floats.
        yield [img_as_uint(c) for c in level]


def _write_level(tif: tifffile.TiffWriter, channels: List[np.ndarray], options: Dict):
    level_shape = (len(channels),) + channels[0].shape
    level_dtype = channels[0].dtype
    info(f"Level: shape={level_shape}, dtype={level_dtype}.")
    if not all(c.shape == level_shape[1:] and c.dtype == level_dtype for c in channels):
        raise ValueError(
            "Not all channels in the level share the same shape and dtype."
        )
    tif.write(
        data=_generate_tiles(channels, options["tile"]),
        shape=level_shape,
        dtype=level_dtype,
        **options,
    )


def _generate_tiles(channels: List[np.ndarray], tile_shape: Tuple[int, int]):
    channel_size_y, channel_size_x = channels[0].shape
    tile_size_y, tile_size_x = tile_shape
    for i, channel in enumerate(channels):
        info(f"Generating tiles for channel {i}.")
        for y in range(0, channel_size_y, tile_size_y):
            for x in range(0, channel_size_x, tile_size_x):
                # Copying apparently improves performance when writing to disk (contiguous array). Adopted without
                # testing.
                yield channel[y : y + tile_size_y, x : x + tile_size_x].copy()


def create_upload_csv(
    save_dir: Path, export_dataset_name: str, export_file_names: List[str]
) -> Path:
    upload_csv_path = save_dir / f"{export_dataset_name}_uploadTemplate.csv"
    with open(upload_csv_path, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Barcode",
                "Project Name",
                "Dataset Name",
                "Image Name",
                "Stain Type",
                "Technology",
                "Acquisition",
                "Antibody",
                "Channel",
                "Species",
                "Block Number",
                "Tissue Type",
                "Pathologist",
            ]
        )
        for i, file_name in enumerate(export_file_names, 1):
            try:
                block_number_and_tissue_type = (
                    "HP-" + file_name.split("HP-")[1].split("_")[0].split(".")[0]
                )
                block_number = block_number_and_tissue_type[:-2]
                tissue_type = block_number_and_tissue_type[-2:]
            except IndexError:
                block_number = ""
                tissue_type = ""
            writer.writerow(
                [
                    i,
                    c.publish_project_name,
                    export_dataset_name,
                    file_name,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "Human",
                    block_number,
                    tissue_type,
                    "",
                ]
            )
    return upload_csv_path
