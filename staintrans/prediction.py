from dataclasses import dataclass
from logging import info, warning
from pathlib import Path
from typing import Callable, Dict, Literal, Optional

import numpy as np
import torch
from dataclasses_json import Undefined, dataclass_json
from pytorch_lightning import LightningModule
from skimage.filters import gaussian
from skimage.transform import pyramid_reduce
from skimage.util import img_as_float32
from torch.utils.data import DataLoader

import constants as c
import tiling
import utils
from prediction import PredictionConfig
from preprocessing import tile_image
from processing import Processing, ProcessingConfig
from staintrans.data import StainTransferTestDataset


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class StainTransferConfig(PredictionConfig):
    direction: Literal["A2B", "B2A"] = "A2B"


class StainTransfer(Processing):
    def __init__(
        self,
        working_dir: Path,
        cfg: StainTransferConfig,
        dry_run: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(
            name if name is not None else "staintrans_prediction",
            working_dir,
            cfg,
            dry_run,
        )

    def _run(
        self,
        model: LightningModule,
        pred_data_per_image: Dict[str, StainTransferTestDataset],
        pred_subdir: str = None,
    ):
        working_dir = self._working_dir
        cfg: StainTransferConfig = self._cfg
        dry_run = self._dry_run

        pred_dir = working_dir / "predictions"
        if pred_subdir is not None:
            pred_dir = pred_dir / pred_subdir

        for image_name, image_data in pred_data_per_image.items():
            pred_tilings_dir = pred_dir / "level_0" / "shape_512_512_overlap_256_256"
            pred_tiling_dir = pred_tilings_dir / image_name
            if (pred_tiling_dir / "info.json").exists():
                warning(
                    f"Prediction {pred_tiling_dir} exists. Skipping stain transfer for this image."
                )
                continue
            else:
                info(f"Stain-transferring {image_name}.")
                if cfg.save_pred and not dry_run:
                    info(f"Stain-transferred tiles will be saved to {pred_tiling_dir}.")
                    pred_tiling_dir.mkdir(parents=True, exist_ok=True)

            info(f"Predicting on {len(image_data)} tiles.")
            *_, tile_shape_y, tile_shape_x = image_data[0]["A"].shape
            if tile_shape_y != 2048 or tile_shape_x != 2048:
                raise ValueError(
                    (
                        "Stain transfer only supports tiles with spatial dimensions 2048x2048, but dimensions were "
                        f"{(tile_shape_y, tile_shape_x)}."
                    )
                )

            data_loader = DataLoader(
                image_data,
                batch_size=cfg.batch_size,
                drop_last=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                shuffle=False,
            )

            model.eval()
            with torch.inference_mode():
                # TODO: Use Trainer.predict instead and write out results via hook?
                for batch_idx, batch in enumerate(data_loader):
                    # TODO: Extract the generator for the direction instead of loading the entire model?
                    pred_tiles = model(
                        batch["A"].to(model.device), direction=cfg.direction
                    )
                    pred_tiles = pred_tiles.detach().cpu().numpy()
                    for tile_idx_in_batch, pred_tile in enumerate(pred_tiles):
                        tile_idx = batch_idx * cfg.batch_size + tile_idx_in_batch
                        # Apply the first part of the WSI inference technique from De Bel et al. (2019) by center-
                        # cropping the 2048x2048 prediction down to 1024x1024. The pixel-weighting step will be done in
                        # a later postprocessing.
                        pred_tile = utils.remove_padding(pred_tile, (512, 512))
                        if cfg.save_pred and not dry_run:
                            pred_save_path = (
                                pred_tiling_dir
                                / image_data.entries[tile_idx].samples[0].name
                            )
                            np.save(pred_save_path, pred_tile)
            if cfg.save_pred and not dry_run:
                sample_tiling_dir = image_data.entries[0].samples[0].parent
                # Rewrite tiling info to reflect center crop above.
                original_image_shape, pad_y, pad_x, pad_mode = tiling.get_tiling_info(
                    sample_tiling_dir
                )
                _, (
                    sample_overlap_y,
                    sample_overlap_x,
                ) = tiling.get_valid_tile_shape_and_overlap(sample_tiling_dir.parent)
                if sample_overlap_y != 768 or sample_overlap_x != 768:
                    raise ValueError(
                        (
                            "Stain transfer only supports tiles with spatial overlap 768x768, but overlap was "
                            f"{(sample_overlap_y, sample_overlap_x)}.",
                        )
                    )
                # Center crop decreases tile overlap, and therefore boundary crossing, by 512 in each spatial direction.
                pad_y = max(0, pad_y[0] - 512), max(0, pad_y[1] - 512)
                pad_x = max(0, pad_x[0] - 512), max(0, pad_x[1] - 512)
                tiling.write_tiling_info(
                    pred_tiling_dir,
                    original_image_shape,
                    pad_y,
                    pad_x,
                    pad_mode,
                )


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class StainTransferPostprocessingConfig(ProcessingConfig):
    rescale_data: bool
    extract_tissue_fg_fn: Optional[str] = None
    multiscale_output: bool = True


class StainTransferPostprocessing(Processing):
    def __init__(
        self,
        working_dir: Path,
        cfg: StainTransferPostprocessingConfig,
        dry_run: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(
            name if name is not None else "prediction_postprocessing",
            working_dir,
            cfg,
            dry_run,
        )

    def _run(self, original_samples_dir: str = None, pred_subdir: str = None) -> None:
        working_dir = self._working_dir
        cfg: StainTransferPostprocessingConfig = self._cfg
        dry_run = self._dry_run

        pred_dir = working_dir / "predictions"
        if pred_subdir is not None:
            pred_dir = pred_dir / pred_subdir

        source_pred_tilings_dir = pred_dir / "level_0" / "shape_512_512_overlap_256_256"
        target_pred_tilings_dir = (
            pred_dir / "blended" / "level_0" / "shape_512_512_overlap_0_0"
        )

        extract_tissue_fg_fn = (
            utils.import_function(cfg.extract_tissue_fg_fn)
            if cfg.extract_tissue_fg_fn is not None
            else None
        )

        for source_pred_tiling_dir in utils.list_files(
            source_pred_tilings_dir, file_pattern="*/"
        ):
            image_name = source_pred_tiling_dir.name
            target_pred_tiling_dir = target_pred_tilings_dir / image_name
            if (target_pred_tiling_dir / "info.json").exists():
                warning(
                    f"Postprocessed prediction {target_pred_tiling_dir} exists. Skipping postprocessing for this image."
                )
                continue
            else:
                info(f"Postprocessing {image_name}.")
                if not dry_run:
                    info(
                        f"Postprocessed tiles will be saved to {target_pred_tiling_dir}."
                    )
                    target_pred_tiling_dir.mkdir(parents=True, exist_ok=True)

            (source_tile_size_y, source_tile_size_x), (
                source_overlap_y,
                source_overlap_x,
            ) = tiling.get_valid_tile_shape_and_overlap(source_pred_tiling_dir.parent)
            if (
                source_tile_size_y != 512
                or source_tile_size_x != 512
                or source_overlap_y != 256
                or source_overlap_x != 256
            ):
                raise ValueError(
                    (
                        "Stain transfer postprocessing only supports tiles with spatial valid tile size 512x512 and "
                        f"overlap 256x256, but were {(source_tile_size_y, source_tile_size_x)} and "
                        f"{source_overlap_y, source_overlap_x}, respectively.",
                    )
                )

            image = tiling.TiledImage(source_pred_tiling_dir)
            original_image = (
                tiling.TiledImage(c.scratch_dir / original_samples_dir / image_name)
                if original_samples_dir is not None
                else None
            )
            blended_tiles = []
            blended_tile_indices = []
            for tile, (y, x) in _TileBlending(image).blended_tiles():
                if cfg.rescale_data:
                    # HACK: Rescale from [-1, 1] to [0, 1] which is the expected input domain of the downstream
                    # segmentation network.
                    tile = tile / 2 + 0.5
                if original_image is not None:
                    tile = _correct_tissue_bg(
                        tile, y, x, original_image, extract_tissue_fg_fn
                    )
                blended_tiles.append(tile)
                blended_tile_indices.append((y, x))
                if not dry_run:
                    target_save_path = (
                        target_pred_tiling_dir / image.get_tile_path(y, x).name
                    )
                    np.save(target_save_path, tile)
            del image

            (
                original_image_shape,
                pad_y,
                pad_x,
                pad_mode,
            ) = tiling.get_tiling_info(source_pred_tiling_dir)

            # Blending decreases tile overlap (removes it), and therefore boundary crossing, by 256 in each spatial
            # direction.
            pad_y = max(0, pad_y[0] - 256), max(0, pad_y[1] - 256)
            pad_x = max(0, pad_x[0] - 256), max(0, pad_x[1] - 256)
            tiling.write_tiling_info(
                target_pred_tiling_dir,
                original_image_shape,
                pad_y,
                pad_x,
                pad_mode,
            )

            # Create multi-scale tiles:
            # TODO: Consolidate logic with dataset preparation notebooks.

            blended_image = utils.fuse_sparse_tiles(
                np.stack(blended_tiles),
                blended_tile_indices,
                original_image_shape,
                overlap=None,
                pad_y=pad_y,
                pad_x=pad_x,
            )
            del blended_tiles

            pyramid_levels = [2, 3] if cfg.multiscale_output else []

            last_level = 0
            last_image = blended_image
            del blended_image
            for level in pyramid_levels:
                current_level_target_tilings_dir = (
                    pred_dir
                    / "blended"
                    / f"level_{level}"
                    / "shape_512_512_overlap_0_0"
                )
                current_level_target_tiling_dir = (
                    current_level_target_tilings_dir / image_name
                )
                if not dry_run:
                    current_level_target_tiling_dir.mkdir(parents=True, exist_ok=True)
                info(f"Downsampling postprocessed prediction to level {level}.")
                image = pyramid_reduce(
                    np.moveaxis(last_image, 0, -1),
                    downscale=2 * (level - last_level),
                    multichannel=True,
                )
                image = np.moveaxis(image, -1, 0)
                # We may need to manually crop the individual levels of the pyramid (by at most 1 pixel) since
                # pyramid_reduce seems to use a different rounding mechanism for the shape dimensions than was used in
                # the terminal images.
                #
                # TODO: Remove hard-coded dependency to serial<->terminal mapping. We don't even need the mapping since
                # the aligned serial images have the same shape as their terminal counterparts.
                terminal_image_name = (
                    c.serial2terminal[image_name]
                    if image_name in c.serial2terminal
                    else image_name
                )
                terminal_img_path = (
                    c.scratch_dir
                    / "dataset_208_preprocessed"  # TODO: Remove hard-coded terminal path.
                    / "he"
                    / f"level_{level}"
                    / (terminal_image_name + ".npy")
                )
                *_, terminal_size_y, terminal_size_x = np.load(
                    terminal_img_path, mmap_mode="r"
                ).shape
                image = image[..., :terminal_size_y, :terminal_size_x]
                info(f"Shape of the level: {image.shape}.")
                last_level = level
                last_image = image

                tile_shape = (512, 512)
                stride = (512, 512)

                anchor_y = stride[0] // 2 ** (level + 1)
                anchor_x = stride[1] // 2 ** (level + 1)
                stride_y = stride[0] // 2**level
                stride_x = stride[1] // 2**level

                tile_image(
                    image,
                    current_level_target_tiling_dir,
                    tile_shape,
                    (anchor_y, anchor_x),
                    (stride_y, stride_x),
                    dry_run=dry_run,
                )


def _correct_tissue_bg(
    tile: np.ndarray,
    y: int,
    x: int,
    original_image: tiling.TiledImage,
    extract_tissue_fg_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    original_tile = original_image[y, x]
    mask = extract_tissue_fg_fn(original_tile)
    mask = gaussian(img_as_float32(mask), 1)
    mask_min = mask.min()
    mask_range = mask.max() - mask_min
    if mask_range != 0.0:  # Prevent division by zero.
        mask = (mask - mask_min) / mask_range
    mask = np.expand_dims(mask, 0)
    # Blend stain-transferred tissue and unchanged background to make sure it does not contain any artifacts. Use a soft
    # mask to smoothen borders.
    tile = tile * mask + original_tile * (1 - mask)
    return tile.astype("float32")


class _TileBlending:
    """
    Implements the second part of the WSI inference technique from De Bel et al. (2019) by pixel-weighting overlapping
    tiles.
    """

    _zeros = np.zeros((3, 1024, 1024))
    _zeros.setflags(write=False)
    _wts_y, _wts_x = 1.0 - (np.abs(np.mgrid[0:1024, 0:1024] - 511.5) / 512)
    _wts = np.minimum(_wts_y, _wts_x)
    _wts.setflags(write=False)

    def __init__(self, image: tiling.TiledImage):
        self._image = image
        self._num_tiles_y, self._num_tiles_x = self._image.num_tiles

    def blended_tiles(self):
        for y_c in range(self._image.num_tiles[0]):
            for x_c in range(self._image.num_tiles[1]):
                yield self._blend_tile(y_c, x_c), (y_c, x_c)

    def _blend_tile(self, y_c: int, x_c: int):
        img = np.zeros((3, 512, 512))
        wts = np.zeros((512, 512))

        y_n = y_c - 1  # North
        y_s = y_c + 1  # South
        x_w = x_c - 1  # West
        x_e = x_c + 1  # East

        # Blend in north row:

        img_nw = self._get_tile(y_n, x_w)[..., -256:, -256:]
        img_nc = self._get_tile(y_n, x_c)[..., -256:, 256:-256]
        img_ne = self._get_tile(y_n, x_e)[..., -256:, :256]

        wts_nw = self._get_weights(y_n, x_w)[-256:, -256:]
        wts_nc = self._get_weights(y_n, x_c)[-256:, 256:-256]
        wts_ne = self._get_weights(y_n, x_e)[-256:, :256]

        img[..., :256, :256] += img_nw * wts_nw
        img[..., :256, :] += img_nc * wts_nc
        img[..., :256, 256:] += img_ne * wts_ne

        wts[:256, :256] += wts_nw
        wts[:256, :] += wts_nc
        wts[:256, 256:] += wts_ne

        # Blend in center row:

        img_cw = self._get_tile(y_c, x_w)[..., 256:-256, -256:]
        img_cc = self._get_tile(y_c, x_c)[..., 256:-256, 256:-256]
        img_ce = self._get_tile(y_c, x_e)[..., 256:-256, :256]

        wts_cw = self._get_weights(y_c, x_w)[256:-256, -256:]
        wts_cc = self._get_weights(y_c, x_c)[256:-256, 256:-256]
        wts_ce = self._get_weights(y_c, x_e)[256:-256, :256]

        img[..., :, :256] += img_cw * wts_cw
        img += img_cc * wts_cc
        img[..., :, 256:] += img_ce * wts_ce

        wts[:, :256] += wts_cw
        wts += wts_cc
        wts[:, 256:] += wts_ce

        # Blend in south row:

        img_sw = self._get_tile(y_s, x_w)[..., :256, -256:]
        img_sc = self._get_tile(y_s, x_c)[..., :256, 256:-256]
        img_se = self._get_tile(y_s, x_e)[..., :256, :256]

        wts_sw = self._get_weights(y_s, x_w)[:256, -256:]
        wts_sc = self._get_weights(y_s, x_c)[:256, 256:-256]
        wts_se = self._get_weights(y_s, x_e)[:256, :256]

        img[..., 256:, :256] += img_sw * wts_sw
        img[..., 256:, :] += img_sc * wts_sc
        img[..., 256:, 256:] += img_se * wts_se

        wts[256:, :256] += wts_sw
        wts[256:, :] += wts_sc
        wts[256:, 256:] += wts_se

        # Normalize
        img /= wts

        img = img.astype("float32")
        return img

    def _get_tile(self, y: int, x: int):
        return (
            self._image[y, x] if self._is_tile_in_bounds(y, x) else _TileBlending._zeros
        )

    def _get_weights(self, y: int, x: int):
        return (
            _TileBlending._wts
            if self._is_tile_in_bounds(y, x)
            else _TileBlending._zeros[0, ...]
        )

    def _is_tile_in_bounds(self, y: int, x: int):
        return 0 <= y < self._num_tiles_y and 0 <= x < self._num_tiles_x
