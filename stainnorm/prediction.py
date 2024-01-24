from logging import info, warning
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tiling
from processing import Processing
from pytorch_lightning import LightningModule

from data import MultiscaleDataset
from prediction import PredictionConfig, predict_on_tiles


class StainNormalization(Processing):
    def __init__(
        self,
        working_dir: Path,
        cfg: PredictionConfig,
        dry_run: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(
            name if name is not None else "prediction",
            working_dir,
            cfg,
            dry_run,
        )

    def _run(
        self,
        model: LightningModule,
        pred_data_per_image: Dict[str, MultiscaleDataset],
        pred_subdir: str = None,
    ) -> None:
        working_dir = self._working_dir
        cfg: PredictionConfig = self._cfg
        dry_run = self._dry_run

        pred_dir = working_dir / "predictions"
        if pred_subdir is not None:
            pred_dir = pred_dir / pred_subdir
        if cfg.save_pred and not dry_run:
            pred_dir.mkdir(parents=True, exist_ok=True)

        for image_name, image_data in pred_data_per_image.items():
            pred_tilings_dir = (
                pred_dir / image_data.entries[0].samples[0].parent.parent.name
            )
            pred_tiling_dir = pred_tilings_dir / image_name
            if (pred_tiling_dir / "info.json").exists():
                warning(
                    f"Prediction {pred_tiling_dir} exists. Skipping normalization for this image."
                )
                continue
            else:
                info(f"Normalizing {image_name}.")
                if cfg.save_pred and not dry_run:
                    pred_tiling_dir.mkdir(parents=True, exist_ok=True)

            pred_tiles = predict_on_tiles(
                model, image_data, cfg.batch_size, cfg.num_workers
            )
            pred_tiles = np.squeeze(pred_tiles)

            if cfg.save_pred and not dry_run:
                info(f"Saving normalized tiles to {pred_tiling_dir}.")
                for tile_idx, tile in enumerate(pred_tiles):
                    pred_save_path = (
                        pred_tiling_dir / image_data.entries[tile_idx].samples[0].name
                    )
                    np.save(pred_save_path, tile)
                sample_tiling_dir = image_data.entries[0].samples[0].parent
                tiling.write_tiling_info(
                    pred_tiling_dir, *tiling.get_tiling_info(sample_tiling_dir)
                )
