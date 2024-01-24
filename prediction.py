from dataclasses import dataclass
from logging import info, warning
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dataclasses_json import Undefined, dataclass_json
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import RichProgressBar
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset

import tiling
import utils
from data import MultiscaleDataset
from processing import Processing, ProcessingConfig


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class PredictionConfig(ProcessingConfig):
    batch_size: int
    num_workers: int = 4
    save_pred: bool = True


class Prediction(Processing):
    def __init__(
        self,
        working_dir: Path,
        cfg: PredictionConfig,
        dry_run: bool = False,
        name: Optional[str] = None,
    ) -> None:
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
        pred_mask_per_image: Optional[Dict[str, List[Path]]] = None,
        pred_subdir: str = None,
    ):
        """
        pred_mask_per_image can be used to mask out regions in the input (e.g. glass slide) that should not be evaluated
        when computing the performance metrics for this prediction. Predictions in these regions will be counted as true
        negatives.
        """
        working_dir = self._working_dir
        cfg: PredictionConfig = self._cfg
        dry_run = self._dry_run

        pred_dir = working_dir / "predictions"
        if pred_subdir is not None:
            pred_dir = pred_dir / pred_subdir
        if not dry_run:
            pred_dir.mkdir(parents=True, exist_ok=True)

        if (pred_dir / "metrics.csv").exists():
            warning(f"Predictions in {pred_dir} already exist. Skipping.")
            return

        metrics = []
        for image_name, image_data in pred_data_per_image.items():
            info(f"Predicting on {image_name}.")
            pred_tiles = predict_on_tiles(
                model, image_data, cfg.batch_size, cfg.num_workers
            )
            pred_tiles = np.squeeze(pred_tiles)

            if cfg.save_pred and not dry_run:
                tile_shape = pred_tiles[0].shape
                pred_tilings_dir = (
                    pred_dir / f"shape_{tile_shape[-2]}_{tile_shape[-1]}_overlap_0_0"
                )
                pred_tilings_dir.mkdir(exist_ok=True)
                pred_tiling_dir = pred_tilings_dir / image_name
                pred_tiling_dir.mkdir()
                info(f"Saving predicted tiles to {pred_tiling_dir}.")
                for tile_idx, tile in enumerate(pred_tiles):
                    pred_save_path = (
                        pred_tiling_dir / image_data.entries[tile_idx].samples[0].name
                    )
                    np.save(pred_save_path, tile)
                sample_tiling_dir = image_data.entries[0].samples[0].parent
                original_image_shape, pad_y, pad_x, pad_mode = tiling.get_tiling_info(
                    sample_tiling_dir
                )
                # Sample may have overlap (at least on disk), prediction does not. Remove if present.
                _, (
                    sample_overlap_y,
                    sample_overlap_x,
                ) = tiling.get_valid_tile_shape_and_overlap(sample_tiling_dir.parent)
                pad_y = max(0, pad_y[0] - sample_overlap_y), max(
                    0, pad_y[1] - sample_overlap_y
                )
                pad_x = max(0, pad_x[0] - sample_overlap_x), max(
                    0, pad_x[1] - sample_overlap_x
                )
                tiling.write_tiling_info(
                    pred_tiling_dir,
                    original_image_shape,
                    pad_y,
                    pad_x,
                    pad_mode,
                )

            if image_data.has_targets:
                target_tile_paths = [e.targets[0] for e in image_data.entries]
                mask_tile_paths = (
                    pred_mask_per_image[image_name]
                    if pred_mask_per_image is not None
                    else None
                )
                metrics.extend(
                    self._compute_metrics(
                        image_name, target_tile_paths, pred_tiles, mask_tile_paths
                    )
                )

        if len(metrics) > 0:
            metrics.append(
                aggregate_metrics(
                    working_dir.name,
                    [m for m in metrics if m["Prediction"] != "all negative"],
                )
            )
            metrics = pd.DataFrame(metrics)
            if not dry_run:
                metrics.to_csv(pred_dir / "metrics.csv")
            info(
                "Validation metrics:\n"
                + tabulate(metrics, headers="keys", tablefmt="psql", floatfmt=".4f")
            )

    def _compute_metrics(
        self, image_name, target_tile_paths, pred_tiles, mask_tile_paths
    ):
        target_tiles = np.stack([np.load(p) for p in target_tile_paths])
        pred_tiles_binarized = pred_tiles > 0.5
        mask_tiles = (
            np.stack([np.load(p) for p in mask_tile_paths])
            if mask_tile_paths is not None
            else None
        )
        return (
            compute_metrics(
                image_name,
                self._working_dir.name,
                target_tiles,
                pred_tiles_binarized,
                mask_tiles,
            ),
            compute_metrics(
                image_name,
                "all negative",
                target_tiles,
                np.zeros(pred_tiles.shape, dtype="bool"),
            ),
        )


def predict_on_tiles(
    model: LightningModule,
    pred_data: Dataset,
    batch_size: int,
    num_workers: int = 4,
    progress: bool = False,
) -> np.ndarray:
    info(f"Predicting on {len(pred_data)} tiles.")
    data_loader = DataLoader(
        pred_data,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )
    trainer = Trainer(callbacks=[RichProgressBar()] if progress else None, gpus=1)
    preds = trainer.predict(model, data_loader)
    return np.concatenate([b.numpy() for b in preds], axis=0)


def compute_metrics(
    image_name: str,
    pred_name: str,
    target_tiles: np.ndarray,
    pred_tiles: np.ndarray,
    mask_tiles: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    tn, fp, fn, tp = utils.confusion_matrix(target_tiles, pred_tiles, mask_tiles)
    return {
        **{
            "Image": image_name,
            "Prediction": pred_name,
            "Number of tiles": pred_tiles.shape[0],
        },
        **utils.compute_metrics(tn, fp, fn, tp),
    }


def aggregate_metrics(pred_name: str, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    num_tiles = 0
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for m in metrics:
        num_tiles += m["Number of tiles"]
        tn += m["TN"]
        fp += m["FP"]
        fn += m["FN"]
        tp += m["TP"]

    return {
        **{"Image": "all", "Prediction": pred_name, "Number of tiles": num_tiles},
        **utils.compute_metrics(tn, fp, fn, tp),
    }
