import os
from dataclasses import dataclass, field
from logging import info, warning
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Literal, Optional, Tuple

from dataclasses_json import Undefined, dataclass_json
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

import constants as c
from data import ResampleDatasetCallback
from processing import Processing, ProcessingConfig
from utils import list_files


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class TrainingConfig(ProcessingConfig):
    train_batch_size: int
    val_batch_size: int
    epochs: int
    val_frequency: int = 1
    checkpoints_file_name_metrics: Optional[List[str]] = None
    checkpoints_monitored_metric: Optional[Tuple[str, Literal["min", "max"]]] = None
    early_stopping: bool = False
    early_stopping_patience: int = 5
    log_learning_rate: bool = False
    progress: bool = False
    continue_training: bool = False
    trainer_kwargs: Dict[str, Any] = field(default_factory=lambda: {})


class Training(Processing):
    def __init__(
        self,
        working_dir: Path,
        cfg: TrainingConfig,
        dry_run: bool = False,
    ):
        super().__init__("training", working_dir, cfg, dry_run)

    def _run(
        self,
        model: LightningModule,
        train_data: Dataset,
        val_data: Optional[Dataset],
        debug: bool = False,
    ) -> LightningModule:
        working_dir = self._working_dir
        cfg: TrainingConfig = self._cfg
        dry_run = self._dry_run

        if debug:
            warning("NOTE: Training will run in debug mode!")

        num_workers = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", 5)) - 1
        info(f"Data will be loaded using {num_workers} workers.")
        train_loader = DataLoader(
            train_data,
            batch_size=cfg.train_batch_size,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
        )
        if val_data is not None:
            val_loader = DataLoader(
                val_data,
                batch_size=cfg.val_batch_size,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False,
            )
        else:
            info("No validation data provided. Validation will be skipped.")
            val_loader = None

        callbacks = []
        if not debug:
            callbacks.append(ResampleDatasetCallback(train_data, "training"))
            callbacks.append(ResampleDatasetCallback(val_data, "validation"))
        else:
            info("Debug mode: Data will not be resampled between epochs (if at all).")
        if cfg.progress:
            callbacks.append(RichProgressBar())
        # Required when continuing training, even when in a dry run.
        checkpoints_dir = working_dir / "checkpoints"
        if not dry_run:
            checkpoints_dir.mkdir(exist_ok=cfg.continue_training)
            checkpoints_monitored_metric = None
            checkpoints_monitored_metric_mode = None
            if cfg.checkpoints_file_name_metrics is None:
                if val_loader is None:
                    raise RuntimeError(
                        "Cannot use validation-based checkpoint name format without validation data."
                    )
                checkpoints_name_metrics = ["val_loss", "val_accuracy"]
                if cfg.checkpoints_monitored_metric is None:
                    checkpoints_monitored_metric = "val_loss"
                    checkpoints_monitored_metric_mode = "min"
            else:
                checkpoints_name_metrics = cfg.checkpoints_file_name_metrics
            checkpoints_name_metrics = [
                "{" + m + ":.4f}" for m in checkpoints_name_metrics
            ]
            checkpoints_name = "{epoch:03d}_" + "_".join(checkpoints_name_metrics)
            info(
                f"Checkpoints will be saved to files like {checkpoints_dir / checkpoints_name}."
            )
            if (
                checkpoints_monitored_metric is None
                and cfg.checkpoints_monitored_metric is not None
            ):
                checkpoints_monitored_metric = cfg.checkpoints_monitored_metric[0]
                checkpoints_monitored_metric_mode = cfg.checkpoints_monitored_metric[1]
            if checkpoints_monitored_metric is not None:
                info(
                    (
                        f"The monitored quantity to determine the best checkpoint is {checkpoints_monitored_metric}, "
                        f"mode {checkpoints_monitored_metric_mode}."
                    )
                )
            else:
                warning(
                    (
                        "No quantity to monitor specified for model checkpointing. The training process will return "
                        'the final checkpoint as "best" checkpoint.'
                    )
                )
            model_checkpoint = ModelCheckpoint(
                dirpath=checkpoints_dir,
                filename=checkpoints_name,
                monitor=checkpoints_monitored_metric,
                save_top_k=-1,
                mode=checkpoints_monitored_metric_mode,
                every_n_epochs=1,
            )
            callbacks.append(model_checkpoint)

            loggers = []
            tb_logs_dir = c.logs_dir
            tb_log_name = working_dir.name
            info(f"TensorBoard logs will be written to {tb_logs_dir / tb_log_name}.")
            loggers.append(TensorBoardLogger(str(tb_logs_dir), name=tb_log_name))
            loggers.append(CSVLogger(str(working_dir), name="metrics"))
        elif debug:
            loggers = [
                CSVLogger(str(c.scratch_dir), name=f"{working_dir.name}_debug_metrics")
            ]
        else:
            loggers = None
        if cfg.early_stopping:
            if val_loader is None:
                raise RuntimeError("Cannot use early stopping without validation data.")
            info("Training will employ early stopping.")
            callbacks.append(
                EarlyStopping(
                    "val_loss", patience=cfg.early_stopping_patience, verbose=True
                )
            )
        if cfg.log_learning_rate:
            if not dry_run:
                info("Learning rate will be logged.")
                callbacks.append(LearningRateMonitor(logging_interval="epoch"))
            else:
                warning("Learning rate logging is disabled in dry run.")
        if hasattr(model, "intermediate_output_visualizer"):
            model.intermediate_output_visualizer.set_batch_info(
                len(train_loader),
                cfg.train_batch_size,
                len(val_loader) if val_data is not None else 0,
                cfg.val_batch_size,
            )
            callbacks.append(model.intermediate_output_visualizer)
            info("Intermediate outputs will be visualized.")
        if not debug:
            overfit_batches = 0
        else:
            overfit_batches = 1
            info(
                f"Debug mode: Intentionally overfitting on {overfit_batches} batch(es)."
            )
        trainer = Trainer(
            callbacks=callbacks,
            check_val_every_n_epoch=cfg.val_frequency,
            enable_checkpointing=not dry_run,
            enable_progress_bar=cfg.progress,
            default_root_dir=str(working_dir),
            gpus=1,
            logger=loggers,
            max_epochs=cfg.epochs,
            overfit_batches=overfit_batches,
            **cfg.trainer_kwargs,
        )
        if cfg.continue_training:
            # Latest checkpoint
            checkpoint_path = list_files(checkpoints_dir, file_extension=".ckpt")[-1]
            info(f"Continuing training from latest checkpoint {checkpoint_path}.")
        else:
            checkpoint_path = None
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)

        if not dry_run:
            best_model_path = trainer.checkpoint_callback.best_model_path
            info(f"Best (or final) model checkpoint is {Path(best_model_path).name}.")
            copy2(best_model_path, checkpoints_dir / "best.ckpt")
            return type(model).load_from_checkpoint(best_model_path)
        else:
            warning("Dry run: Returning final model instead of best checkpoint.")
            return model
