"""
Evaluate the multi-scale segmentation model. Evaluation can optionally be performed using model versions trained with
stain color augmentation, and and can optionally be done on data that has previously been stain normalized or stain
transferred. The stain normalization parameters used in the conducted parameters were (stainnorm_color_space,
stainnorm_factor):
    - "hed", 0.00625,
    - "hed", 0.0125,
    - "hed", 0.025,
    - "hsv", 1.0.
"""
from logging import info
from pathlib import Path
from typing import Dict, List, Optional

import configure_logging  # Just needs to be imported.
import constants as c
import crossvalidation as cv
from data import DataEntry, MultiscaleDataset, NumpyDatasetSource
from data_collection import DataCollection, DataCollectionConfig
from experiment import Experiment
from multiscale import get_experiment_name, get_segm_model_experiment_name
from multiscale.model import MsY2Net
from prediction import Prediction, PredictionConfig
from staintrans import get_pred_subdir


def evaluate_multiscale(opts):
    terminal_targets_dir = cv.get_terminal_targets_dir()
    serial_targets_dir = cv.get_serial_targets_dir()
    serial_tissue_masks_dir = cv.get_serial_tissue_masks_dir()

    cv_folds = cv.get_enumerated_folds(opts.cv_folds)
    for i, (
        *_,
        terminal_test_images,
        serial_test_images,
    ) in cv_folds:
        if opts.stainnorm_color_space is not None:
            terminal_sample_dirs = cv.get_normalized_terminal_sample_dirs(
                opts.stainnorm_color_space, opts.stainnorm_factor, i
            )
            serial_sample_dirs = cv.get_normalized_serial_sample_dirs(
                opts.stainnorm_color_space, opts.stainnorm_factor, i
            )
        elif opts.staintrans_name is not None:
            terminal_sample_dirs = cv.get_staintrans_terminal_sample_dirs(
                opts.staintrans_name, opts.staintrans_checkpoint_epoch, i
            )
            serial_sample_dirs = cv.get_staintrans_serial_sample_dirs(
                opts.staintrans_name, opts.staintrans_checkpoint_epoch, i
            )
        else:
            terminal_sample_dirs = cv.get_terminal_sample_dirs()
            serial_sample_dirs = cv.get_serial_sample_dirs()
        with Experiment(name=get_experiment_name(opts, i), seed=c.seed) as exp:
            checkpoint_path = (
                c.experiments_dir
                / get_segm_model_experiment_name(
                    opts,
                    i,
                )
                / "checkpoints"
                / "best.ckpt"
            )
            info(f"Loading segmentation model from checkpoint path: {checkpoint_path}.")
            model = MsY2Net.load_from_checkpoint(checkpoint_path)

            if opts.terminal:
                *_, terminal_test_entries = DataCollection(
                    exp.working_dir,
                    DataCollectionConfig(
                        sample_dirs=terminal_sample_dirs,
                        targets_dir=terminal_targets_dir,
                        test_images=terminal_test_images,
                    ),
                    exp.dry_run,
                    name="terminal_test_data_collection",
                ).run()
                evaluate_multiscale_on_entries(
                    exp,
                    model,
                    "terminal",
                    terminal_test_entries,
                    tissue_masks=None,
                    pred_subdir="terminal/" + get_pred_subdir(opts),
                    save_pred=opts.save_pred,
                )

            if opts.serial:
                serial_targets = [
                    c.serial2terminal[serial] for serial in serial_test_images
                ]
                *_, serial_test_entries = DataCollection(
                    exp.working_dir,
                    DataCollectionConfig(
                        sample_dirs=serial_sample_dirs,
                        targets_dir=serial_targets_dir,
                        test_images=serial_test_images,
                        test_targets=serial_targets,
                    ),
                    exp.dry_run,
                    name="serial_test_data_collection",
                ).run()

                if opts.staintrans_name is not None:
                    *_, serial_test_tissue_masks_entries = DataCollection(
                        exp.working_dir,
                        DataCollectionConfig(
                            sample_dirs=[serial_tissue_masks_dir],
                            test_images=serial_test_images,
                        ),
                        exp.dry_run,
                        name="serial_tissue_masks_collection",
                    ).run()
                    serial_tissue_masks = {
                        image_name: [e.samples[0] for e in entries]
                        for image_name, entries in serial_test_tissue_masks_entries.items()
                    }
                    # Sanity check
                    for image_name, entries in serial_test_entries.items():
                        for entry, mask_path in zip(
                            entries, serial_tissue_masks[image_name]
                        ):
                            assert entry.samples[0].name == mask_path.name
                else:
                    serial_tissue_masks = None

                evaluate_multiscale_on_entries(
                    exp,
                    model,
                    "serial",
                    serial_test_entries,
                    tissue_masks=serial_tissue_masks,
                    pred_subdir="serial/" + get_pred_subdir(opts),
                    save_pred=opts.save_pred,
                )

            del model


def evaluate_multiscale_on_entries(
    exp: Experiment,
    model: MsY2Net,
    serial_or_terminal: str,
    test_entries: Dict[str, List[DataEntry]],
    tissue_masks: Optional[Dict[str, List[Path]]],
    pred_subdir: str,
    save_pred: bool,
):
    test_data = {
        image_name: MultiscaleDataset(NumpyDatasetSource(entries))
        for image_name, entries in test_entries.items()
    }

    Prediction(
        exp.working_dir,
        PredictionConfig(
            batch_size=18,
            save_pred=save_pred,
        ),
        exp.dry_run,
        name="prediction_" + serial_or_terminal,
    ).run(
        model,
        test_data,
        pred_mask_per_image=tissue_masks,
        pred_subdir=pred_subdir,
    )
