"""
Train the multi-scale segmentation model. Training can optionally be performed using stain color augmentation. The
augmentation parameters used in the conducted experiments were (stainaugm_color_space, stainaugm_factor):
    - "hed", 0.00625,
    - "hed", 0.0125,
    - "hed", 0.025,
    - "hed", 0.05,
    - "hed", 0.1,
    - "hed", 0.2,
    - "hed", 0.3.
"""
import configure_logging  # Just needs to be imported.
import constants as c
import crossvalidation as cv
from data_collection import DataCollection, DataCollectionConfig
from data_preparation import DataPreparation, DataPreparationConfig
from data_sampling import StratifyingDataSampling
from evaluate_multiscale import evaluate_multiscale_on_entries
from experiment import Experiment
from multiscale import get_experiment_name
from multiscale.model import MsY2Net
from training import Training, TrainingConfig


def train_multiscale(opts):
    terminal_sample_dirs = cv.get_terminal_sample_dirs()
    terminal_targets_dir = cv.get_terminal_targets_dir()

    serial_sample_dirs = cv.get_serial_sample_dirs()
    serial_targets_dir = cv.get_serial_targets_dir()

    cv_folds = cv.get_enumerated_folds(opts.cv_folds)
    for i, (
        train_images,
        val_images,
        terminal_test_images,
        serial_test_images,
    ) in cv_folds:
        experiment_name = get_experiment_name(opts, i)
        with Experiment(name=experiment_name, seed=c.seed) as exp:
            train_entries, val_entries, terminal_test_entries = DataCollection(
                exp.working_dir,
                DataCollectionConfig(
                    sample_dirs=terminal_sample_dirs,
                    targets_dir=terminal_targets_dir,
                    train_images=train_images,
                    val_images=val_images,
                    test_images=terminal_test_images,
                ),
                exp.dry_run,
                name="terminal_data_collection",
            ).run()

            # Train model on terminal H&E:

            train_strata, val_strata = StratifyingDataSampling(
                exp.working_dir,
                exp.dry_run,
            ).run(train_entries, val_entries)

            train_data, val_data = DataPreparation(
                exp.working_dir,
                DataPreparationConfig(
                    color_augment=opts.stainaugm_color_space,
                    color_augment_hed_sigma=opts.stainaugm_factor
                    if opts.stainaugm_color_space == "hed"
                    else None,
                    color_augment_hed_extract_tissue_fg_fn="datasets.labeled.extract_tissue_fg",
                    color_augment_hsv_offset=opts.stainaugm_factor
                    if opts.stainaugm_color_space == "hsv"
                    else None,
                    resampling_seed=1914177978,
                ),
                exp.dry_run,
            ).run(train_strata, val_strata)

            best_model = Training(
                exp.working_dir,
                TrainingConfig(
                    train_batch_size=18,
                    val_batch_size=18,
                    epochs=120,
                    early_stopping=True,
                    early_stopping_patience=30,
                ),
                exp.dry_run,
            ).run(MsY2Net(), train_data, val_data)

            # Evaluate model on terminal H&E:

            evaluate_multiscale_on_entries(
                exp,
                best_model,
                terminal_test_entries,
                tissue_masks=None,
                pred_subdir="terminal",
                save_pred=opts.save_pred,
            )

            # Evaluate model on serial H&E:

            serial_test_targets = [
                c.serial2terminal[serial] for serial in serial_test_images
            ]
            *_, serial_test_entries = DataCollection(
                exp.working_dir,
                DataCollectionConfig(
                    sample_dirs=serial_sample_dirs,
                    targets_dir=serial_targets_dir,
                    test_images=serial_test_images,
                    test_targets=serial_test_targets,
                ),
                exp.dry_run,
                name="serial_test_data_collection",
            ).run()
            evaluate_multiscale_on_entries(
                exp,
                best_model,
                serial_test_entries,
                tissue_masks=None,
                pred_subdir="serial",
                save_pred=opts.save_pred,
            )

            del best_model
