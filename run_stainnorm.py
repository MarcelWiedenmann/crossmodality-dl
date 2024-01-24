"""
Runs the training and application of a particularly parameterized version of the stain color normalization model. The
parameters used in the conducted experiments were:
    - "hed", 0.00625,
    - "hed", 0.0125,
    - "hed", 0.025,
    - "hsv", 1.0.
"""
from argparse import ArgumentParser
from logging import info
from pprint import pformat
from typing import Dict, List

import configure_logging  # Just needs to be imported.
import constants as c
import crossvalidation as cv
from data import DataEntry, MultiscaleDataset, NumpyDatasetSource
from data_collection import DataCollection, DataCollectionConfig
from data_preparation import DataPreparation, DataPreparationConfig
from data_sampling import DataEntryGroup, StratifyingDataSampling
from experiment import Experiment
from prediction import PredictionConfig
from stainnorm.data import StainNormalizationDataset
from stainnorm.model import StainNormalizationNet
from stainnorm.prediction import StainNormalization
from training import Training, TrainingConfig


def train_stainnorm(opts):
    color_space = opts.color_space
    factor = opts.factor

    terminal_sample_dirs = cv.get_terminal_sample_dirs()
    terminal_targets_dir = cv.get_terminal_targets_dir()

    cv_folds = cv.get_enumerated_folds(opts.cv_folds)
    for i, (
        train_images,
        val_images,
        *_,
    ) in cv_folds:
        experiment_name = f"stainnorm_{color_space}_{factor:.5f}_cv_fold_{i}"
        with Experiment(name=experiment_name, seed=c.seed, dry_run=opts.dry_run) as exp:
            train_entries, val_entries, _ = DataCollection(
                exp.working_dir,
                DataCollectionConfig(
                    sample_dirs=terminal_sample_dirs,
                    targets_dir=terminal_targets_dir,
                    train_images=train_images,
                    val_images=val_images,
                ),
                exp.dry_run,
                name="terminal_data_collection",
            ).run()

            train_strata, val_strata = StratifyingDataSampling(
                exp.working_dir,
                exp.dry_run,
            ).run(train_entries, val_entries)

            # Flatten multi-scale entries into multiple single-scale entries.
            train_strata = _flatten_multiscale_entries(train_strata)
            val_strata = _flatten_multiscale_entries(val_strata)

            train_data, val_data = DataPreparation(
                exp.working_dir,
                DataPreparationConfig(
                    basic_augment=False,
                    color_augment=color_space,
                    color_augment_hed_sigma=factor if color_space == "hed" else None,
                    color_augment_hed_extract_tissue_fg_fn="datasets.labeled.extract_tissue_fg",
                    color_augment_hsv_offset=factor if color_space == "hsv" else None,
                ),
                exp.dry_run,
            ).run(train_strata, val_strata, create_dataset_fn=StainNormalizationDataset)

            Training(
                exp.working_dir,
                TrainingConfig(
                    train_batch_size=64,
                    val_batch_size=64,
                    epochs=120,
                    checkpoints_file_name_metrics=["val_loss"],
                    checkpoints_monitored_metric=("val_loss", "min"),
                    early_stopping=True,
                    early_stopping_patience=10,
                ),
                exp.dry_run,
            ).run(StainNormalizationNet(), train_data, val_data)


def _flatten_multiscale_entries(strata):
    return [
        DataEntryGroup(
            [
                DataEntry((s,), ())
                for entry in strata[0].all_entries
                for s in entry.samples
            ],
            None,
        )
    ]


def apply_stainnorm(opts):
    color_space = opts.color_space
    factor = opts.factor

    terminal_sample_dirs = cv.get_terminal_sample_dirs()
    serial_sample_dirs = cv.get_serial_sample_dirs()

    cv_folds = cv.get_enumerated_folds(opts.cv_folds)
    for i, (
        *_,
        terminal_test_images,
        serial_test_images,
    ) in cv_folds:
        experiment_name = f"stainnorm_{color_space}_{factor:.5f}_cv_fold_{i}"
        with Experiment(name=experiment_name, seed=c.seed, dry_run=opts.dry_run) as exp:
            checkpoint_path = exp.working_dir / "checkpoints" / "best.ckpt"
            info(f"Loading stainnorm model from checkpoint path: {checkpoint_path}.")
            model = StainNormalizationNet.load_from_checkpoint(checkpoint_path)

            if opts.terminal:
                _, _, terminal_test_entries = DataCollection(
                    exp.working_dir,
                    DataCollectionConfig(
                        sample_dirs=terminal_sample_dirs,
                        test_images=terminal_test_images,
                    ),
                    exp.dry_run,
                    name="terminal_data_collection",
                ).run()

                _apply_stainnorm_to_entries(
                    exp,
                    model,
                    "terminal",
                    terminal_test_entries,
                )

            if opts.serial:
                _, _, serial_test_entries = DataCollection(
                    exp.working_dir,
                    DataCollectionConfig(
                        sample_dirs=serial_sample_dirs,
                        test_images=serial_test_images,
                    ),
                    exp.dry_run,
                    name="serial_data_collection",
                ).run()

                _apply_stainnorm_to_entries(
                    exp,
                    model,
                    "serial",
                    serial_test_entries,
                )


def _apply_stainnorm_to_entries(
    exp: Experiment,
    model: StainNormalizationNet,
    serial_or_terminal: str,
    test_entries: Dict[str, List[DataEntry]],
):
    # Normalization must be applied to the different scale levels separately. Group by level and then by image.
    level_names = ["level_0", "level_2", "level_3"]
    test_entries_per_level = [{}, {}, {}]
    for image_name, entries in test_entries.items():
        for test_data in test_entries_per_level:
            test_data[image_name] = []
        for entry in entries:
            for j, s in enumerate(entry.samples):
                test_entries_per_level[j][image_name].append(DataEntry((s,), ()))

    stainnorm = StainNormalization(
        exp.working_dir,
        PredictionConfig(
            batch_size=64,
            num_workers=0,  # Work around multi-processing memory leak.
        ),
        exp.dry_run,
        name="prediction_" + serial_or_terminal,
    )
    for level_name, test_entries in zip(level_names, test_entries_per_level):
        info(f"Normalizing level {level_name}.")
        test_data = {
            image_name: MultiscaleDataset(NumpyDatasetSource(entries))
            for image_name, entries in test_entries.items()
        }
        stainnorm.run(
            model, test_data, pred_subdir=serial_or_terminal + "/" + level_name
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("color_space", type=str)
    parser.add_argument("factor", type=float)
    parser.add_argument("--cv_fold", action="append", type=int, dest="cv_folds")
    parser.add_argument("--dry_run", action="store_true")
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train_stainnorm)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--terminal", action="store_true")
    apply_parser.add_argument("--serial", action="store_true")
    apply_parser.set_defaults(func=apply_stainnorm)

    args = parser.parse_args()
    if args.func is apply_stainnorm:
        if not args.terminal and not args.serial:
            parser.error("At least one of terminal and serial data must be specified.")
    info("Arguments of stainnorm run:\n" + pformat(vars(args)))

    args.func(args)
