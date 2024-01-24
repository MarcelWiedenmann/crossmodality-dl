"""
Train the stain transfer baseline from Shaban et al. (2019).

Training can be done with either unpaired or weakly paired data. The original paper performs training with entirely
unpaired data. Our data set is however paired at the tile level ("weakly paired"), so we can optionally make use of that
and measure if it brings any benefits.
"""
from argparse import ArgumentParser
from logging import info
from pprint import pformat

import configure_logging  # Just needs to be imported.
import constants as c
import crossvalidation as cv
from data import DataEntry, NumpyDatasetSource
from data_collection import DataCollection, DataCollectionConfig
from data_sampling import FilteringDataSampling, FilteringDataSamplingConfig
from experiment import Experiment
from singlescale.bulten2019 import ResUNet
from staintrans import shaban2019
from staintrans.data import (
    StainTransferValidationDataset,
    UnpairedStainTransferTrainingDataset,
)
from staintrans.data_preparation import DataPreparation, DataPreparationConfig
from staintrans.data_sampling import DataSampling, DataSamplingConfig
from training import Training, TrainingConfig


def train(opts):
    terminal_samples_dir = cv.get_terminal_sample_dirs(
        shape=(opts.tile_size, opts.tile_size)
    )[0]
    serial_samples_dir = cv.get_serial_sample_dirs(
        shape=(opts.tile_size, opts.tile_size)
    )[0]
    # For task-specific validation.
    terminal_segm_targets_dir = cv.get_terminal_targets_dir(
        shape=(opts.tile_size, opts.tile_size)
    )

    cv_folds = cv.get_enumerated_folds(opts.cv_folds)
    for i, (terminal_train_images, terminal_val_images, *_) in cv_folds:
        experiment_name = "staintrans_shaban2019_"
        if opts.name_suffix is not None:
            experiment_name += opts.name_suffix + "_"
        experiment_name += f"cv_fold_{i}"
        with Experiment(name=experiment_name, seed=c.seed, dry_run=opts.dry_run) as exp:
            serial_train_images = [c.terminal2serial[t] for t in terminal_train_images]
            serial_val_images = [c.terminal2serial[t] for t in terminal_val_images]

            # Collect serial tiles (source domain):

            serial_train_entries, serial_val_entries, _ = DataCollection(
                exp.working_dir,
                DataCollectionConfig(
                    sample_dirs=[serial_samples_dir],
                    train_images=serial_train_images,
                    val_images=serial_val_images,
                ),
                exp.dry_run,
                name="serial_data_collection",
            ).run()

            # Collect terminal tiles (target domain):

            terminal_train_entries, terminal_val_entries, _ = DataCollection(
                exp.working_dir,
                DataCollectionConfig(
                    sample_dirs=[terminal_samples_dir],
                    targets_dir=terminal_segm_targets_dir,
                    train_images=terminal_train_images,
                    val_images=terminal_val_images,
                ),
                exp.dry_run,
                name="terminal_data_collection",
            ).run()

            # Prepare serial and terminal training tiles, either weakly paired or unpaired:

            if opts.weakly_paired:
                train_entries = DataSampling(
                    exp.working_dir,
                    DataSamplingConfig(
                        num_train_samples=opts.weakly_paired_num_train_samples,
                    ),
                    exp.dry_run,
                    name="train_data_sampling",
                ).run(serial_train_entries, terminal_train_entries, c.serial2terminal)

                train_data = DataPreparation(
                    exp.working_dir,
                    DataPreparationConfig(
                        rescale_data=True,
                        include_segmentation_targets=False,
                        resampling_seed=1914177978,
                    ),
                    exp.dry_run,
                    name="train_data_preparation",
                ).run(train_entries)
            else:
                # We do not pair the training tiles but at least demand a minimum amount of tissue to be present in each
                # tile, otherwise we would teach the model to e.g. map tissue to glass slide and vice versa, which is
                # not really desirable.

                _, serial_train_entries = FilteringDataSampling(
                    exp.working_dir,
                    FilteringDataSamplingConfig(
                        tile_filter_tissue_fg_ratio=0.8,
                    ),
                    exp.dry_run,
                    name="serial_train_data_sampling",
                ).run({}, serial_train_entries)
                serial_train_entries = serial_train_entries[0].all_entries

                _, terminal_train_entries = FilteringDataSampling(
                    exp.working_dir,
                    FilteringDataSamplingConfig(
                        tile_filter_tissue_fg_ratio=0.8,
                    ),
                    exp.dry_run,
                    name="terminal_train_data_sampling",
                ).run({}, terminal_train_entries)
                terminal_train_entries = terminal_train_entries[0].all_entries
                terminal_train_entries = [
                    DataEntry((e.samples[0],), ()) for e in terminal_train_entries
                ]

                info(
                    (
                        f"Collected {len(serial_train_entries)} serial and {len(terminal_train_entries)} terminal "
                        "unpaired training entries."
                    )
                )

                train_data = UnpairedStainTransferTrainingDataset(
                    NumpyDatasetSource(serial_train_entries),
                    NumpyDatasetSource(terminal_train_entries),
                    rescale_data=True,
                    resampling_seed=1914177978,
                )

            # Filter and pair serial and terminal validation tiles, as well as matching segmentation masks for task-
            # specific validation:

            _, serial_val_entries = FilteringDataSampling(
                exp.working_dir,
                FilteringDataSamplingConfig(
                    tile_filter_tissue_fg_ratio=0.8,
                ),
                exp.dry_run,
                name="serial_val_data_sampling",
            ).run({}, serial_val_entries)
            serial_val_entries = serial_val_entries[0].all_entries

            _, terminal_val_entries = FilteringDataSampling(
                exp.working_dir,
                FilteringDataSamplingConfig(
                    tile_filter_tissue_fg_ratio=0.8,
                ),
                exp.dry_run,
                name="terminal_val_data_sampling",
            ).run({}, terminal_val_entries)
            terminal_val_entries = terminal_val_entries[0].all_entries

            val_entries = []
            serial_val_samples = set(s.samples[0] for s in serial_val_entries)
            for terminal_entry in terminal_val_entries:
                terminal_sample = terminal_entry.samples[0]
                terminal_segm_target = terminal_entry.targets[0]
                serial_image_name = c.terminal2serial[terminal_sample.parent.name]
                serial_sample = (
                    c.scratch_dir
                    / serial_samples_dir
                    / serial_image_name
                    / (
                        serial_image_name
                        + "_tile_"
                        + terminal_sample.name.split("_tile_")[1]
                    )
                )
                if serial_sample in serial_val_samples:
                    val_entries.append(
                        DataEntry(
                            (serial_sample,), (terminal_sample, terminal_segm_target)
                        )
                    )
            info(
                f"Collected {len(val_entries)} paired serial-terminal validation entries."
            )

            val_data = StainTransferValidationDataset(
                NumpyDatasetSource(val_entries), rescale_data=True
            )

            # Train model:

            segm_model_path = (
                c.experiments_dir
                / f"singlescale2_no_px_cv_fold_{i}"
                / "checkpoints"
                / "best.ckpt"
            )
            info(
                f"Segmentation performance will be validated using model {segm_model_path}."
            )
            segm_model = ResUNet.load_from_checkpoint(segm_model_path)

            model = shaban2019.CycleGan(segmentation_model=segm_model)

            training = Training(
                exp.working_dir,
                TrainingConfig(
                    train_batch_size=opts.batch_size,
                    val_batch_size=opts.batch_size,
                    # The model was trained for 26 epochs in the original paper. Let's give it some more time, to check
                    # whether additional updates are beneficial when applying it to our data set.
                    epochs=40,
                    checkpoints_file_name_metrics=["segm_jaccard_A2B_val"],
                    checkpoints_monitored_metric=("segm_jaccard_A2B_val", "max"),
                    log_learning_rate=True,
                    continue_training=opts.continue_training,
                    trainer_kwargs={
                        "track_grad_norm": 2,
                        "num_sanity_val_steps": 4,
                        # "fast_dev_run": 4,
                    },
                ),
                exp.dry_run,
            )
            training.run(model, train_data, val_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name_suffix", type=str)
    parser.add_argument("--cv_fold", action="append", type=int, dest="cv_folds")
    parser.add_argument("--dry_run", action="store_true")
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--tile_size", type=int, default=256)
    train_parser.add_argument("--unpaired", action="store_false", dest="weakly_paired")
    train_parser.add_argument(
        "--weakly_paired_num_train_samples", type=int, default=40000
    )
    # The batch size used in the original paper is 4. Saturating the available GPU memory with a batch size of 12 has
    # however produced better results in our experiments, so use this value by default.
    train_parser.add_argument("--batch_size", type=int, default=12)
    train_parser.add_argument(
        "--continue", action="store_true", dest="continue_training"
    )
    train_parser.set_defaults(func=train)

    args = parser.parse_args()
    info("Arguments of staintrans shaban2019 run:\n" + pformat(vars(args)))

    args.func(args)
