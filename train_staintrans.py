from logging import info

import constants as c
import crossvalidation as cv
from data import DataEntry, NumpyDatasetSource
from data_collection import DataCollection, DataCollectionConfig
from data_sampling import FilteringDataSampling, FilteringDataSamplingConfig
from experiment import Experiment
from singlescale.bulten2019 import ResUNet
from staintrans.data import StainTransferValidationDataset
from staintrans.data_preparation import DataPreparation, DataPreparationConfig
from staintrans.data_sampling import DataSampling, DataSamplingConfig
from staintrans.model import CycleGan
from training import Training, TrainingConfig


def train_staintrans(opts):
    terminal_samples_dir = cv.get_terminal_sample_dirs(
        shape=(opts.tile_size, opts.tile_size)
    )[0]
    serial_samples_dir = cv.get_serial_sample_dirs(
        shape=(opts.tile_size, opts.tile_size)
    )[0]
    # For task-specific validation.
    terminal_segm_targets_dir = cv.get_terminal_targets_dir()

    cv_folds = cv.get_enumerated_folds(opts.cv_folds)
    for i, (
        terminal_train_images,
        terminal_val_images,
        *_,
    ) in cv_folds:
        experiment_name = f"staintrans_{opts.name}_cv_fold_{i}"
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

            # Pair and sample serial and terminal training tiles:

            train_entries = DataSampling(
                exp.working_dir,
                DataSamplingConfig(
                    num_train_samples=opts.num_train_samples,
                ),
                exp.dry_run,
                name="train_data_sampling",
            ).run(serial_train_entries, terminal_train_entries, c.serial2terminal)

            train_data = DataPreparation(
                exp.working_dir,
                DataPreparationConfig(
                    rescale_data=opts.rescale_data,
                    include_segmentation_targets="saliency" in opts.losses,
                    resampling_seed=1914177978,
                ),
                exp.dry_run,
                name="train_data_preparation",
            ).run(train_entries)

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
                NumpyDatasetSource(val_entries), opts.rescale_data
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

            model = CycleGan(
                gen_lr=opts.lr_gen,
                dis_lr=opts.lr_dis,
                target_identity_loss="target" in opts.losses,
                source_identity_loss="source" in opts.losses,
                saliency_loss="saliency" in opts.losses,
                bottleneck_loss="bottleneck" in opts.losses,
                feature_loss="feature" in opts.losses,
                cycle_lambda=opts.cycle_lambda,
                bottleneck_lambda=opts.bottleneck_lambda,
                feature_lambda=opts.feature_lambda,
                padding_mode="reflect",
                generator=opts.generator,
                segmentation_model=segm_model,
            )

            Training(
                exp.working_dir,
                TrainingConfig(
                    train_batch_size=opts.batch_size,
                    val_batch_size=opts.batch_size,
                    epochs=200,
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
            ).run(model, train_data, val_data)
