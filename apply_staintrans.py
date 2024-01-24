import re
from logging import info, warning
from pprint import pformat

import crossvalidation as cv
from data import NumpyDatasetSource
from data_collection import DataCollection, DataCollectionConfig
from experiment import Experiment
from staintrans import get_model_type
from staintrans.data import StainTransferTestDataset
from staintrans.prediction import (
    StainTransfer,
    StainTransferConfig,
    StainTransferPostprocessing,
    StainTransferPostprocessingConfig,
)
from utils import list_files


def apply_staintrans(opts):
    # Only transfer full-resolution tiles. Other levels will be obtained via downsampling if required.
    sample_dir = _get_sample_dir(opts, shape=(512, 512), overlap=(768, 768))
    sample_dir_no_overlap = _get_sample_dir(opts, shape=(512, 512), overlap=(0, 0))

    cv_folds = cv.get_enumerated_folds(opts.cv_folds)
    for i, (
        *_,
        terminal_test_images,
        serial_test_images,
    ) in cv_folds:
        experiment_name = f"staintrans_{opts.name}_cv_fold_{i}"
        with Experiment(seed=1634278956, name=experiment_name) as exp:
            test_images = (
                serial_test_images if opts.serial2terminal else terminal_test_images
            )
            if opts.apply:
                # Apply model to transform H&Es:

                data_collection = DataCollection(
                    exp.working_dir,
                    DataCollectionConfig(
                        sample_dirs=[sample_dir],  # Only apply to full-resolution data.
                        test_images=test_images,
                    ),
                    exp.dry_run,
                    name="test_data_collection",
                )
                _, _, test_entries = data_collection.run()

                test_data = {
                    image_name: StainTransferTestDataset(
                        NumpyDatasetSource(entries),
                        opts.rescale_data,
                    )
                    for image_name, entries in test_entries.items()
                }

                checkpoints_dir = exp.working_dir / "checkpoints"
                if opts.checkpoint_epoch is not None:
                    checkpoint_path = list_files(
                        checkpoints_dir,
                        file_pattern=f"epoch={opts.checkpoint_epoch:03d}_*",
                    )[0]
                else:
                    checkpoint_path = checkpoints_dir / "best.ckpt"
                    if not checkpoint_path.exists():
                        (
                            checkpoint_path,
                            opts.checkpoint_epoch,
                            checkpoint_metric,
                        ) = _find_currently_best_checkpoint(checkpoints_dir)
                        warning(
                            (
                                f'No "best" model checkpoint found. Using checkpoint of epoch {opts.checkpoint_epoch} '
                                f"instead, which had the highest recorded validation metric of {checkpoint_metric}."
                            )
                        )

                info(
                    f"Loading staintrans model from checkpoint path: {checkpoint_path}."
                )
                model_type = get_model_type(opts.model)
                model = model_type.load_from_checkpoint(checkpoint_path).to("cuda")
                info("Loaded staintrans model hparams:\n" + pformat(model.hparams))
                info(f"Loaded model's device: {model.device}.")

                serial_or_terminal = "serial" if opts.serial2terminal else "terminal"
                epoch_format_str = (
                    f"{opts.checkpoint_epoch:03}"
                    if opts.checkpoint_epoch is not None
                    else "best"
                )
                staintrans = StainTransfer(
                    exp.working_dir,
                    StainTransferConfig(
                        # NOTE: The batch size differs from the one used for training (6) because the higher-resolution
                        # tiles used here would otherwise exceed the available video memory.
                        batch_size=5,
                        direction="A2B" if opts.serial2terminal else "B2A",
                    ),
                    exp.dry_run,
                    name=f"{serial_or_terminal}_prediction_checkpoint_{epoch_format_str}",
                )
                staintrans.run(
                    model,
                    test_data,
                    pred_subdir=f"{serial_or_terminal}/checkpoint_{epoch_format_str}/",
                )

                del model

            if opts.postprocess:
                staintrans_post = StainTransferPostprocessing(
                    exp.working_dir,
                    StainTransferPostprocessingConfig(
                        rescale_data=opts.rescale_data,
                        # TODO: Switch to the function from the independent data set once the workaround there has been
                        # removed (if opts.serial2terminal).
                        extract_tissue_fg_fn="datasets.labeled.extract_tissue_fg",
                        # Multi-scale output is only required for segmentation (which is done on terminal H&E), so save
                        # the expensive downsampling when translating into the serial domain.
                        multiscale_output=opts.serial2terminal,
                    ),
                    exp.dry_run,
                )
                staintrans_post.run(
                    original_samples_dir=sample_dir_no_overlap,
                    pred_subdir=f"{serial_or_terminal}/checkpoint_{epoch_format_str}/",
                )


def _get_sample_dir(opts, shape, overlap):
    return (
        cv.get_serial_sample_dirs(shape=shape, overlap=overlap)[0]
        if opts.serial2terminal
        else cv.get_terminal_sample_dirs(shape=shape, overlap=overlap)[0]
    )


def _find_currently_best_checkpoint(checkpoints_dir):
    checkpoint_paths = list_files(checkpoints_dir, file_extension=".ckpt")
    paths_with_epochs_and_metrics = [
        ((cp,) + _extract_epoch_and_metric(cp.name)) for cp in checkpoint_paths
    ]
    return sorted(paths_with_epochs_and_metrics, key=lambda m: m[-1])[-1]


def _extract_epoch_and_metric(checkpoint_name):
    epoch, metric = re.match(
        "epoch=(.*)_segm_jaccard_A2B_val=(.*)\\.ckpt", checkpoint_name
    ).groups()
    return int(epoch), float(metric)
