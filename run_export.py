from argparse import ArgumentParser
from logging import info
from pprint import pformat
from shutil import copytree

import configure_logging  # Just needs to be imported.
import constants as c
import crossvalidation as cv
from experiment import Experiment
from exporting import ImageExport, create_upload_csv
from multiscale import get_experiment_name
from staintrans import get_pred_subdir

# TODO: Support exporting terminal images


def export(opts):
    cv_folds = cv.get_enumerated_folds(opts.cv_folds)
    for i, (
        *_,
        terminal_test_images,
        serial_test_images,
    ) in cv_folds:
        if opts.segmentation:
            experiment_name = get_experiment_name(opts, i)
        else:
            experiment_name = f"staintrans_{opts.staintrans_name}_cv_fold_{i}"
        with Experiment(name=experiment_name) as exp:
            export_dir, publish_dir = _get_export_publish_dirs(exp)
            export_file_names = []
            for terminal_image_name, serial_image_name in zip(
                terminal_test_images, serial_test_images
            ):
                if (
                    opts.image_names is not None
                    and serial_image_name not in opts.image_names
                ):
                    info(f"Skipping image {serial_image_name} as per arguments.")
                    continue
                info(f"Exporting image {serial_image_name}.")

                export_file_name = serial_image_name + f"_pred_{experiment_name}"

                image_export = ImageExport()

                custom_metadata = {
                    "OME": {
                        "Image": {
                            "@Name": serial_image_name,
                            "Pixels": {
                                "@PhysicalSizeX": 3.2499899231269475e-4,
                                "@PhysicalSizeXUnit": "mm",
                                "@PhysicalSizeY": 3.2499899231269475e-4,
                                "@PhysicalSizeYUnit": "mm",
                            },
                        }
                    }
                }
                image_export.add_metadata(custom_metadata)

                rgb_dir = c.scratch_dir / _get_rgb_data_dir(opts, i) / serial_image_name
                image_export.add_rgb(rgb_dir)

                if opts.segmentation:
                    gt_dir = (
                        c.scratch_dir
                        / cv.get_serial_targets_dir()
                        / terminal_image_name
                    )
                    image_export.add_channel("Ground_truth", gt_dir)

                    pred_subdir = get_pred_subdir(opts)
                    pred_dir = (
                        exp.working_dir
                        / "predictions"
                        / "serial"
                        / pred_subdir
                        / "shape_512_512_overlap_0_0"
                        / serial_image_name
                    )
                    image_export.add_prediction(pred_dir)

                    export_file_name += pred_subdir

                export_file_name += ".ome.tiff"
                export_file_names.append(export_file_name)

                image_export.export(export_dir / export_file_name)

            upload_csv_path = create_upload_csv(
                export_dir, publish_dir.name, export_file_names
            )
            info(f"Publishing exported files from {export_dir} to {publish_dir}.")
            copytree(export_dir, publish_dir)
            upload_csv_url = f"{c.remote_fs}{upload_csv_path}"
            info(f"CSV for upload to OMERO can be downloaded from {upload_csv_url}.")


def _get_export_publish_dirs(exp):
    # HACK: Re-using the same export directory for all image exports of the same experiment would be preferable, but the
    # option to specify individual images kind of clashes with that since it implies the creation of multiple upload
    # CSVs per experiment. For now, work around this by using unique directories per export.
    suffix = ""
    counter = 1
    while (exp.working_dir / f"exports{suffix}").exists():
        suffix = f"_{counter}"
        counter += 1
    export_dir = exp.working_dir / f"exports{suffix}"
    export_dir.mkdir(parents=True, exist_ok=False)
    publish_dir = c.publish_dir / f"predictions_{exp.name}{suffix}"
    return export_dir, publish_dir


def _get_rgb_data_dir(opts, cv_fold):
    if opts.staintrans_name is not None:
        checkpoint_epoch = (
            opts.staintrans_checkpoint_epoch
            if opts.staintrans_checkpoint_epoch is not None
            else None
        )
        return cv.get_staintrans_serial_sample_dirs(
            opts.staintrans_name, checkpoint_epoch, cv_fold
        )[0]
    else:
        return cv.get_serial_sample_dirs(
            shape=(opts.tile_size, opts.tile_size),
            overlap=(opts.overlap_size, opts.overlap_size),
        )[0]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cv_folds", type=int, nargs="*")
    parser.add_argument("--image_names", type=str, nargs="*")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap_size", type=int, default=0)
    parser.add_argument("--staintrans_name", type=str)
    parser.add_argument("--staintrans_checkpoint_epoch", type=int)
    parser.add_argument("--segmentation", action="store_true")
    parser.add_argument("--stainaugm_color_space", type=str)
    parser.add_argument("--stainaugm_factor", type=float)

    args = parser.parse_args()
    if args.image_names is not None and (
        args.cv_folds is None or len(args.cv_folds) > 1
    ):
        parser.error(
            "Individual image names can only be specified for a single cross-validation fold."
        )
    if args.staintrans_name is None and not args.segmentation:
        parser.error(
            "At least one of stain transfer results or segmentation results must be specified."
        )
    if args.segmentation is None and args.stainaugm_color_space is not None:
        parser.error("Augmentation can only be specified along with segmentation.")
    if args.stainaugm_color_space is None != args.stainaugm_factor is None:
        parser.error(
            "Either both or none of the augmentation-related options must be specified."
        )
    info("Arguments of export run:\n" + pformat(vars(args)))

    # TODO: Support source images derived from stain normalization
    args.stainnorm_color_space = None
    args.stainnorm_factor = None

    export(args)
