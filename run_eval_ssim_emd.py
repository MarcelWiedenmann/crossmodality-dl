"""
Evaluates the similarity of stain-transferred serial H&E tiles to their terminal H&E counterparts and their original
serial H&E versions in terms of:
- The structural similarity index measure (SSIM) between the (grayscale) tiles
- The earth mover's distance (EMD) between the intensity histograms of the tiles, averaged over all color channels
Note that the SSIM is a full-reference measure, so care should be taken when interpreting results between the
(transferred) serial tiles and their terminal counterparts, as these are not perfectly aligned.
"""
from argparse import ArgumentParser
from logging import info, warning
from pprint import pformat

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from skimage.color import rgb2gray, rgb2hsv
from skimage.metrics import structural_similarity
from tabulate import tabulate

import configure_logging  # Just needs to be imported.
import constants as c
import crossvalidation as cv
from data_collection import DataCollection, DataCollectionConfig
from experiment import Experiment
from staintrans.data_sampling import filter_tiles_by_tissue_fg_overlap_and_pair


def run_eval_ssim_emd(opts):
    reference_samples_dir = (
        cv.get_terminal_sample_dirs()[0]
        if opts.terminal
        else cv.get_serial_sample_dirs()[0]
    )

    cv_folds = cv.get_enumerated_folds(opts.cv_folds)
    for i, (
        *_,
        terminal_test_images,
        serial_test_images,
    ) in cv_folds:
        with Experiment(name=_get_experiment_name(opts, i)) as exp:
            metrics_save_path = _get_save_path(exp, opts)
            if metrics_save_path.exists():
                warning(f"Metrics at {metrics_save_path} already exist. Skipping.")
                _print_metrics(pd.read_csv(metrics_save_path, index_col=0))
                continue

            if opts.staintrans_name is not None:
                serial_samples_dir = cv.get_staintrans_serial_sample_dirs(
                    opts.staintrans_name, None, i
                )[0]
            elif opts.stainnorm_color_space is not None:
                serial_samples_dir = cv.get_normalized_serial_sample_dirs(
                    opts.stainnorm_color_space, opts.stainnorm_factor, i
                )[0]
            else:
                # Compute metrics for the original serial <--> terminal pairs.
                serial_samples_dir = cv.get_serial_sample_dirs()[0]
            *_, serial_test_entries = DataCollection(
                exp.working_dir,
                DataCollectionConfig(
                    sample_dirs=[serial_samples_dir],
                    test_images=serial_test_images,
                ),
                exp.dry_run,
                "eval_metrics_ssim_emd_serial_data_collection",
            ).run()

            *_, reference_test_entries = DataCollection(
                exp.working_dir,
                DataCollectionConfig(
                    sample_dirs=[reference_samples_dir],
                    test_images=terminal_test_images
                    if opts.terminal
                    else serial_test_images,
                ),
                exp.dry_run,
                "eval_metrics_ssim_emd_reference_data_collection",
            ).run()

            tissue_fg_overlap_paths = {
                serial_image_name: c.scratch_dir
                / cv.get_serial_sample_dirs()[0]
                / serial_image_name
                / "serial_terminal_tissue_fg_overlaps.lut"
                for serial_image_name in serial_test_images
            }
            all_entries = filter_tiles_by_tissue_fg_overlap_and_pair(
                serial_test_entries,
                reference_test_entries,
                c.serial2terminal if opts.terminal else None,
                pair_filter_tissue_fg_overlap=0.8,
                tissue_fg_overlap_paths=tissue_fg_overlap_paths,
            )

            info("Computing metrics...")
            metrics = []
            fold_ssim_sum = 0.0
            fold_emd_sum = 0.0
            fold_num_pairs = 0
            for serial_image_name, image_entries in all_entries.items():
                image_ssim_sum = 0.0
                image_emd_sum = 0.0
                image_num_pairs = len(image_entries)
                for _, entry in image_entries:
                    serial_tile = np.load(entry.samples[0])
                    reference_tile = np.load(entry.targets[0])
                    image_ssim_sum += _compute_ssim(serial_tile, reference_tile)
                    image_emd_sum += _compute_emd(serial_tile, reference_tile)
                image_ssim = image_ssim_sum / image_num_pairs
                image_emd = image_emd_sum / image_num_pairs
                info(
                    (
                        f"Average metrics over the {image_num_pairs} pairs of image {serial_image_name}: "
                        f"SSIM={image_ssim}, EMD={image_emd}."
                    )
                )
                metrics.append(
                    {
                        "Image": serial_image_name,
                        "Number of pairs": image_num_pairs,
                        "SSIM": image_ssim,
                        "EMD": image_emd,
                    }
                )
                fold_ssim_sum += image_ssim_sum
                fold_emd_sum += image_emd_sum
                fold_num_pairs += image_num_pairs
            fold_ssim = fold_ssim_sum / fold_num_pairs
            fold_emd = fold_emd_sum / fold_num_pairs
            info(
                f"Average metrics over the {fold_num_pairs} pairs of fold #{i}: SSIM={fold_ssim}, EMD={fold_emd}."
            )
            metrics.append(
                {
                    "Image": "all",
                    "Number of pairs": fold_num_pairs,
                    "SSIM": fold_ssim,
                    "EMD": fold_emd,
                }
            )
            metrics = pd.DataFrame(metrics)
            metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(metrics_save_path)
            _print_metrics(metrics)


def _compute_ssim(serial, reference):
    """
    SSIM parameterization is equivalent to the reference implementation from Wang et al. (2004).
    """
    serial_grayscale = rgb2gray(np.moveaxis(serial, 0, -1))
    reference_grayscale = rgb2gray(np.moveaxis(reference, 0, -1))
    return structural_similarity(
        serial_grayscale,
        reference_grayscale,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )


_hist_bins = np.arange(256)


def _compute_emd(serial, reference):
    serial_hsv = np.moveaxis(rgb2hsv(np.moveaxis(serial, 0, -1)), -1, 0)
    reference_hsv = np.moveaxis(rgb2hsv(np.moveaxis(reference, 0, -1)), -1, 0)
    hist_dist = 0.0
    for channel in range(3):
        # Computing histograms at what corresponds to byte precision should be sufficient (and most intuitive).
        serial_hist, _ = np.histogram(serial_hsv[channel], bins=256, range=(0, 1))
        reference_hist, _ = np.histogram(reference_hsv[channel], bins=256, range=(0, 1))
        hist_dist += wasserstein_distance(
            _hist_bins,
            _hist_bins,
            u_weights=serial_hist,
            v_weights=reference_hist,
        )
    hist_dist /= 3.0
    return hist_dist


def print_eval_ssim_emd_summary(opts):
    all_metrics = pd.DataFrame()
    for i in range(len(cv.get_folds())):
        with Experiment(name=_get_experiment_name(opts, i)) as exp:
            metrics_save_path = _get_save_path(exp, opts)
            metrics = pd.read_csv(metrics_save_path, index_col=0)
            metrics = metrics[metrics.Image == "all"]
            all_metrics = pd.concat([all_metrics, metrics])
    num_pairs = np.sum(all_metrics["Number of pairs"])
    ssim_mean = np.average(all_metrics.SSIM, weights=all_metrics["Number of pairs"])
    ssim_std = np.sqrt(
        np.average(
            (all_metrics.SSIM - ssim_mean) ** 2, weights=all_metrics["Number of pairs"]
        )
    )
    emd_mean = np.average(all_metrics.EMD, weights=all_metrics["Number of pairs"])
    emd_std = np.sqrt(
        np.average(
            (all_metrics.EMD - emd_mean) ** 2, weights=all_metrics["Number of pairs"]
        )
    )
    summary = pd.DataFrame(
        {
            "Number of pairs": [num_pairs],
            "SSIM (mean)": [ssim_mean],
            "SSIM (std)": [ssim_std],
            "EMD (mean)": [emd_mean],
            "EMD (std)": [emd_std],
        }
    )
    _print_metrics(summary)


def _get_experiment_name(opts, cv_fold):
    if opts.staintrans_name is not None:
        experiment_name = f"staintrans_{opts.staintrans_name}_"
    elif opts.stainnorm_color_space is not None:
        experiment_name = (
            f"norm_{opts.stainnorm_color_space}_{opts.stainnorm_factor:.5f}_"
        )
    else:
        experiment_name = "eval_metrics_ssim_emd"
    return experiment_name + f"cv_fold_{cv_fold}"


def _get_save_path(exp, opts):
    if opts.staintrans_name is not None:
        pred_subdir = "serial/checkpoint_best/blended"
    elif opts.stainnorm_color_space is not None:
        pred_subdir = "serial"
    else:
        pred_subdir = ""
    serial_reference_suffix = "" if opts.terminal else "_serial_reference"
    return (
        exp.working_dir
        / "predictions"
        / pred_subdir
        / f"metrics_ssim_emd{serial_reference_suffix}.csv"
    )


def _print_metrics(metrics):
    info(
        "Metrics:\n"
        + tabulate(metrics, headers="keys", tablefmt="psql", floatfmt=".3f")
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--stainnorm_color_space", type=str)
    parser.add_argument("--stainnorm_factor", type=float)
    parser.add_argument("--staintrans_name", type=str)
    parser.add_argument("--cv_fold", action="append", type=int, dest="cv_folds")
    parser.add_argument("--only_print_summary", action="store_true")
    args = parser.parse_args()
    if args.stainnorm_color_space is None != args.stainnorm_factor is None:
        parser.error(
            "Either both or none of the normalization-related options must be specified."
        )
    if args.stainnorm_color_space is not None and args.staintrans_name is not None:
        parser.error(
            "At most one of stain normalization and stain transfer can be specified."
        )
    if args.only_print_summary and args.cv_folds is not None:
        parser.error(
            "Cannot specify cross-validation folds when just printing the summary over all folds."
        )
    info("Arguments of the run:\n" + pformat(vars(args)))

    # Compare stain-normalized and stain-transferred results to both the terminal and the serial references. Compare the
    # serial reference only to the terminal reference, not to itself.
    is_terminal_reference = [True]
    if args.stainnorm_color_space is not None or args.staintrans_name is not None:
        is_terminal_reference.append(False)

    for is_terminal in is_terminal_reference:
        args.terminal = is_terminal
        terminal_or_serial = "terminal" if is_terminal else "serial"
        if args.only_print_summary:
            info(f"Summary of run against {terminal_or_serial} reference data.")
            print_eval_ssim_emd_summary(args)
        else:
            info(f"Evaluating against {terminal_or_serial} reference data.")
            run_eval_ssim_emd(args)
