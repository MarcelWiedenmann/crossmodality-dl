from argparse import ArgumentParser
from logging import info, warning
from pprint import pformat

import pandas as pd
from tabulate import tabulate

import configure_logging  # Just needs to be imported.
import constants as c
import utils


def run_print_eval_metrics(opts):
    """
    +--------------+-----+-----+-----+-----+-----+---------+---------------+---------------+
    | Experiment   | 1   | 2   | 3   | 4   | 5   | Total   | Sensitivity   | Specificity   |
    |--------------+-----+-----+-----+-----+-----+---------+---------------+---------------|
    | ...          |     |     |     |     |     |         |               |               |
    +--------------+-----+-----+-----+-----+-----+---------+---------------+---------------+
    """
    all_results = pd.DataFrame(
        columns=[str(i) for i in range(1, opts.num_folds + 1)]
        + ["Total", "Sensitivity", "Specificity"]
    )
    for exp in opts.experiments:
        folds = utils.list_files(c.experiments_dir, file_pattern=exp + "_cv_fold_*")
        folds_jaccards = {}
        exp_confusion_matrix = {
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
        for fold in folds:
            fold_idx = int(fold.name.rsplit("_cv_fold_")[1])
            metrics_path = fold / "predictions" / opts.dataset
            if (metrics_path / "metrics.csv").exists():
                metrics_path = metrics_path / "metrics.csv"
            elif (metrics_path / "best").exists():
                metrics_path = metrics_path / "best" / "metrics.csv"
                if not metrics_path.exists():
                    warning('"Best" results directory found but no metrics. Evaluation in progress? Skipping fold.')
                    continue
            else:
                metrics_path, metrics_epoch = _find_latest_results_dir(metrics_path)
                if metrics_path is not None:
                    metrics_path = metrics_path / "metrics.csv"
                    warning(
                        f'No "best" metrics found. Printing latest metrics of epoch {metrics_epoch} instead.'
                    )
                else:
                    warning(f"No results available in {fold}. Skipping fold.")
                    continue
            metrics = pd.read_csv(metrics_path, index_col=0)
            metrics = metrics[metrics.Image == "all"]
            metrics = metrics.loc[:, ("TN", "FP", "FN", "TP", "Jaccard")]
            tn, fp, fn, tp, jaccard = (
                metrics.TN.squeeze(),
                metrics.FP.squeeze(),
                metrics.FN.squeeze(),
                metrics.TP.squeeze(),
                metrics.Jaccard.squeeze(),
            )
            folds_jaccards[fold_idx] = jaccard
            exp_confusion_matrix["tn"] += tn
            exp_confusion_matrix["fp"] += fp
            exp_confusion_matrix["fn"] += fn
            exp_confusion_matrix["tp"] += tp
        if len(folds_jaccards) == 0:
            warning(f"No results available for {exp}. Skipping experiment.")
            continue
        folds_jaccards = [folds_jaccards.get(i, "-") for i in range(opts.num_folds)]
        exp_metrics = utils.compute_metrics(
            exp_confusion_matrix["tn"],
            exp_confusion_matrix["fp"],
            exp_confusion_matrix["fn"],
            exp_confusion_matrix["tp"],
        )
        all_results = all_results.append(
            pd.Series(
                folds_jaccards
                + [
                    exp_metrics["Jaccard"],
                    exp_metrics["Sensitivity"],
                    exp_metrics["Specificity"],
                ],
                index=[str(i) for i in range(1, opts.num_folds + 1)]
                + ["Total", "Sensitivity", "Specificity"],
                name=exp,
            ),
        )
    print(
        tabulate(
            all_results.rename_axis("Experiment"),
            headers="keys",
            tablefmt="psql",
            floatfmt=".3f",
        )
    )


def _find_latest_results_dir(dataset_dir):
    result_dirs = utils.list_files(dataset_dir, file_pattern="*/")
    result_dirs_and_epochs = [(rd, int(rd.name)) for rd in result_dirs]
    return sorted(result_dirs_and_epochs, key=lambda rd: rd[-1])[-1]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiments", type=str, nargs="+")
    parser.add_argument("dataset", type=str, choices=["serial", "terminal"])

    args = parser.parse_args()
    args.num_folds = 5
    info("Arguments of the run:\n" + pformat(vars(args)))

    run_print_eval_metrics(args)
