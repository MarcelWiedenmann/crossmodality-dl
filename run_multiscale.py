"""
Runs the training and evaluation of the multi-scale segmentation model.
"""
from argparse import ArgumentParser
from logging import info
from pprint import pformat

import configure_logging  # Just needs to be imported.
from evaluate_multiscale import evaluate_multiscale
from train_multiscale import train_multiscale


def run_multiscale(opts):
    if opts.func is train_multiscale:
        if opts.stainaugm_color_space is None != opts.stainaugm_factor is None:
            parser.error(
                "Either both or none of the augmentation-related options must be specified."
            )
    if opts.func is evaluate_multiscale:
        if not opts.terminal and not opts.serial:
            parser.error("At least one of terminal and serial data must be specified.")
        if opts.stainaugm_color_space is None != opts.stainaugm_factor is None:
            parser.error(
                "Either both or none of the augmentation-related options must be specified."
            )
        if opts.stainnorm_color_space is None != opts.stainnorm_factor is None:
            parser.error(
                "Either both or none of the normalization-related options must be specified."
            )
        if opts.stainnorm_color_space is not None and opts.staintrans_name is not None:
            parser.error(
                "At most one of stain normalization and stain transfer can be specified."
            )

    info("Arguments of multi-scale segmentation run:\n" + pformat(vars(opts)))

    opts.func(opts)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cv_fold", action="append", type=int, dest="cv_folds")
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--stainaugm_color_space", type=str)
    train_parser.add_argument("--stainaugm_factor", type=float)
    train_parser.add_argument(
        "--save_predictions", action="store_true", dest="save_pred"
    )
    train_parser.set_defaults(func=train_multiscale)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--terminal", action="store_true")
    eval_parser.add_argument("--serial", action="store_true")
    eval_parser.add_argument("--stainaugm_color_space", type=str)
    eval_parser.add_argument("--stainaugm_factor", type=float)
    eval_parser.add_argument("--stainnorm_color_space", type=str)
    eval_parser.add_argument("--stainnorm_factor", type=float)
    eval_parser.add_argument("--staintrans_name", type=str)
    eval_parser.add_argument("--staintrans_checkpoint_epoch", type=int)
    eval_parser.add_argument(
        "--save_predictions", action="store_true", dest="save_pred"
    )
    eval_parser.set_defaults(func=evaluate_multiscale)

    run_multiscale(parser.parse_args())
