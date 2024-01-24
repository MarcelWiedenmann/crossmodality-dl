"""
Runs the training of the developed stain transfer model and the application of the model as well as of the baseline
stain transfer models.
"""
from argparse import ArgumentParser
from logging import info
from pprint import pformat

import configure_logging  # Just needs to be imported.
from apply_staintrans import apply_staintrans
from evaluate_multiscale import evaluate_multiscale
from run_multiscale import run_multiscale
from train_staintrans import train_staintrans


def run_staintrans(opts):
    if opts.func is train_staintrans:
        if opts.lr is not None == (opts.lr_gen is not None or opts.lr_dis is not None):
            parser.error(
                "Either general learning rate or role-specific learning rates must be specified."
            )
        elif opts.lr_gen is None != opts.lr_dis is None:
            parser.error(
                "Either both or none of the role-specific learning rates must be specified."
            )
        elif opts.lr is not None:
            opts.lr_gen = opts.lr
            opts.lr_dis = opts.lr
    elif opts.func is apply_staintrans:
        if not opts.apply and not opts.postprocess:
            parser.error("At least one of apply and postprocess must be specified.")

    info("Arguments of staintrans run:\n" + pformat(vars(opts)))

    opts.func(opts)


def _run_evaluate_staintrans(opts):
    if not opts.serial2terminal:
        parser.error("Can only evaluate direction from serial to terminal.")

    opts.func = apply_staintrans
    run_staintrans(opts)

    opts.terminal = False
    opts.serial = True
    opts.stainnorm_color_space = None
    opts.stainnorm_factor = None
    opts.staintrans_name = opts.name
    opts.staintrans_checkpoint_epoch = opts.checkpoint_epoch
    opts.func = evaluate_multiscale
    del opts.name
    del opts.model
    del opts.checkpoint_epoch
    del opts.rescale_data
    del opts.serial2terminal
    del opts.apply
    del opts.postprocess
    run_multiscale(opts)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--cv_fold", action="append", type=int, dest="cv_folds")
    parser.add_argument("--dry_run", action="store_true")
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--tile_size", type=int, default=512)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument(
        "--no_rescale_data", action="store_false", dest="rescale_data"
    )
    train_parser.add_argument("--num_train_samples", type=int, default=10000)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--lr_gen", type=float, default=None)
    train_parser.add_argument("--lr_dis", type=float, default=None)
    train_parser.add_argument(
        "--losses",
        type=str,
        choices=["target", "source", "saliency", "bottleneck", "feature"],
        nargs="*",
        default=["bottleneck", "feature"],
    )
    train_parser.add_argument("--cycle_lambda", type=float, default=1)
    train_parser.add_argument("--bottleneck_lambda", type=float, default=0.5)
    train_parser.add_argument("--feature_lambda", type=float, default=0.5)
    train_parser.add_argument(
        "--generator",
        type=str,
        choices=["debel", "vanilla"],
        default="debel",
    )
    train_parser.add_argument(
        "--continue", action="store_true", dest="continue_training"
    )
    train_parser.set_defaults(func=train_staintrans)

    apply_parser = subparsers.add_parser("apply")
    # Which model to select, own vs baselines. Corresponds to their module names in the "staintrans" package.
    apply_parser.add_argument(
        "--model", choices=["model", "shaban2019", "debel2021"], default="model"
    )
    apply_parser.add_argument("--checkpoint_epoch", type=int)
    apply_parser.add_argument(
        "--no_rescale_data", action="store_false", dest="rescale_data"
    )
    apply_parser.add_argument(
        "--terminal2serial", action="store_false", dest="serial2terminal"
    )
    apply_parser.add_argument("--apply", action="store_true")
    apply_parser.add_argument("--postprocess", action="store_true")
    apply_parser.set_defaults(func=apply_staintrans)

    eval_parser = subparsers.add_parser("eval", parents=[apply_parser], add_help=False)
    eval_parser.add_argument("--stainaugm_color_space", type=str)
    eval_parser.add_argument("--stainaugm_factor", type=float)
    eval_parser.add_argument(
        "--save_predictions", action="store_true", dest="save_pred"
    )
    eval_parser.set_defaults(
        apply=True, postprocess=True, func=_run_evaluate_staintrans
    )

    run_staintrans(parser.parse_args())
