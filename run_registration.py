"""
Runs the registration of the serial H&E images with the terminal ones using elastix via wsireg.
"""
import os
from argparse import ArgumentParser
from logging import info

import numpy as np
from wsireg.wsireg2d import WsiReg2D

import configure_logging  # Just needs to be imported.
import constants as c


def run_registration(serial_img_name_or_idx):
    serial_img_name = _get_serial_img_name(serial_img_name_or_idx)
    terminal_img_name = c.serial2terminal[serial_img_name]

    serial_img_path = _get_serial_img_path(serial_img_name)
    terminal_img_path = _get_terminal_img_path(terminal_img_name)

    info(
        f"Will register serial image {serial_img_name} to terminal image {terminal_img_name}."
    )

    # RGB images are only detected if channels come last. Also, the output writer seems to only properly support uint,
    # so convert the pixel types from float32.
    terminal_img = (np.moveaxis(np.load(terminal_img_path), 0, -1) * 255).astype(
        "uint8"
    )
    serial_img = (np.moveaxis(np.load(serial_img_path), 0, -1) * 255).astype("uint8")

    info(f"Serial image: shape={serial_img.shape}, dtype={serial_img.dtype}.")
    info(f"Terminal image: shape={terminal_img.shape}, dtype={terminal_img.dtype}.")

    # Dots seem to confuse wsireg, so replace them.
    serial_img_name_no_dots = serial_img_name.replace(".", "_")
    project_dir = c.scratch_dir / "serial_he_registered" / serial_img_name_no_dots
    project_dir.mkdir(parents=True)
    reg_graph = WsiReg2D(
        serial_img_name_no_dots,
        project_dir,
    )

    reg_graph.add_modality(
        modality_name="terminal",
        image_fp=terminal_img,
        image_res=0.325,
    )

    reg_graph.add_modality(
        modality_name="serial",
        image_fp=serial_img,
        image_res=0.325,
    )

    reg_graph.add_reg_path(
        src_modality_name="serial",
        tgt_modality_name="terminal",
        reg_params=["rigid", "nl"],
    )

    info("Starting registration.")
    reg_graph.register_images()
    reg_graph.save_transformations()
    reg_graph.transform_images(transform_non_reg=False)

    # Recover the old file name for consistency. The "-serial_to_[...]" suffix is added by wsireg.
    os.rename(
        project_dir
        / (serial_img_name_no_dots + "-serial_to_terminal_registered.ome.tiff"),
        project_dir / serial_img_path.name,
    )


def _get_serial_img_name(serial_img_name_or_idx):
    try:
        idx = int(serial_img_name_or_idx)
        serial_img_name = list(c.serial2terminal.keys())[idx]
    except ValueError:
        serial_img_name = serial_img_name_or_idx
    return serial_img_name


def _get_serial_img_path(serial_img_name):
    return (
        c.scratch_dir
        / "serial_he_preprocessed"
        / "level_0"
        / (serial_img_name + ".npy")
    )


def _get_terminal_img_path(terminal_img_name):
    return (
        c.scratch_dir
        / "dataset_208_preprocessed"
        / "he"
        / "level_0"
        / (terminal_img_name + ".npy")
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("serial_image_names_or_indices", nargs="+", type=str)

    args = parser.parse_args()

    for img_name_or_idx in args.serial_image_names_or_indices:
        run_registration(img_name_or_idx)
