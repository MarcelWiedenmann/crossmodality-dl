from typing import Iterable, List, Optional, Tuple

import constants as c

# Labeled data set:


def get_terminal_sample_dirs(
    shape: Tuple[int, int] = (512, 512), overlap: Tuple[int, int] = (0, 0)
) -> List[str]:
    return [
        f"dataset_208_tiled/he/level_0/shape_{shape[0]}_{shape[1]}_overlap_{overlap[0]}_{overlap[1]}",
        f"dataset_208_tiled/he/level_2/shape_{shape[0]}_{shape[1]}_overlap_{overlap[0]}_{overlap[1]}",
        f"dataset_208_tiled/he/level_3/shape_{shape[0]}_{shape[1]}_overlap_{overlap[0]}_{overlap[1]}",
    ]


def get_normalized_terminal_sample_dirs(
    color_space: str, factor: float, cv_fold: int
) -> List[str]:
    pred_dir = f"experiments/stainnorm_{color_space}_{factor:.5f}_cv_fold_{cv_fold}/predictions/"
    return [
        pred_dir + "level_0/shape_512_512_overlap_0_0",
        pred_dir + "level_2/shape_512_512_overlap_0_0",
        pred_dir + "level_3/shape_512_512_overlap_0_0",
    ]


def get_staintrans_terminal_sample_dirs(
    staintrans_name: str, checkpoint_epoch: Optional[int], cv_fold: int
) -> List[str]:
    if checkpoint_epoch is not None:
        epoch_format_str = f"{checkpoint_epoch:03}"
    else:
        epoch_format_str = "best"
    pred_dir = (
        f"experiments/staintrans_{staintrans_name}_cv_fold_{cv_fold}/"
        f"predictions/terminal/checkpoint_{epoch_format_str}/blended/"
    )
    return [
        pred_dir + "level_0/shape_512_512_overlap_0_0",
        pred_dir + "level_2/shape_512_512_overlap_0_0",
        pred_dir + "level_3/shape_512_512_overlap_0_0",
    ]


def get_terminal_targets_dir(shape: Tuple[int, int] = (512, 512)) -> str:
    return (
        "dataset_208_tiled/cksox/masks_thr_1000/holes_filled_dia_BR_15_LU_50/level_0/"
        f"shape_{shape[0]}_{shape[1]}_overlap_0_0"
    )


# Independent data set:


def get_serial_sample_dirs(
    shape: Tuple[int, int] = (512, 512), overlap: Tuple[int, int] = (0, 0)
) -> List[str]:
    return [
        f"serial_he_tiled/level_0/shape_{shape[0]}_{shape[1]}_overlap_{overlap[0]}_{overlap[1]}",
        f"serial_he_tiled/level_2/shape_{shape[0]}_{shape[1]}_overlap_{overlap[0]}_{overlap[1]}",
        f"serial_he_tiled/level_3/shape_{shape[0]}_{shape[1]}_overlap_{overlap[0]}_{overlap[1]}",
    ]


def get_normalized_serial_sample_dirs(
    color_space: str, factor: float, cv_fold: int
) -> List[str]:
    pred_dir = f"experiments/stainnorm_{color_space}_{factor:.5f}_cv_fold_{cv_fold}/predictions/serial/"
    return [
        pred_dir + "level_0/shape_512_512_overlap_0_0",
        pred_dir + "level_2/shape_512_512_overlap_0_0",
        pred_dir + "level_3/shape_512_512_overlap_0_0",
    ]


def get_staintrans_serial_sample_dirs(
    staintrans_name: str, checkpoint_epoch: Optional[int], cv_fold: int
) -> List[str]:
    if checkpoint_epoch is not None:
        epoch_format_str = f"{checkpoint_epoch:03}"
    else:
        epoch_format_str = "best"
    pred_dir = (
        f"experiments/staintrans_{staintrans_name}_cv_fold_{cv_fold}/"
        f"predictions/serial/checkpoint_{epoch_format_str}/blended/"
    )
    return [
        pred_dir + "level_0/shape_512_512_overlap_0_0",
        pred_dir + "level_2/shape_512_512_overlap_0_0",
        pred_dir + "level_3/shape_512_512_overlap_0_0",
    ]


def get_serial_targets_dir(shape: Tuple[int, int] = (512, 512)) -> str:
    return get_terminal_targets_dir(shape)


def get_serial_tissue_masks_dir(shape: Tuple[int, int] = (512, 512)) -> str:
    return (
        f"serial_he_tissue_masks_tiled/level_0/shape_{shape[0]}_{shape[1]}_overlap_0_0"
    )


# Folds:


def get_folds() -> List[Tuple[List[str], List[str], List[str], List[str]]]:
    """
    Fixed CV folds for all experiments. Grouping was done once in advance using random sampling stratified by organ (BR,
    LU) for the divison into training and test subsets, and subsequent simple random sampling for splitting off a
    validation subset from each training subset. Serial test images are always counterparts of terminal test images per
    fold.
    """
    return [
        # Fold 1
        (
            # Train
            [
                "H2021-192_exp2_s01_HP-224363BR.ome.tiff",
                "H2021-192_exp2_s03_HP-224551BR.ome.tiff",
                "H2021-192_exp2_s05_HP-82163LU.ome.tiff",
                "H2021-192_exp2_s07_HP-58283LU.ome.tiff",
                "H2021-192_exp2_s08_HP-58289LU.ome.tiff",
                "H2021-192_exp2_s09_HP-58982LU.ome.tiff",
            ],
            # Validation
            ["H2021-192_exp2_s02_HP-224470BR.ome.tiff"],
            # Terminal test
            [
                "H2021-192_exp2_s04_HP-224388BR.ome.tiff",
                "H2021-192_exp2_s06_HP-70699LU.ome.tiff",
            ],
            # Serial test
            [
                c.terminal2serial["H2021-192_exp2_s04_HP-224388BR.ome.tiff"],
                c.terminal2serial["H2021-192_exp2_s06_HP-70699LU.ome.tiff"],
            ],
        ),
        # Fold 2
        (
            # Train
            [
                "H2021-192_exp2_s02_HP-224470BR.ome.tiff",
                "H2021-192_exp2_s03_HP-224551BR.ome.tiff",
                "H2021-192_exp2_s04_HP-224388BR.ome.tiff",
                "H2021-192_exp2_s05_HP-82163LU.ome.tiff",
                "H2021-192_exp2_s08_HP-58289LU.ome.tiff",
                "H2021-192_exp2_s09_HP-58982LU.ome.tiff",
            ],
            # Validation
            ["H2021-192_exp2_s06_HP-70699LU.ome.tiff"],
            # Terminal test
            [
                "H2021-192_exp2_s01_HP-224363BR.ome.tiff",
                "H2021-192_exp2_s07_HP-58283LU.ome.tiff",
            ],
            # Serial test
            [
                c.terminal2serial["H2021-192_exp2_s01_HP-224363BR.ome.tiff"],
                c.terminal2serial["H2021-192_exp2_s07_HP-58283LU.ome.tiff"],
            ],
        ),
        # Fold 3
        (
            # Train
            [
                "H2021-192_exp2_s01_HP-224363BR.ome.tiff",
                "H2021-192_exp2_s03_HP-224551BR.ome.tiff",
                "H2021-192_exp2_s04_HP-224388BR.ome.tiff",
                "H2021-192_exp2_s05_HP-82163LU.ome.tiff",
                "H2021-192_exp2_s06_HP-70699LU.ome.tiff",
                "H2021-192_exp2_s07_HP-58283LU.ome.tiff",
            ],
            # Validation
            ["H2021-192_exp2_s08_HP-58289LU.ome.tiff"],
            # Terminal test
            [
                "H2021-192_exp2_s02_HP-224470BR.ome.tiff",
                "H2021-192_exp2_s09_HP-58982LU.ome.tiff",
            ],
            # Serial test
            [
                c.terminal2serial["H2021-192_exp2_s02_HP-224470BR.ome.tiff"],
                c.terminal2serial["H2021-192_exp2_s09_HP-58982LU.ome.tiff"],
            ],
        ),
        # Fold 4
        (
            # Train
            [
                "H2021-192_exp2_s01_HP-224363BR.ome.tiff",
                "H2021-192_exp2_s02_HP-224470BR.ome.tiff",
                "H2021-192_exp2_s06_HP-70699LU.ome.tiff",
                "H2021-192_exp2_s07_HP-58283LU.ome.tiff",
                "H2021-192_exp2_s08_HP-58289LU.ome.tiff",
                "H2021-192_exp2_s09_HP-58982LU.ome.tiff",
            ],
            # Validation
            ["H2021-192_exp2_s04_HP-224388BR.ome.tiff"],
            # Terminal test
            [
                "H2021-192_exp2_s03_HP-224551BR.ome.tiff",
                "H2021-192_exp2_s05_HP-82163LU.ome.tiff",
            ],
            # Serial test
            [
                c.terminal2serial["H2021-192_exp2_s03_HP-224551BR.ome.tiff"],
                c.terminal2serial["H2021-192_exp2_s05_HP-82163LU.ome.tiff"],
            ],
        ),
        # Fold 5
        (
            # Train
            [
                "H2021-192_exp2_s01_HP-224363BR.ome.tiff",
                "H2021-192_exp2_s02_HP-224470BR.ome.tiff",
                "H2021-192_exp2_s03_HP-224551BR.ome.tiff",
                "H2021-192_exp2_s04_HP-224388BR.ome.tiff",
                "H2021-192_exp2_s06_HP-70699LU.ome.tiff",
                "H2021-192_exp2_s07_HP-58283LU.ome.tiff",
                "H2021-192_exp2_s09_HP-58982LU.ome.tiff",
            ],
            # Validation
            ["H2021-192_exp2_s05_HP-82163LU.ome.tiff"],
            # Terminal test
            ["H2021-192_exp2_s08_HP-58289LU.ome.tiff"],
            # Serial test
            [c.terminal2serial["H2021-192_exp2_s08_HP-58289LU.ome.tiff"]],
        ),
    ]


def get_enumerated_folds(
    fold_indices: Optional[List[int]] = None,
) -> Iterable[Tuple[int, Tuple[List[str], List[str], List[str], List[str]]]]:
    cv_folds = get_folds()
    if fold_indices is not None:
        cv_folds = [cv_folds[i] for i in fold_indices]
        cv_folds = zip(fold_indices, cv_folds)
    else:
        cv_folds = enumerate(cv_folds)
    return cv_folds
