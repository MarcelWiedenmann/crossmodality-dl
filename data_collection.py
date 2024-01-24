from dataclasses import dataclass, field
from logging import info
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Optional, Tuple

from dataclasses_json import Undefined, dataclass_json

import constants as c
import utils
from data import DataEntry
from processing import Processing, ProcessingConfig
from tiling import get_tile_indices


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class DataCollectionConfig(ProcessingConfig):
    sample_dirs: List[str]
    targets_dir: Optional[str] = None

    train_images: List[str] = field(default_factory=lambda: [])
    val_images: List[str] = field(default_factory=lambda: [])
    test_images: List[str] = field(default_factory=lambda: [])
    # Test targets may be reused across serial sections, so allow their names to differ from the names of the test
    # samples.
    test_targets: Optional[List[str]] = None

    def __post_init__(self):
        all_splits = self.train_images, self.val_images, self.test_images
        if self.test_targets is not None:
            all_splits += (self.test_targets,)
        all_images = set().union(*all_splits)
        if len(all_images) != sum(len(split) for split in all_splits):
            raise ValueError(
                "Training, validation, and test splits are not pairwise disjoint."
            )
        if self.test_targets is None:
            self.test_targets = list(self.test_images)


class DataCollection(Processing):
    def __init__(
        self,
        working_dir: Path,
        cfg: DataCollectionConfig,
        dry_run: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(
            name if name is not None else "data_collection",
            working_dir,
            cfg,
            dry_run,
        )

    def _run(
        self,
    ) -> Tuple[
        Dict[str, List[DataEntry]],
        Dict[str, List[DataEntry]],
        Dict[str, List[DataEntry]],
    ]:
        """
        Returns training, validation, and test splits. The splits are returned grouped by (sample) image name. Depending
        on the configuration, any of the splits may be empty. Likewise, the target of each entry may be None.
        """
        cfg: DataCollectionConfig = self._cfg
        samples_dir = c.scratch_dir / cfg.sample_dirs[0]
        targets_dir = (
            c.scratch_dir / cfg.targets_dir if cfg.targets_dir is not None else None
        )

        train_sample_tilings = _list_tilings(
            samples_dir, cfg.train_images, cfg.train_images, "Training sample"
        )
        val_sample_tilings = _list_tilings(
            samples_dir, cfg.val_images, cfg.val_images, "Validation sample"
        )
        test_sample_tilings = _list_tilings(
            samples_dir, cfg.test_images, cfg.test_images, "Test sample"
        )

        train_sample_tiles = _list_tiles_per_image(train_sample_tilings)
        val_sample_tiles = _list_tiles_per_image(val_sample_tilings)
        test_sample_tiles = _list_tiles_per_image(test_sample_tilings)

        if targets_dir is not None:
            train_target_tilings = _list_tilings(
                targets_dir, cfg.train_images, cfg.train_images, "Training target"
            )
            val_target_tilings = _list_tilings(
                targets_dir, cfg.val_images, cfg.val_images, "Validation target"
            )
            test_target_tilings = _list_tilings(
                targets_dir, cfg.test_images, cfg.test_targets, "Test target"
            )

            train_target_tiles = _list_tiles_per_image(train_target_tilings)
            val_target_tiles = _list_tiles_per_image(val_target_tilings)
            test_target_tiles = _list_tiles_per_image(test_target_tilings)
        else:
            train_target_tiles = {}
            val_target_tiles = {}
            test_target_tiles = {}

        multiscale_dirs = [c.scratch_dir / d for d in cfg.sample_dirs[1:]]
        for d in multiscale_dirs:
            info(f"Adding matching multi-scale tiles from {d}.")
        train_entries = _create_entries(
            train_sample_tiles, multiscale_dirs, train_target_tiles
        )
        val_entries = _create_entries(
            val_sample_tiles, multiscale_dirs, val_target_tiles
        )
        test_entries = _create_entries(
            test_sample_tiles, multiscale_dirs, test_target_tiles
        )

        num_train = sum(len(entries) for entries in train_entries.values())
        num_val = sum(len(entries) for entries in val_entries.values())
        num_test = sum(len(entries) for entries in test_entries.values())
        num_total = num_train + num_val + num_test
        info(
            (
                f"Number of collected entries: train={num_train} ({num_train/num_total:.2f}), "
                f"val={num_val} ({num_val/num_total:.2f}), test={num_test} ({num_test/num_total:.2f}), "
                f"total={num_total}."
            )
        )

        return train_entries, val_entries, test_entries


def _list_tilings(
    base_dir: Path,
    image_names: List[str],
    tiling_names: List[str],
    tilings_group_name: str,
) -> List[Tuple[str, Path]]:
    tilings = [
        (image_name, base_dir / tiling_name)
        for image_name, tiling_name in zip(image_names, tiling_names)
    ]
    info(f"{tilings_group_name} directories:\n" + pformat(tilings))
    return tilings


def _list_tiles_per_image(tilings: List[Tuple[str, Path]]) -> Dict[str, List[Path]]:
    tiles_per_image = {}
    for image_name, tiling in tilings:
        tiles = utils.list_files(tiling, file_extension=".npy")
        if len(tiles) == 0:
            raise ValueError(f"No tiles found in directory {tiling}.")
        max_flat_idx, *_ = get_tile_indices(tiles[-1])
        if len(tiles) != max_flat_idx + 1:
            raise ValueError(
                f"Expected {max_flat_idx + 1} tiles in directory {tiling} but found {len(tiles)}."
            )
        tiles_per_image[image_name] = tiles
    return tiles_per_image


def _create_entries(
    samples_per_image: Dict[str, List[Path]],
    multiscale_dirs: List[Path],
    targets_per_image: Dict[str, List[Path]],
) -> Dict[str, List[DataEntry]]:
    assert len(targets_per_image) == 0 or len(samples_per_image) == len(
        targets_per_image
    )
    entries_per_image = {}
    for (name, samples) in samples_per_image.items():
        targets = targets_per_image.get(name)
        entries = []
        for i, s in enumerate(samples):
            multiscale_samples = [d / s.parent.name / s.name for d in multiscale_dirs]
            for ms in multiscale_samples:
                if not ms.exists():
                    raise ValueError(f"Multi-scale sample {ms} is missing.")
            multiscale_samples = tuple([s] + multiscale_samples)
            target = (targets[i],) if targets is not None else ()
            entries.append(DataEntry(multiscale_samples, target))
        entries_per_image[name] = entries
    return entries_per_image
