from dataclasses import dataclass
from logging import info
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import tiling
from dataclasses_json import Undefined, dataclass_json
from processing import Processing, ProcessingConfig
from scipy.stats import wasserstein_distance

from data import DataEntry
from staintrans.data import DataEntryAndHistDist


class DataEntryAndHistDistGroup(NamedTuple):
    all_entries: List[DataEntryAndHistDist]
    num_entries_to_choose: int

    def __len__(self) -> int:
        return self.num_entries_to_choose


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class DataSamplingConfig(ProcessingConfig):
    pair_filter_tissue_fg_overlap: float = 0.8
    num_train_samples: Optional[int] = None


class DataSampling(Processing):
    def __init__(
        self,
        working_dir: Path,
        cfg: DataSamplingConfig,
        dry_run: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(
            name if name is not None else "staintrans_data_sampling",
            working_dir,
            cfg,
            dry_run,
        )

    def _run(
        self,
        serial_train_entries: Dict[str, List[DataEntry]],
        terminal_train_entries: Dict[str, List[DataEntry]],
        serial2terminal: Dict[str, str],
    ) -> List[DataEntryAndHistDistGroup]:
        cfg: DataSamplingConfig = self._cfg

        train_entries = filter_tiles_by_tissue_fg_overlap_and_pair(
            serial_train_entries,
            terminal_train_entries,
            serial2terminal,
            cfg.pair_filter_tissue_fg_overlap,
        )

        train_entries = _flatten_and_add_tissue_ratio_and_hist_dist(train_entries)

        entries = [
            DataEntryAndHistDist(e, trhd.hist_dist) for e, trhd in train_entries.items()
        ]
        num_train_samples = (
            cfg.num_train_samples if cfg.num_train_samples is not None else len(entries)
        )
        info(f"Number of sampled examples: train={num_train_samples}.")
        return [DataEntryAndHistDistGroup(entries, num_train_samples)]


def filter_tiles_by_tissue_fg_overlap_and_pair(
    serial_train_entries: Dict[str, List[DataEntry]],
    terminal_train_entries: Dict[str, List[DataEntry]],
    serial2terminal: Optional[Dict[str, str]],
    pair_filter_tissue_fg_overlap: float,
    tissue_fg_overlap_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, List[Tuple[int, DataEntry]]]:
    """
    Format of the returned dictionary:
      key: serial image name
      value: list of tuples of form (flat tile index, data entry of serial tile path and terminal tile path)
    """
    filtered_pairs_per_image = {}
    for serial_image_name, serial_entries in serial_train_entries.items():
        terminal_image_name = (
            serial2terminal[serial_image_name]
            if serial2terminal is not None
            else serial_image_name
        )
        terminal_entries = terminal_train_entries[terminal_image_name]

        if tissue_fg_overlap_paths is not None:
            tissue_fg_overlaps_path = tissue_fg_overlap_paths[serial_image_name]
        else:
            serial_tiling_dir = serial_entries[0].samples[0].parent
            tissue_fg_overlaps_path = (
                serial_tiling_dir / "serial_terminal_tissue_fg_overlaps.lut"
            )
        info(
            (
                f"Filtering image pair {serial_image_name} <-> {terminal_image_name} by overlap of serial and terminal "
                f"tissue foreground. Overlap path: {tissue_fg_overlaps_path}, threshold: "
                f"{pair_filter_tissue_fg_overlap}."
            )
        )
        tissue_fg_overlaps = np.load(tissue_fg_overlaps_path)

        filtered_pairs = []
        for p, (serial_entry, terminal_entry) in enumerate(
            zip(serial_entries, terminal_entries)
        ):
            _check_are_counterparts(
                p,
                serial_entry.samples[0],
                terminal_entry.samples[0],
                # The terminal (segmentation) target may be required when employing segmentation-specific losses.
                terminal_entry.targets[0] if len(terminal_entry.targets) > 0 else None,
                serial2terminal,
            )
            overlap = tissue_fg_overlaps[p]
            if overlap > pair_filter_tissue_fg_overlap:
                pair = DataEntry(
                    (serial_entry.samples[0],),
                    (terminal_entry.samples[0], terminal_entry.targets[0])
                    if len(terminal_entry.targets) > 0
                    else (terminal_entry.samples[0],),
                )
                filtered_pairs.append((p, pair))
        filtered_pairs_per_image[serial_image_name] = filtered_pairs
    info(
        f"Number of filtered examples: {sum(len(t) for t in filtered_pairs_per_image.values())}."
    )
    return filtered_pairs_per_image


def _check_are_counterparts(
    flat_idx: int,
    serial_tile_path: Path,
    terminal_tile_path: Path,
    terminal_target_tile_path: Optional[Path],
    serial2terminal: Optional[Dict[str, str]],
):
    if serial2terminal is not None:
        assert (
            serial2terminal[serial_tile_path.parent.name]
            == terminal_tile_path.parent.name
        )
    else:
        assert serial_tile_path.parent.name == terminal_tile_path.parent.name
        assert serial_tile_path.name == terminal_tile_path.name

    (
        serial_valid_tile_shape,
        serial_overlap_shape,
    ) = tiling.get_valid_tile_shape_and_overlap(serial_tile_path.parent.parent)
    (
        terminal_valid_tile_shape,
        terminal_overlap_shape,
    ) = tiling.get_valid_tile_shape_and_overlap(terminal_tile_path.parent.parent)
    assert serial_valid_tile_shape == terminal_valid_tile_shape
    assert serial_overlap_shape == terminal_overlap_shape

    serial_flat_idx, serial_idx_yx = tiling.get_tile_indices(serial_tile_path)
    terminal_flat_idx, terminal_idx_yx = tiling.get_tile_indices(terminal_tile_path)
    assert serial_flat_idx == flat_idx
    assert terminal_flat_idx == flat_idx
    assert serial_idx_yx == terminal_idx_yx

    if terminal_target_tile_path is not None:
        assert terminal_tile_path.parent.name == terminal_target_tile_path.parent.name
        assert terminal_tile_path.name == terminal_target_tile_path.name


class _TissueRatioAndHistDist(NamedTuple):
    tissue_ratio: float
    hist_dist: float


def _flatten_and_add_tissue_ratio_and_hist_dist(
    entries_per_image: Dict[str, List[Tuple[int, DataEntry]]]
) -> Dict[DataEntry, _TissueRatioAndHistDist]:
    entries_and_ratios = {}

    for indexed_entries in entries_per_image.values():
        serial_tiling = indexed_entries[0][1].samples[0].parent
        terminal_tiling = indexed_entries[0][1].targets[0].parent

        tissue_ratios = np.load(serial_tiling / "tissue_fg_ratios.lut")
        serial_hists = np.load(serial_tiling / "hsv_hist.lut")
        terminal_hists = np.load(terminal_tiling / "hsv_hist.lut")
        hist_bins = np.arange(256)

        for p, entry in indexed_entries:
            assert entry.samples[0].parent == serial_tiling
            assert entry.targets[0].parent == terminal_tiling
            tissue_ratio = tissue_ratios[p]
            serial_hist = serial_hists[p]
            terminal_hist = terminal_hists[p]
            hist_dist = 0.0
            for c in range(3):
                hist_dist += wasserstein_distance(
                    hist_bins,
                    hist_bins,
                    u_weights=serial_hist[c],
                    v_weights=terminal_hist[c],
                )
            hist_dist /= 3.0
            assert entry not in entries_and_ratios
            entries_and_ratios[entry] = _TissueRatioAndHistDist(tissue_ratio, hist_dist)

    return entries_and_ratios
