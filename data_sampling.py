from dataclasses import dataclass
from logging import info
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from dataclasses_json import Undefined, dataclass_json
from scipy.cluster.vq import kmeans2

import tiling
from data import DataEntry
from processing import Processing, ProcessingConfig


class DataEntryGroup(NamedTuple):
    all_entries: List[DataEntry]
    num_entries_to_choose: Optional[int]

    def __len__(self) -> int:
        return (
            self.num_entries_to_choose
            if self.num_entries_to_choose is not None
            else len(self.all_entries)
        )


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class FilteringDataSamplingConfig(ProcessingConfig):
    tile_filter_label_fg_ratio: Optional[float] = None
    tile_filter_tissue_fg_ratio: Optional[float] = None


class FilteringDataSampling(Processing):
    def __init__(
        self,
        working_dir: Path,
        cfg: FilteringDataSamplingConfig,
        dry_run: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(
            name if name is not None else "filtering_data_sampling",
            working_dir,
            cfg,
            dry_run,
        )

    def _run(
        self,
        train_entries: Dict[str, List[DataEntry]],
        val_entries: Dict[str, List[DataEntry]],
    ) -> Tuple[List[DataEntryGroup], List[DataEntryGroup]]:
        cfg: FilteringDataSamplingConfig = self._cfg

        train_entries = _flatten_and_filter(train_entries, cfg)
        val_entries = _flatten_and_filter(val_entries, cfg)
        num_train = len(train_entries)
        num_val = len(val_entries)
        train_entries = DataEntryGroup(train_entries, num_train)
        val_entries = DataEntryGroup(val_entries, num_val)

        num_total = num_train + num_val
        info(
            (
                f"Number of filtered examples: train={num_train} ({num_train/num_total:.2f}), "
                f"val={num_val} ({num_val/num_total:.2f}), total={num_total}."
            )
        )

        return [train_entries], [val_entries]


def _flatten_and_filter(
    entries_per_image: Dict[str, List[DataEntry]], cfg: FilteringDataSamplingConfig
) -> List[DataEntry]:
    filtered_entries = []

    for image_name, entries in entries_per_image.items():
        fullscale_sample_tiling = entries[0].samples[0].parent
        target_tiling = (
            entries[0].targets[0].parent if len(entries[0].targets) > 0 else None
        )

        tissue_ratio_path = fullscale_sample_tiling / "tissue_fg_ratios.lut"
        label_ratio_path = (
            target_tiling / "label_fg_ratios.lut" if target_tiling is not None else None
        )
        entry_filter = _create_entry_filter_predicate(
            image_name,
            cfg,
            tissue_ratio_path,
            label_ratio_path,
        )

        for entry in entries:
            fullscale_sample = entry.samples[0]
            assert fullscale_sample.parent == fullscale_sample_tiling
            assert len(entry.targets) == 0 or entry.targets[0].parent == target_tiling
            assert (
                len(entry.targets) == 0
                or entry.targets[0].name == fullscale_sample.name
            )
            if entry_filter(entry):
                filtered_entries.append(entry)

    return filtered_entries


def _create_entry_filter_predicate(
    image_name: str,
    cfg: FilteringDataSamplingConfig,
    tissue_ratio_path: Path,
    label_ratio_path: Optional[Path],
) -> Callable[[DataEntry], bool]:
    predicates = []
    if cfg.tile_filter_label_fg_ratio is not None:
        info(
            (
                f"Filtering {image_name} by label FG/BG ratio. Ratios path: {label_ratio_path}, threshold: "
                f"{cfg.tile_filter_label_fg_ratio}."
            )
        )
        predicates.append(
            _create_fg_ratio_filter_predicate(
                lambda e: e.targets[0],
                label_ratio_path,
                cfg.tile_filter_label_fg_ratio,
            )
        )
    if cfg.tile_filter_tissue_fg_ratio is not None:
        info(
            (
                f"Filtering {image_name} by tissue FG/BG ratio. Ratios path: {tissue_ratio_path}, threshold: "
                f"{cfg.tile_filter_tissue_fg_ratio}."
            )
        )
        predicates.append(
            _create_fg_ratio_filter_predicate(
                lambda e: e.samples[0],
                tissue_ratio_path,
                cfg.tile_filter_tissue_fg_ratio,
            )
        )

    def _apply_predicates(entry: DataEntry):
        for p in predicates:
            if not p(entry):
                return False
        return True

    return _apply_predicates


def _create_fg_ratio_filter_predicate(
    tile_from_entry: Callable[[DataEntry], Path],
    tile_ratios_path: Path,
    threshold_ratio: float,
) -> Callable[[DataEntry], bool]:
    tile_ratios = np.load(tile_ratios_path)

    def fg_ratio_predicate(entry: DataEntry) -> bool:
        tile = tile_from_entry(entry)
        flat_idx, _ = tiling.get_tile_indices(tile)
        return tile_ratios[flat_idx] > threshold_ratio

    return fg_ratio_predicate


##### Stratified sampling #####


class StratifyingDataSampling(Processing):
    def __init__(
        self,
        working_dir: Path,
        dry_run: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(
            name if name is not None else "stratifying_data_sampling",
            working_dir,
            None,
            dry_run,
        )

    def _run(
        self,
        train_entries: Dict[str, List[DataEntry]],
        val_entries: Dict[str, List[DataEntry]],
    ) -> Tuple[List[DataEntryGroup], List[DataEntryGroup]]:
        train_entries = _flatten_and_add_fg_ratios(train_entries)
        fg_ratios = list(train_entries.values())

        # Cluster tissue and label ratios of all training tiles. This allows stratifying the tiles based on their
        # "relevance" for training. At the moment, three levels of relevance are presumed:
        #  1. Highly relevant tiles: contain significant portions of both tissue and labels
        #  2. Reasonably relevant tiles: contain significant portions of tissue but less labels (i.e. they are mostly
        #     stroma)
        #  3. Less relevant tiles: contain significant portions of neither tissue nor labels (i.e. they are mostly glass
        #     slide).
        # These presumptions are encouraged by the initialization of the clusters' centroids below.
        # Note that we do not include the validation tiles into the clustering but stratify them based on the clustering
        # results on only the training tiles, i.e. the stratification is a learned characteristic of the training data.
        mean_tissue = np.mean([r.tissue for r in fg_ratios])
        mean_label = np.mean([r.label for r in fg_ratios])
        info(
            f"Mean FG/BG ratios in training split prior to sampling: tissue={mean_tissue}, label={mean_label}."
        )

        init_centroids = np.array([[0, 0], [mean_tissue, mean_label], [1, 1]])
        k = len(init_centroids)
        centroids, cluster_labels = kmeans2(
            np.asarray([(r.tissue, r.label) for r in fg_ratios]), init_centroids
        )
        # Order clusters from most to least relevant (i.e. order according to label ratio).
        cluster_order = np.argsort(centroids[:, 1])[::-1]
        info(
            (
                f"Stratifying training and validation splits into {k} strata, respectively, "
                f"around centroids derived from training split:\n{centroids[cluster_order]}."
            )
        )

        # Stratify training entries according to the determined clustering.
        train_entries_strata = [[] for _ in range(k)]
        for ct, entry in zip(cluster_labels, train_entries.keys()):
            train_entries_strata[cluster_order[ct]].append(entry)
        info(
            f"Stratified training split into strata of sizes: {[len(stratum) for stratum in train_entries_strata]}."
        )

        # Stratify validation entries by assigning each sample to its closest centroid in terms of tissue and label
        # ratios.
        val_entries = _flatten_and_add_fg_ratios(val_entries)
        val_entries_strata = [[] for _ in range(k)]
        for entry, ratios in val_entries.items():
            ct = np.argmin(
                [
                    (ct[0] - ratios.tissue) ** 2 + (ct[1] - ratios.label) ** 2
                    for ct in centroids
                ]
            )
            val_entries_strata[cluster_order[ct]].append(entry)
        info(
            f"Stratified validation split into strata of sizes: {[len(stratum) for stratum in val_entries_strata]}."
        )

        train_strata, num_train = _add_num_samples(
            train_entries_strata, centroids, cluster_order, 1.0, "training"
        )
        val_strata, num_val = _add_num_samples(
            val_entries_strata, centroids, cluster_order, 1.0, "validation"
        )

        num_total = num_train + num_val
        info(
            (
                f"Number of sampled examples: train={num_train} ({num_train/num_total:.2f}), "
                f"val={num_val} ({num_val/num_total:.2f}), total={num_total}."
            )
        )

        return train_strata, val_strata


class _FgRatios(NamedTuple):
    tissue: float
    label: float


def _flatten_and_add_fg_ratios(
    entries_per_image: Dict[str, List[DataEntry]]
) -> Dict[DataEntry, _FgRatios]:
    entries_and_ratios = {}

    for entries in entries_per_image.values():
        fullscale_sample_tiling = entries[0].samples[0].parent
        target_tiling = entries[0].targets[0].parent

        tissue_ratios = np.load(fullscale_sample_tiling / "tissue_fg_ratios.lut")
        label_ratios = np.load(target_tiling / "label_fg_ratios.lut")

        for entry in entries:
            fullscale_sample = entry.samples[0]
            target = entry.targets[0]
            assert fullscale_sample.parent == fullscale_sample_tiling
            assert target.parent == target_tiling
            assert fullscale_sample.name == target.name
            fullscale_sample_flat_idx, _ = tiling.get_tile_indices(fullscale_sample)
            target_flat_idx, _ = tiling.get_tile_indices(target)
            assert fullscale_sample_flat_idx == target_flat_idx
            tissue = tissue_ratios[fullscale_sample_flat_idx]
            label = label_ratios[target_flat_idx]
            assert entry not in entries_and_ratios
            entries_and_ratios[entry] = _FgRatios(tissue, label)

    return entries_and_ratios


def _add_num_samples(
    entries_strata: List[List[DataEntry]],
    centroids: np.ndarray,
    cluster_order: np.ndarray,
    highest_stratum_sample_rate: float,
    split_name: str,
) -> Tuple[List[DataEntryGroup], int]:
    strata = []
    num_samples_strata = 0
    mean_tissue = 0
    mean_label = 0
    desired_mean_label = 0.51
    for i, entries_stratum in enumerate(entries_strata):
        centroid = centroids[cluster_order[i]]
        num_total_stratum = len(entries_stratum)
        # Undersample less relevant entries compared to more relevant ones such that the resulting overall sampling is
        # roughly balanced in terms of label foreground and still contains some glass slide as negative examples.
        if i == 0:
            num_samples_stratum = int(highest_stratum_sample_rate * num_total_stratum)
        else:
            while True:
                info(f"The desired mean label FG/BG ratio is {desired_mean_label:.3f}.")
                try:
                    current_mean_label = mean_label / num_samples_strata
                    sample_rate = (current_mean_label - desired_mean_label) / (
                        desired_mean_label - centroid[1]
                    )
                except ZeroDivisionError:  # Just in case.
                    sample_rate = highest_stratum_sample_rate
                if sample_rate >= 0.0:
                    break
                else:
                    info(
                        "Could not fulfill desired ratio. Retrying with decreased number."
                    )
                    desired_mean_label = current_mean_label - 0.01
            num_samples_stratum = int(sample_rate * num_samples_strata)
            desired_mean_label -= 0.01
        info(
            f"Sampling {num_samples_stratum} out of {num_total_stratum} {split_name} entries from stratum #{i+1}."
        )
        strata.append(DataEntryGroup(entries_stratum, num_samples_stratum))
        num_samples_strata += num_samples_stratum
        mean_tissue += num_samples_stratum * centroid[0]
        mean_label += num_samples_stratum * centroid[1]
    mean_tissue /= num_samples_strata
    mean_label /= num_samples_strata
    info(
        f"Expected mean FG/BG ratios in {split_name} split: tissue={mean_tissue}, label={mean_label}."
    )
    return strata, num_samples_strata
