from typing import Dict, List, NamedTuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from data import DataEntry, NumpyDatasetSource, SourceWrappingDataset


class StainTransferTrainingDataset(SourceWrappingDataset):
    def __init__(
        self,
        source: NumpyDatasetSource,
        rescale_data: bool,
        include_segmentation_targets: bool,
    ):
        super().__init__(source)
        first_entry = source.entries[0]
        assert len(first_entry.samples) == 1
        if include_segmentation_targets:
            assert len(first_entry.targets) == 2
        else:
            assert len(first_entry.targets) in [1, 2]
        self._rescale_data = rescale_data
        self._include_segmentation_targets = include_segmentation_targets

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, str]]:
        entry = self.source.entries[idx]
        path_a = entry.samples[0]
        path_b = entry.targets[0]
        a = NumpyDatasetSource.load_array(path_a)
        b = NumpyDatasetSource.load_array(path_b)
        path_a = str(path_a)
        path_b = str(path_b)
        if self._rescale_data:
            # [0, 1] --> [-1, 1]
            a = (a - 0.5) * 2
            b = (b - 0.5) * 2
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)
        if self._include_segmentation_targets:
            sm = torch.from_numpy(
                NumpyDatasetSource.load_array(entry.targets[1], convert_from_mask=True)
            )
            return {
                "A": a,
                "B": b,
                "SM": sm,
                "A_paths": path_a,
                "B_paths": path_b,
            }
        else:
            return {
                "A": a,
                "B": b,
                "A_paths": path_a,
                "B_paths": path_b,
            }


class UnpairedStainTransferTrainingDataset(Dataset):
    def __init__(
        self,
        source_a: NumpyDatasetSource,
        source_b: NumpyDatasetSource,
        rescale_data: bool,
        resampling_seed: int,
    ):
        super().__init__()
        first_entry_a = source_a.entries[0]
        assert len(first_entry_a.samples) == 1
        assert len(first_entry_a.targets) == 0
        first_entry_b = source_b.entries[0]
        assert len(first_entry_b.samples) == 1
        assert len(first_entry_b.targets) == 0
        self._source_a = source_a
        self._source_b = source_b
        self._rescale_data = rescale_data
        self._rng = np.random.default_rng(resampling_seed)

        self._indices_b = np.arange(len(source_b))  # Identity by default.
        self.resample()

    def resample(self):
        """
        Shuffle the elements in B so that no accidental pairing is introduced.
        """
        self._indices_b = self._rng.permutation(np.arange(len(self._source_b)))

    def __len__(self) -> int:
        """
        Since A and B are unpaired, data set sizes can differ, so return the larger size. For the smaller data set,
        out-of-bounds accesses will be wrapped around.
        """
        return max(len(self._source_a), len(self._source_b))

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, str]]:
        idx_a = idx % len(self._source_a)
        idx_b = self._indices_b[idx % len(self._source_b)]
        path_a = self._source_a.entries[idx_a].samples[0]
        path_b = self._source_b.entries[idx_b].samples[0]
        a = torch.from_numpy(NumpyDatasetSource.load_array(path_a))
        b = torch.from_numpy(NumpyDatasetSource.load_array(path_b))
        path_a = str(path_a)
        path_b = str(path_b)
        if self._rescale_data:
            # [0, 1] --> [-1, 1]
            a = (a - 0.5) * 2
            b = (b - 0.5) * 2
        return {
            "A": a,
            "B": b,
            "A_paths": str(path_a),
            "B_paths": str(path_b),
        }


class DataEntryAndHistDist(NamedTuple):
    entry: DataEntry
    hist_dist: float


class ResamplingStainTransferTrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        entries: List[DataEntryAndHistDist],
        rescale_data: bool,
        include_segmentation_targets: bool,
        num_samples: int,
        resampling_seed: int,
    ):
        super().__init__()
        self._delegate = StainTransferTrainingDataset(
            NumpyDatasetSource([e.entry for e in entries]),
            rescale_data,
            include_segmentation_targets,
        )
        self._num_samples = num_samples
        self._rng = np.random.default_rng(resampling_seed)

        # Focus on pairs of tiles around the median histogram distance as these are not too similar and also do not
        # suffer from artifacts such as folds in the tissue that were observed to often cause larger histogram
        # dissimilarities.
        weights = np.asarray([e.hist_dist for e in entries])
        weights = 1 / ((weights - np.median(weights)) ** 2 + 1)
        weights = weights / weights.sum()  # Probability distribution
        self._sample_weights = weights

        self._sampled_indices = np.arange(len(self._delegate))  # Identity by default.
        self.resample()

    def resample(self):
        self._sampled_indices = self._rng.choice(
            len(self._delegate),
            size=self._num_samples,
            replace=False,
            p=self._sample_weights,
        )

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, str]]:
        return self._delegate[self._sampled_indices[idx]]


class StainTransferValidationDataset(SourceWrappingDataset):
    def __init__(self, source: NumpyDatasetSource, rescale_data: bool):
        super().__init__(source)
        first_entry = source.entries[0]
        assert len(first_entry.samples) == 1
        assert len(first_entry.targets) == 2
        self._rescale_data = rescale_data

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, str]]:
        source = self.source
        entry = source.entries[idx]
        a = source.load_array(entry.samples[0])
        b = source.load_array(entry.targets[0])
        # Ground truth segmentation mask.
        sm = torch.from_numpy(
            source.load_array(entry.targets[1], convert_from_mask=True)
        )
        if self._rescale_data:
            # [0, 1] --> [-1, 1]
            a = (a - 0.5) * 2
            b = (b - 0.5) * 2
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)
        return {
            "A": a,
            "B": b,
            "SM": sm,
            "A_paths": str(entry.samples[0]),
            "B_paths": str(entry.targets[0]),
        }


class StainTransferTestDataset(SourceWrappingDataset):
    def __init__(self, source: NumpyDatasetSource, rescale_data: bool):
        super().__init__(source)
        first_entry = source.entries[0]
        assert len(first_entry.samples) == 1
        assert len(first_entry.targets) == 0
        self._rescale_data = rescale_data

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, str]]:
        source = self.source
        entry = source.entries[idx]
        a = source.load_array(entry.samples[0])
        if self._rescale_data:
            # [0, 1] --> [-1, 1]
            a = (a - 0.5) * 2
        a = torch.from_numpy(a)
        return {
            "A": a,
            "A_paths": str(entry.samples[0]),
        }
