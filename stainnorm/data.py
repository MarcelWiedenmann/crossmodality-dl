from typing import Tuple

import numpy as np

from data import MultiscaleDataset, NumpyDatasetSource


class StainNormalizationDataset(MultiscaleDataset):
    def __init__(self, source: NumpyDatasetSource):
        super().__init__(source)
        first_entry = source.entries[0]
        assert len(first_entry.samples) == 1
        assert len(first_entry.targets) == 0
        self.has_targets = True

    def load_arrays(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        source = self.source
        sample = source.load_array(source.entries[idx].samples[0])
        target = sample.copy()
        return sample, target
