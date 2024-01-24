import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dataclasses_json import Undefined, dataclass_json
from processing import Processing, ProcessingConfig
from torch.utils.data import ConcatDataset, Dataset

from staintrans.data import ResamplingStainTransferTrainingDataset
from staintrans.data_sampling import DataEntryAndHistDistGroup


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class DataPreparationConfig(ProcessingConfig):
    rescale_data: bool
    include_segmentation_targets: bool
    resampling_seed: int = random.randint(0, 2**32)


class DataPreparation(Processing):
    def __init__(
        self,
        working_dir: Path,
        cfg: DataPreparationConfig,
        dry_run: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(
            name if name is not None else "staintrans_data_preparation",
            working_dir,
            cfg,
            dry_run,
        )

    def _run(self, train_strata: List[DataEntryAndHistDistGroup]) -> Dataset:
        cfg: DataPreparationConfig = self._cfg

        train_data = []
        for stratum in train_strata:
            data = ResamplingStainTransferTrainingDataset(
                stratum.all_entries,
                cfg.rescale_data,
                cfg.include_segmentation_targets,
                stratum.num_entries_to_choose,
                cfg.resampling_seed,
            )
            train_data.append(data)
        train_data = ConcatDataset(train_data)

        return train_data
