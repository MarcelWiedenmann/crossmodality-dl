import random
from dataclasses import dataclass
from logging import info
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from dataclasses_json import Undefined, dataclass_json
from torch.utils.data import ConcatDataset, Dataset

import utils
from data import (
    AugmentedMultiscaleDataset,
    MultiscaleDataset,
    NumpyDatasetSource,
    ResamplingMultiscaleDataset,
)
from data_sampling import DataEntryGroup
from processing import Processing, ProcessingConfig


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class DataPreparationConfig(ProcessingConfig):
    basic_augment: bool = True
    color_augment: Optional[str] = None
    color_augment_hed_sigma: Optional[float] = None
    color_augment_hed_extract_tissue_fg_fn: Optional[str] = None
    color_augment_hsv_offset: Optional[float] = None
    resampling_seed: int = random.randint(0, 2**32)

    def __post_init__(self):
        color_augment = self.color_augment
        if color_augment == "hed":
            if self.color_augment_hed_sigma is None:
                raise ValueError(f"Not all parameters provided for {color_augment}.")
            # Masking is truly optional.
        elif color_augment == "hsv":
            if self.color_augment_hsv_offset is None:
                raise ValueError(f"Not all parameters provided for {color_augment}.")
        elif color_augment is not None:
            raise ValueError(f"Unrecognized color space: {color_augment}.")


class DataPreparation(Processing):
    def __init__(
        self,
        working_dir: Path,
        cfg: DataPreparationConfig,
        dry_run: bool = False,
        debug: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(
            name if name is not None else "data_preparation",
            working_dir,
            cfg,
            dry_run,
        )
        self._debug = debug

    def _run(
        self,
        train_strata: List[DataEntryGroup],
        val_strata: List[DataEntryGroup],
        create_dataset_fn: Callable[
            [NumpyDatasetSource], MultiscaleDataset
        ] = MultiscaleDataset,
    ) -> Tuple[Dataset, Dataset]:
        cfg: DataPreparationConfig = self._cfg
        debug = self._debug

        train_data = []
        if debug:
            info("Debug mode: Training data will not be augmented.")
        for stratum in train_strata:
            data = create_dataset_fn(NumpyDatasetSource(stratum.all_entries))
            if not debug:
                data = AugmentedMultiscaleDataset(
                    data,
                    basic=cfg.basic_augment,
                    color=cfg.color_augment,
                    color_hed_sigma=cfg.color_augment_hed_sigma,
                    color_hed_extract_tissue_fg_fn=utils.import_function(
                        cfg.color_augment_hed_extract_tissue_fg_fn
                    )
                    if cfg.color_augment_hed_extract_tissue_fg_fn is not None
                    else None,
                    color_hsv_offset=cfg.color_augment_hsv_offset,
                )
            if stratum.num_entries_to_choose is not None:
                data = ResamplingMultiscaleDataset(
                    data, stratum.num_entries_to_choose, cfg.resampling_seed
                )
            train_data.append(data)
        train_data = ConcatDataset(train_data) if len(train_data) > 0 else Dataset()

        val_data = []
        for stratum in val_strata:
            data = create_dataset_fn(NumpyDatasetSource(stratum.all_entries))
            if stratum.num_entries_to_choose is not None:
                data = ResamplingMultiscaleDataset(
                    data,
                    stratum.num_entries_to_choose,
                    cfg.resampling_seed,
                    only_initial_resampling=True,
                )
            val_data.append(data)
        val_data = ConcatDataset(val_data) if len(val_data) > 0 else Dataset()

        return train_data, val_data
