import math
import random
from dataclasses import dataclass
from logging import info
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from pytorch_lightning.callbacks import LambdaCallback
from skimage.color import hed2rgb, rgb2hed
from skimage.filters import gaussian
from skimage.util import img_as_float32
from torch.utils.data import Dataset

from tiling import get_tile_indices
from utils import remove_padding


@dataclass(frozen=True)
class DataEntry:
    samples: Tuple[Path, ...]
    targets: Tuple[Path, ...]

    def __post_init__(self):
        samples = self.samples
        assert len(samples) > 0
        first_sample = samples[0]
        for sample in samples[1:]:
            assert sample.name == first_sample.name

        targets = self.targets
        first_target = targets[0] if len(targets) > 0 else None
        first_sample_flat_idx, (
            first_sample_idx_y,
            first_sample_idx_x,
        ) = get_tile_indices(first_sample)
        for target in self.targets:
            assert target.name == first_target.name
            # Targets may have different names than samples if e.g. labels are reused across data sets. But we can at
            # least check whether the tile indices match.
            target_flat_idx, (target_idx_y, target_idx_x) = get_tile_indices(target)
            assert target_flat_idx == first_sample_flat_idx
            assert target_idx_y == first_sample_idx_y
            assert target_idx_x == first_sample_idx_x


class NumpyDatasetSource:
    def __init__(self, entries: List[DataEntry]):
        assert len(entries) > 0
        first_entry = entries[0]
        for entry in entries[1:]:
            assert len(entry.samples) == len(first_entry.samples)
            assert len(entry.targets) == len(first_entry.targets)
        self.entries = NumpyDatasetSource.Entries(entries)
        self.has_targets = len(first_entry.targets) > 0

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def load_array(
        path: Path,
        remove_overlap: Optional[Tuple[int, int]] = None,
        convert_from_mask: bool = False,
    ) -> np.ndarray:
        a = np.load(path)
        if remove_overlap is not None:
            a = remove_padding(a, remove_overlap)
        if convert_from_mask:
            a = np.expand_dims(a.astype("float32"), 0)
        return a

    class Entries:
        """
        Encodes the list of entries as numpy arrays to work around memory problems caused by multiprocessing. See:
        https://github.com/pytorch/pytorch/issues/13246.
        """

        def __init__(self, entries: List[DataEntry]):
            all_samples = []
            all_targets = []
            for _ in range(len(entries[0].samples)):
                all_samples.append([])
            for _ in range(len(entries[0].targets)):
                all_targets.append([])
            for entry in entries:
                for i, s in enumerate(entry.samples):
                    all_samples[i].append(str(s))
                for i, t in enumerate(entry.targets):
                    all_targets[i].append(str(t))
            self.all_samples = np.stack(
                [np.array(samples).astype(np.string_) for samples in all_samples]
            )
            if len(all_targets) > 0:
                self.all_targets = np.stack(
                    [np.array(targets).astype(np.string_) for targets in all_targets]
                )
            else:
                self.all_targets = []

        def __len__(self) -> int:
            return len(self.all_samples[0])

        def __getitem__(self, idx) -> DataEntry:
            samples = tuple(
                [
                    Path(str(samples[idx], encoding="utf-8"))
                    for samples in self.all_samples
                ]
            )
            targets = tuple(
                [
                    Path(str(targets[idx], encoding="utf-8"))
                    for targets in self.all_targets
                ]
            )
            return DataEntry(samples, targets)


class SourceWrappingDataset(Dataset):
    def __init__(self, source: NumpyDatasetSource):
        super().__init__()
        self.source = source
        self.entries = self.source.entries
        self.has_targets = self.source.has_targets

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError()


class MultiscaleDataset(SourceWrappingDataset):
    def __init__(self, source: NumpyDatasetSource):
        super().__init__(source)
        assert len(source.entries[0].targets) <= 1

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        return tuple(torch.from_numpy(a) for a in self.load_arrays(idx))

    def load_arrays(self, idx) -> Tuple[np.ndarray, ...]:
        source = self.source
        entry = source.entries[idx]
        arrays = [source.load_array(s) for s in entry.samples]
        if source.has_targets:
            arrays.append(source.load_array(entry.targets[0], convert_from_mask=True))
        return tuple(arrays)


class AugmentedMultiscaleDataset(Dataset):
    def __init__(
        self,
        delegate: MultiscaleDataset,
        basic: bool,
        color: Optional[str],
        color_hed_sigma: Optional[float],
        color_hed_extract_tissue_fg_fn: Optional[Callable[[np.ndarray], np.ndarray]],
        color_hsv_offset: Optional[float],
    ):
        super().__init__()
        if not delegate.has_targets:
            raise ValueError(
                "Data augmentation is not supported for data sets without targets."
            )
        self._delegate = delegate
        self._numpy_augmentations = []
        self._torch_augmentations = []
        if basic:
            info("Will perform basic data augmentation.")
            self._torch_augmentations.append(random_rotate)
            self._torch_augmentations.append(random_flip)
            self._torch_augmentations.append(random_gaussian_blur)
            self._torch_augmentations.append(random_gaussian_noise)
        if color == "hed":
            info(f"Will perform HED stain color augmentation, sigma={color_hed_sigma}.")
            self._numpy_augmentations.append(
                lambda s, t: random_stain_color_change(
                    s, t, color_hed_sigma, color_hed_extract_tissue_fg_fn
                )
            )
        elif color == "hsv":
            info(f"Will perform HSV color augmentation, offset={color_hsv_offset}.")
            self._torch_augmentations.append(
                lambda s, t: random_color_change(s, t, color_hsv_offset)
            )

    def __len__(self) -> int:
        return len(self._delegate)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        *sample, target = self._delegate.load_arrays(idx)
        sample, target = self._augment(sample, target)
        return *sample, target

    def _augment(
        self, sample: List[np.ndarray], target: np.ndarray
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        for augment in self._numpy_augmentations:
            sample, target = augment(sample, target)
        sample = tuple(torch.from_numpy(s) for s in sample)
        target = torch.from_numpy(target)
        for augment in self._torch_augmentations:
            sample, target = augment(sample, target)
        return sample, target


def random_flip(
    sample: List[torch.Tensor], target: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    flip = random.randint(0, 2)
    # Perform no flipping if flip == 0.
    if flip == 1:
        sample = [TF.hflip(s) for s in sample]
        target = TF.hflip(target)
    elif flip == 2:
        sample = [TF.vflip(s) for s in sample]
        target = TF.vflip(target)
    return sample, target


def random_rotate(
    sample: List[torch.Tensor], target: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    angles = [0, 90, 180, 270]
    angle = angles[random.randint(0, len(angles) - 1)]
    sample = [TF.rotate(s, angle) for s in sample]
    target = TF.rotate(target, angle)
    return sample, target


def random_gaussian_blur(
    sample: List[torch.Tensor], target: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    sigma = 2.0 * random.random()
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    sample = [TF.gaussian_blur(s, kernel_size, sigma) for s in sample]
    return sample, target


def random_gaussian_noise(
    sample: List[torch.Tensor], target: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    sigma = 0.1 * random.random()
    sample = [s + sigma * torch.randn_like(s) for s in sample]
    return sample, target


def random_stain_color_change(
    sample: List[np.ndarray],
    target: np.ndarray,
    sigma: float,
    extract_tissue_fg_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Random color jitter in HED color space. The implementation is based on the description by Tellez et al. (2018).
    Additionally allows to only change the color of tissue foreground and not glass slide background by providing a
    function that masks out background.

    The used variable names (besides "mask") follow their definitions in Tellez et al. (2018). The paper used
    sigma = 0.05 by default for "light" augmentation and sigma = 0.2 for "strong" augmentation.
    """
    alpha = np.random.uniform(1 - sigma, 1 + sigma, (1, 3))
    beta = np.random.uniform(-sigma, sigma, (1, 3))

    changed_sample = []
    for P in sample:
        mask = (
            _create_soft_tissue_mask(P, extract_tissue_fg_fn)
            if extract_tissue_fg_fn is not None
            else None
        )

        P = np.moveaxis(P, 0, -1)  # Standard PyTorch axis order to skimage axis order.
        S = rgb2hed(P)
        S = np.reshape(S, (-1, 3))
        S_prime = alpha * S + beta
        P_prime = hed2rgb(S_prime)
        P_prime = np.reshape(P_prime, P.shape)
        if mask is not None:
            # Blend changed tissue and unchanged background. Soft mask smoothens borders.
            P_prime = P_prime * mask + P * (1 - mask)
        P_prime = np.moveaxis(P_prime, -1, 0)
        P_prime = P_prime.astype("float32")
        changed_sample.append(P_prime)

    return changed_sample, target


def _create_soft_tissue_mask(
    img: np.ndarray, extract_tissue_fg_fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    mask = extract_tissue_fg_fn(img)
    mask = gaussian(img_as_float32(mask), 1)
    mask_min = mask.min()
    mask_range = mask.max() - mask_min
    if mask_range != 0.0:  # Prevent division by zero.
        mask = (mask - mask_min) / mask_range
    mask = np.expand_dims(mask, -1)
    return mask


def random_color_change(
    sample: List[torch.Tensor], target: torch.Tensor, offset: float
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Random color jitter in HSV color space.

    :param offset: Expected to lie in [0, 1].
    """
    hue_offset = random.uniform(-0.5 * offset, 0.5 * offset)
    sat_offset = random.uniform(1 - offset, 1 + offset)

    changed_sample = []
    for img in sample:
        img = TF.adjust_hue(img, hue_offset)
        img = TF.adjust_saturation(img, sat_offset)
        changed_sample.append(img)

    return changed_sample, target


class ResamplingMultiscaleDataset(Dataset):
    def __init__(
        self,
        delegate: Dataset,
        num_samples: int,
        resampling_seed: int,
        only_initial_resampling: bool = False,
    ):
        super().__init__()
        self._delegate = delegate
        self._num_samples = num_samples
        self._rng = np.random.default_rng(resampling_seed)

        self._sampled_indices = np.arange(len(self._delegate))  # Identity by default.
        self._do_resample = True
        self.resample()
        self._do_resample = not only_initial_resampling

    def resample(self):
        if self._do_resample:
            if self._num_samples != len(self._delegate):
                self._sampled_indices = self._rng.choice(
                    len(self._delegate), size=self._num_samples, replace=False
                )

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        return self._delegate[self._sampled_indices[idx]]


class ResampleDatasetCallback(LambdaCallback):
    def __init__(self, dataset: Dataset, stage: str):
        self._dataset = dataset
        self._stage = stage
        if stage == "training":
            super().__init__(on_train_epoch_start=self._resample_dataset)
        elif stage == "validation":
            super().__init__(on_validation_epoch_start=self._resample_dataset)
        else:
            raise ValueError(stage)

    def _resample_dataset(self, *args):
        ds = (
            self._dataset.datasets
            if hasattr(self._dataset, "datasets")  # ConcatDataset
            else [self._dataset]
        )
        num_resampled = 0
        for d in ds:
            if hasattr(d, "resample"):
                # ResamplingMultiscaleDataset, ResamplingStainTransferTrainingDataset, or
                # UnpairedStainTransferTrainingDataset
                d.resample()
                num_resampled += 1
        if num_resampled != 0:
            info(
                f"Resampled {num_resampled} data set(s) at the start of the {self._stage} epoch."
            )
