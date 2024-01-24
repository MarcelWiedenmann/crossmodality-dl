from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from torch import Tensor, nn
from torch.utils.tensorboard.writer import SummaryWriter


class WeightInitializer:
    def __init__(self, std: float, truncate: bool = False):
        self._std = std
        self._init_normal = nn.init.trunc_normal_ if truncate else nn.init.normal_

    def __call__(self, module: nn.Module):
        module.apply(self._initialize_weights)
        return module

    def _initialize_weights(self, module: nn.Module):
        class_name = module.__class__.__name__
        if hasattr(module, "weight"):
            if class_name.find("Conv") != -1 or class_name.find("Linear") != -1:
                self._init_normal(module.weight.data, mean=0, std=self._std)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias.data, val=0)
            if class_name.find("BatchNorm2d") != -1:
                self._init_normal(module.weight.data, mean=1.0, std=self._std)
                nn.init.constant_(module.bias.data, val=0)


class ReplayBuffer:
    """
    Used to update the discriminator based on a history of generated images instead of only the latest generated one.
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._buffer: List[Tuple[Tensor, Optional[List[Tensor]]]] = []
        self._samples = None

    def push(self, images: Tensor, features: Optional[List[Tensor]]):
        """
        Adds newly generated images (and their feature maps, if using the task-specific feature loss) until the buffer's
        capacity has been reached. From then on, randomly replaces older entries at 50% chance per newly generated
        entry.
        """
        samples = []
        for i in range(len(images)):
            entry = self._make_entry(images, features, i)

            if len(self._buffer) < self._capacity:
                self._buffer.append(self._clone_entry(entry))
                samples.append(self._clone_entry(entry))
            else:
                if np.random.uniform(0, 1) > 0.5:
                    idx_to_replace = np.random.randint(0, self._capacity)
                    replaced = self._buffer[idx_to_replace]
                    self._buffer[idx_to_replace] = self._clone_entry(entry)
                    samples.append(replaced)
                else:
                    samples.append(self._clone_entry(entry))
        self._samples = self._make_samples(samples)

    def _make_entry(self, images, features, i):
        image = torch.unsqueeze(images[i].detach(), 0)
        fms = (
            [torch.unsqueeze(fm[i].detach(), 0) for fm in features]
            if features is not None
            else None
        )
        return image, fms

    def _clone_entry(self, entry):
        image, fms = entry
        return image.clone(), [fm.clone() for fm in fms] if fms is not None else None

    def _make_samples(self, samples: List[Tuple[Tensor, Optional[List[Tensor]]]]):
        images, fms = list(zip(*samples))
        images = torch.cat(images)
        features = [torch.cat(fm) for fm in zip(*fms)] if fms[0] is not None else None
        return images, features

    def pop(self) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """
        Returns as many images from the buffer as the last push operation added.
        """
        samples = self._samples
        self._samples = None
        return samples


class IntermediateOutputVisualizer(Callback):

    _num_samples = 3

    def __init__(self):
        super().__init__()
        self._num_train_batches = None
        self._train_batch_size = None
        self._num_val_batches = None
        self._val_batch_size = None

        self._stage = None
        self._logger = None
        self._train_indices = None
        self._val_indices = None
        self._current_epoch = None

        # A->B
        self._A_real_paths = None
        self._A_real = None
        self._B_fake = None
        self._B_discrimination_of_fake = None
        self._A_cycle = None
        # B->A
        self._B_real_paths = None
        self._B_real = None
        self._A_fake = None
        self._A_discrimination_of_fake = None
        self._B_cycle = None

    def set_batch_info(
        self, num_train_batches, train_batch_size, num_val_batches, val_batch_size
    ):
        self._num_train_batches = num_train_batches
        self._train_batch_size = train_batch_size
        self._num_val_batches = num_val_batches
        self._val_batch_size = val_batch_size

        # Initialize fixed indices that stay the same over all epochs to visualize development of predictions.
        # Subtract by one since the last batch might be truncated, which may cause the sample index to exceed bounds
        # otherwise.
        self._train_indices = {
            np.random.randint(num_train_batches - 1): np.random.randint(
                train_batch_size
            )
        }
        self._val_indices = {
            np.random.randint(num_val_batches - 1): np.random.randint(val_batch_size)
        }

    def on_train_epoch_start(self, _, pl_module):
        self._prepare_epoch("train", pl_module)

    def on_validation_epoch_start(self, _, pl_module):
        self._prepare_epoch("val", pl_module)

    def _prepare_epoch(self, stage, model):
        self._stage = stage

        if model.logger is None:
            # We are in a dry run.
            self._logger = None
        elif isinstance(model.logger.experiment, list):
            self._logger = next(
                filter(lambda e: isinstance(e, SummaryWriter), model.logger.experiment)
            )
        else:
            self._logger = model.logger.experiment

        if stage == "train":
            indices = self._train_indices
            num_batches = self._num_train_batches
            batch_size = self._train_batch_size
        else:
            indices = self._val_indices
            num_batches = self._num_val_batches
            batch_size = self._val_batch_size

        fixed_indices = next(iter(indices.items()))

        # Make sure the fixed batch index is not used a second time, to simplify the bookkeeping further below.
        # Subtract by one since the last batch might be truncated, which may cause the sample index to exceed bounds
        # otherwise.
        vacant_batch_indices = list(range(num_batches - 1))
        vacant_batch_indices.remove(fixed_indices[0])
        new_batch_indices = np.random.choice(
            vacant_batch_indices, self._num_samples - 1, replace=False
        )
        new_sample_indices = np.random.choice(batch_size, self._num_samples - 1)
        indices = dict(
            [fixed_indices] + list(zip(new_batch_indices, new_sample_indices))
        )

        if stage == "train":
            self._train_indices = indices
        else:
            self._val_indices = indices
        self._current_epoch = model.current_epoch

        # A->B
        self._A_real_paths = [None] * self._num_samples
        self._A_real = [None] * self._num_samples
        self._B_fake = [None] * self._num_samples
        self._B_discrimination_of_fake = [None] * self._num_samples
        self._A_cycle = [None] * self._num_samples
        # B->A
        self._B_real_paths = [None] * self._num_samples
        self._B_real = [None] * self._num_samples
        self._A_fake = [None] * self._num_samples
        self._A_discrimination_of_fake = [None] * self._num_samples
        self._B_cycle = [None] * self._num_samples

    def visualize_intermediate_output(
        self,
        batch_idx,
        # A->B
        A_real_paths,
        A_real,
        B_fake,
        B_discrimination_of_fake,
        A_cycle,
        # B->A
        B_real_paths,
        B_real,
        A_fake,
        A_discrimination_of_fake,
        B_cycle,
    ):
        if self._stage == "train":
            indices = self._train_indices
        else:
            indices = self._val_indices

        if batch_idx not in indices.keys():
            return

        j = list(indices.keys()).index(batch_idx)
        sample_idx = indices[batch_idx]

        def from_tensor(t):
            t = t.detach().cpu().numpy()
            if t.ndim == 3:
                t = (t - np.min(t)) / np.ptp(t)
                t = np.moveaxis(t, 0, -1)
            return t

        # A->B
        self._A_real_paths[j] = A_real_paths[sample_idx]
        self._A_real[j] = from_tensor(A_real[sample_idx])
        self._B_fake[j] = from_tensor(B_fake[sample_idx])
        self._B_discrimination_of_fake[j] = from_tensor(
            B_discrimination_of_fake[sample_idx]
        )
        self._A_cycle[j] = from_tensor(A_cycle[sample_idx])
        # B->A
        self._B_real_paths[j] = B_real_paths[sample_idx]
        self._B_real[j] = from_tensor(B_real[sample_idx])
        self._A_fake[j] = from_tensor(A_fake[sample_idx])
        self._A_discrimination_of_fake[j] = from_tensor(
            A_discrimination_of_fake[sample_idx]
        )
        self._B_cycle[j] = from_tensor(B_cycle[sample_idx])

    def on_train_epoch_end(self, *_):
        self._log_output()

    def on_validation_epoch_end(self, *_):
        self._log_output()

    def _log_output(self):
        if all(p is None for p in self._A_real_paths):
            # No intermediate outputs added this epoch.
            return

        fig = plt.figure(layout="tight", figsize=(15, 29), dpi=75)
        subfigs = fig.subfigures(nrows=2 * self._num_samples)
        for i in range(self._num_samples):
            # A->B
            self._plot_output(
                subfigs[2 * i],
                "A",
                "B",
                self._A_real_paths[i],
                self._A_real[i],
                self._B_fake[i],
                self._B_discrimination_of_fake[i],
                self._A_cycle[i],
            )
            # B->A
            self._plot_output(
                subfigs[2 * i + 1],
                "B",
                "A",
                self._B_real_paths[i],
                self._B_real[i],
                self._A_fake[i],
                self._A_discrimination_of_fake[i],
                self._B_cycle[i],
            )

        # Logger is not available when in a dry run.
        if self._logger is not None:
            self._logger.add_figure(f"{self._stage}_outputs", fig, self._current_epoch)

    @staticmethod
    def _plot_output(
        subfig,
        source,
        target,
        real_path,
        real,
        fake,
        discrimination_of_fake,
        cycle,
    ):
        subfig.suptitle(Path(real_path).name)
        axs = subfig.subplots(ncols=4)

        axs[0].imshow(real)
        axs[0].set_axis_off()
        axs[0].set_title(f"Real {source}")
        axs[1].imshow(fake)
        axs[1].set_axis_off()
        axs[1].set_title(f"Fake {target}")
        axs[2].imshow(discrimination_of_fake, cmap="RdBu")
        axs[2].set_axis_off()
        axs[2].set_title(f"Discrimination of {target}")
        axs[3].imshow(cycle)
        axs[3].set_axis_off()
        axs[3].set_title(f"Restored {source}")
