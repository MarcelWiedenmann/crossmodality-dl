"""
Implementation of StainGAN proposed by Shaban et al. (2019). Serves as baseline for evaluation of the developed stain
transfer model. The implementation was done from scratch following the reference implementation provided by the authors
at https://github.com/xtarx/StainGAN (which in turn is mostly copied from the original CycleGAN reference implementation
at https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and the specificiations in the paper.

The original StainGAN implementation stems from this paper:

M. T. Shaban, C. Baur, N. Navab, and S. Albarqouni. Staingan: Stain style transfer for digital histological images. In
2019 Ieee 16th international symposium on biomedical imaging (Isbi 2019), pages 953–956. IEEE, 2019.

The original CycleGAN implementation stems from this paper:

J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial
networks. In Proceedings of the IEEE international conference on computer vision, pages 2223–2232, 2017.

See LICENSES/CYCLEGAN_LICENSE for the license of the original CycleGAN implementation.
"""
from logging import info
from math import isnan
from pprint import pformat
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from staintrans.discriminator import Discriminator, DiscriminatorLoss
from staintrans.utils import (
    IntermediateOutputVisualizer,
    ReplayBuffer,
    WeightInitializer,
)
from utils import compute_metrics, confusion_matrix


class CycleGan(LightningModule):
    def __init__(
        self,
        *,
        # pylint: disable=unused-argument
        # These are accessed via self.hparams.
        gen_lr: float = 2e-4,
        dis_lr: float = 2e-4,
        beta_1: float = 0.5,
        beta_2: float = 0.999,
        lr_decay_start_epoch: int = 100,
        lr_decay_end_epoch: int = 200,
        cycle_lambda: float = 10.0,
        segmentation_model: Optional[LightningModule] = None,
    ):
        """
        Default values correspond to the specifications in the original paper and its reference implementation. (In
        particular, learning rate decay is not used by default since the total number of epochs is less than 100.)
        """
        super().__init__()

        self.save_hyperparameters(ignore="segmentation_model")
        info("Model hparams:\n" + pformat(self.hparams))

        weight_init = WeightInitializer(std=0.02)

        # Generators
        self.gen_A2B = weight_init(_Generator(num_res_blocks=9))
        self.gen_B2A = weight_init(_Generator(num_res_blocks=9))
        self.gen_loss = _GeneratorLoss(cycle_lambda)

        # Discriminators
        self.dis_A = weight_init(Discriminator())
        self.dis_B = weight_init(Discriminator())
        self.dis_loss = DiscriminatorLoss()

        self.fake_buffer_A = ReplayBuffer(capacity=50)
        self.fake_buffer_B = ReplayBuffer(capacity=50)

        self.intermediate_output_visualizer = IntermediateOutputVisualizer()

        self.segm_model = segmentation_model

        self.example_input_array = (
            torch.zeros(1, 3, 256, 256),
            torch.zeros(1, 3, 256, 256),
        )

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        """
        Declutter TensorBoard logs by moving gradient norms to a subdirectory. Also, logging only once per epoch
        is sufficient.
        """
        grad_norm_dict = {"gradients/" + k: v for (k, v) in grad_norm_dict.items()}
        self.log_dict(
            grad_norm_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

    def forward(
        self,
        *real: Tuple[torch.Tensor, ...],
        direction: Optional[Literal["A2B", "B2A"]] = None,
    ):
        """
        Supports three modes:

        1. real consists of two tensors, the first of which is from domain A and the other from domain B: transfers the
           first tensor from A to B and the second tensor from B to A
        2. real is a single tensor and direction is a string "A2B": transfers tensor from A to B
        3. real is a single tensor and direction is a string "B2A": transfers tensor from B to A
        """
        if len(real) == 2:
            A_real, B_real = real
            return self.gen_A2B(A_real), self.gen_B2A(B_real)
        elif len(real) == 1:
            real = real[0]
            if direction == "A2B":
                return self.gen_A2B(real)
            elif direction == "B2A":
                return self.gen_B2A(real)
            else:
                raise ValueError(direction)
        else:
            raise ValueError(len(real))

    def on_train_epoch_start(self) -> None:
        """
        Move the segmentation model used for validation to the CPU before training to maximize the available space on
        the GPU.
        """
        if self.segm_model is not None:
            self.segm_model.cpu()

    def training_step(self, batch, _, optimizer_idx):
        A_real, B_real = batch["A"], batch["B"]

        if optimizer_idx == 0:
            # Generators:

            A_fake, B_fake, gen_total_loss, *_ = self._train_val_step_generators(
                A_real, B_real, is_train=True
            )

            # Detach and store generated outputs for discrimination.
            self.fake_buffer_A.push(A_fake)
            self.fake_buffer_B.push(B_fake)

            return gen_total_loss

        if optimizer_idx == 1:
            # Discriminators:

            # A (i.e. B->A)
            A_fake = self.fake_buffer_A.pop()
            dis_A_loss = self._train_step_discriminator(self.dis_A, A_real, A_fake, "A")

            # B (i.e. A->B)
            B_fake = self.fake_buffer_B.pop()
            dis_B_loss = self._train_step_discriminator(self.dis_B, B_real, B_fake, "B")

            return dis_A_loss + dis_B_loss

    def on_validation_epoch_start(self) -> None:
        """
        See on_train_epoch_start. Make sure that the segmentation model is moved back to the GPU. (Maybe this is already
        done by Lightning.)
        """
        if self.segm_model is not None:
            self.segm_model.to(self.device)

    def validation_step(self, batch, batch_idx, **__):
        A_real, B_real = batch["A"], batch["B"]

        # Generators:

        (
            A_fake,
            B_fake,
            _,
            A_discrimination_of_fake,
            B_discrimination_of_fake,
            A_cycle,
            B_cycle,
        ) = self._train_val_step_generators(A_real, B_real, is_train=False)

        # Discriminators:

        A_discrimination_of_real = self.dis_A(A_real)
        B_discrimination_of_real = self.dis_B(B_real)
        dis_A_loss = self.dis_loss(A_discrimination_of_real, A_discrimination_of_fake)
        dis_B_loss = self.dis_loss(B_discrimination_of_real, B_discrimination_of_fake)

        self.log_dict(
            {"dis_A_loss_val": dis_A_loss, "dis_B_loss_val": dis_B_loss},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=A_real.size(0),
        )

        # Visualize some intermediate results
        self.intermediate_output_visualizer.visualize_intermediate_output(
            batch_idx,
            # A->B
            batch["A_paths"],
            A_real,
            B_fake,
            B_discrimination_of_fake,
            A_cycle,
            # B->A
            batch["B_paths"],
            B_real,
            A_fake,
            A_discrimination_of_fake,
            B_cycle,
        )

        # Segmentation metrics
        if self.segm_model is not None:
            segm_target = batch["SM"].cpu().numpy()
            segm_target = segm_target.squeeze().astype("bool")
            # HACK: Rescale from [-1, 1] to [0, 1] which is the expected input domain of the segmentation network.
            segm_pred = self.segm_model.predict_step(B_fake / 2 + 0.5)
            segm_pred = segm_pred.cpu().numpy().squeeze() > 0.5
            segm_metrics = compute_metrics(*confusion_matrix(segm_target, segm_pred))
            jaccard = segm_metrics["Jaccard"]
            sensitivity = segm_metrics["Sensitivity"]
            specificity = segm_metrics["Specificity"]
            if not (isnan(jaccard) or isnan(sensitivity) or isnan(specificity)):
                self.log_dict(
                    {
                        "segm_jaccard_A2B_val": jaccard,
                        "segm_sensitivity_A2B_val": sensitivity,
                        "segm_specificity_A2B_val": specificity,
                    },
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=A_real.size(0),
                )

    def _train_val_step_generators(self, A_real, B_real, is_train: bool):
        B_fake, A_fake = self(A_real, B_real)

        # Adversarial
        self.dis_A.requires_grad_(requires_grad=False)
        self.dis_B.requires_grad_(requires_grad=False)
        A_discrimination_of_fake = self.dis_A(A_fake)
        B_discrimination_of_fake = self.dis_B(B_fake)

        # Cycle consistency
        A_cycle, B_cycle = self._forward_cycle(A_fake, B_fake)

        # Target identity
        A_identity, B_identity = self._forward_identity(A_real, B_real)

        (
            gen_total_loss,
            gen_gan_loss_A2B,
            gen_gan_loss_B2A,
            gen_cycle_loss_A2A,
            gen_cycle_loss_B2B,
            gen_target_identity_loss_B_A2B,
            gen_target_identity_loss_A_B2A,
        ) = self.gen_loss(
            A_real,
            B_real,
            A_discrimination_of_fake,
            B_discrimination_of_fake,
            A_cycle,
            B_cycle,
            A_identity,
            B_identity,
        )

        self._log_generator_losses(
            gen_total_loss,
            gen_gan_loss_A2B,
            gen_gan_loss_B2A,
            gen_cycle_loss_A2A,
            gen_cycle_loss_B2B,
            gen_target_identity_loss_B_A2B,
            gen_target_identity_loss_A_B2A,
            is_train=is_train,
            batch_size=A_real.size(0),
        )

        return (
            A_fake,
            B_fake,
            gen_total_loss,
            A_discrimination_of_fake,
            B_discrimination_of_fake,
            A_cycle,
            B_cycle,
        )

    def _forward_cycle(self, A_fake, B_fake):
        A_cycle = self.gen_B2A(B_fake)
        B_cycle = self.gen_A2B(A_fake)
        return A_cycle, B_cycle

    def _forward_identity(self, A_real, B_real):
        A_identity = self.gen_B2A(A_real)
        B_identity = self.gen_A2B(B_real)
        return A_identity, B_identity

    def _log_generator_losses(
        self,
        gen_total_loss,
        gen_gan_loss_A2B,
        gen_gan_loss_B2A,
        gen_cycle_loss_A2A,
        gen_cycle_loss_B2B,
        gen_target_identity_loss_B_A2B,
        gen_target_identity_loss_A_B2A,
        is_train: bool,
        batch_size: int,
    ):
        losses = {
            "gen_total_loss": gen_total_loss,
            "gen_gan_loss_A2B": gen_gan_loss_A2B,
            "gen_gan_loss_B2A": gen_gan_loss_B2A,
            "gen_cycle_loss_A2A": gen_cycle_loss_A2A,
            "gen_cycle_loss_B2B": gen_cycle_loss_B2B,
            "gen_target_identity_loss_B_A2B": gen_target_identity_loss_B_A2B,
            "gen_target_identity_loss_A_B2A": gen_target_identity_loss_A_B2A,
        }
        for k, v in losses.items():
            if v is not None:
                self.log(
                    k if is_train else k + "_val",
                    v,
                    prog_bar=False,
                    logger=True,
                    on_step=is_train,
                    # TODO: Replace on_epoch by custom logging in training_epoch_end that uses the epoch number as key?
                    # Currently, the x axis in TensorBoard shows steps instead of epochs, which is rather unintuitive.
                    on_epoch=True,
                    batch_size=batch_size,
                )

    def _train_step_discriminator(self, dis, real, fake, domain: str):
        dis.requires_grad_(requires_grad=True)
        discrimination_of_real = dis(real)
        discrimination_of_fake = dis(fake)
        dis_loss = self.dis_loss(discrimination_of_real, discrimination_of_fake)
        self.log(
            f"dis_{domain}_loss",
            dis_loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=real.size(0),
        )
        return dis_loss

    def configure_optimizers(self):
        gen_A2B_params = list(self.gen_A2B.parameters())
        gen_B2A_params = list(self.gen_B2A.parameters())
        gen_params = gen_A2B_params + gen_B2A_params
        gen_optim = torch.optim.Adam(
            gen_params,
            lr=self.hparams.gen_lr,
            betas=(
                self.hparams.beta_1,
                self.hparams.beta_2,
            ),
        )

        dis_A_params = list(self.dis_A.parameters())
        dis_B_params = list(self.dis_B.parameters())
        dis_params = dis_A_params + dis_B_params
        dis_optim = torch.optim.Adam(
            dis_params,
            lr=self.hparams.dis_lr,
            betas=(
                self.hparams.beta_1,
                self.hparams.beta_2,
            ),
        )

        gen_lr_sched = {
            "scheduler": LambdaLR(gen_optim, lr_lambda=self._lr_lambda, verbose=True),
            "name": "gen_lr",
        }

        dis_lr_sched = {
            "scheduler": LambdaLR(dis_optim, lr_lambda=self._lr_lambda, verbose=True),
            "name": "dis_lr",
        }

        return [gen_optim, dis_optim], [gen_lr_sched, dis_lr_sched]

    def _lr_lambda(self, epoch):
        start_epoch = self.hparams.lr_decay_start_epoch
        end_epoch = self.hparams.lr_decay_end_epoch
        return 1.0 - max(0, epoch - start_epoch) / float(end_epoch - start_epoch + 1)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hack to keep weights of segmentation model from being saved.
        """
        if self.segm_model is not None:
            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                if k.startswith("segm_model."):
                    del state_dict[k]

    def load_state_dict(self, *args, **kwargs):
        """
        Hack to exclude segmentation model from loading process.
        """
        segm_model = self.segm_model
        self.segm_model = None
        result = super().load_state_dict(*args, **kwargs)
        self.segm_model = segm_model
        return result


class _Generator(nn.Module):
    """
    Implementation of the generator used by Shaban et al. (2019) who adapted the image transformation network from
    Johnson et al. (2016). The transposed convolutions in the decoder have been replaced by pairs of upsampling and
    convolution.
    """

    def __init__(self, num_res_blocks):
        super().__init__()
        layers = [
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        ]

        # Downsampling:

        layers += [
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        ]

        # Residuals blocks:

        for _ in range(num_res_blocks):
            layers.append(_Generator._ResBlock(256))

        # Upsampling:

        layers += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                256,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
        ]

        layers += [
            nn.Conv2d(
                64,
                3,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
            nn.Tanh(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    class _ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                ),
                nn.InstanceNorm2d(channels),
                nn.ReLU(True),
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                ),
                nn.InstanceNorm2d(channels),
            )

        def forward(self, x):
            return x + self.conv(x)


class _GeneratorLoss:
    def __init__(self, cycle_lambda):
        assert 0 < cycle_lambda

        self._gan_loss = nn.MSELoss()

        self._cycle_loss = nn.L1Loss()
        self._cycle_lambda = cycle_lambda

        self._target_identity_loss = nn.L1Loss()
        self._target_identity_lambda = self._cycle_lambda * 0.5

    def __call__(
        self,
        A_real,
        B_real,
        A_discrimination_of_fake,
        B_discrimination_of_fake,
        A_cycle,
        B_cycle,
        A_identity,
        B_identity,
    ):
        # Adversarial loss
        gan_loss_A2B = self._compute_gan_loss(B_discrimination_of_fake)
        gan_loss_B2A = self._compute_gan_loss(A_discrimination_of_fake)

        # Cycle consistency loss
        cycle_loss_A2A = self._compute_cycle_loss(A_cycle, A_real)
        cycle_loss_B2B = self._compute_cycle_loss(B_cycle, B_real)
        cycle_loss_total = cycle_loss_A2A + cycle_loss_B2B

        # Identity loss from Zhu et al. (2017)
        target_identity_loss_B_A2B = self._compute_target_identity_loss(
            B_identity, B_real
        )
        target_identity_loss_A_B2A = self._compute_target_identity_loss(
            A_identity, A_real
        )

        total_loss_A2B = gan_loss_A2B + cycle_loss_total + target_identity_loss_B_A2B
        total_loss_B2A = gan_loss_B2A + cycle_loss_total + target_identity_loss_A_B2A
        total_loss = total_loss_A2B + total_loss_B2A - cycle_loss_total

        return (
            total_loss,
            gan_loss_A2B,
            gan_loss_B2A,
            cycle_loss_A2A,
            cycle_loss_B2B,
            target_identity_loss_B_A2B,
            target_identity_loss_A_B2A,
        )

    def _compute_gan_loss(self, discrimination_of_fake):
        """
        Adversarial loss: Fake images should be considered real by the discriminator.
        """

        target_fake = torch.ones_like(discrimination_of_fake, requires_grad=False)
        return self._gan_loss(discrimination_of_fake, target_fake)

    def _compute_cycle_loss(self, after_cycle, real):
        """
        Cycle consistency loss: Applying A->B->A to A should result in something similar to A (same for B->A->B and B).
        """
        cycle_loss = self._cycle_loss(after_cycle, real)
        return self._cycle_lambda * cycle_loss

    def _compute_target_identity_loss(self, identity, target_real):
        """
        Identity loss: Applying B->A to A should result in something similar to A (same for A->B and B).
        """
        target_identity_loss = self._target_identity_loss(identity, target_real)
        return self._target_identity_lambda * target_identity_loss
