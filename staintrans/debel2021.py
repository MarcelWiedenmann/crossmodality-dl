"""
Implementation of the residual CycleGAN proposed by De Bel et al. (2021). Serves as baseline for evaluation of the developed stain
transfer model. The implementation was done from scratch following the specifications in the paper.

T. de Bel, J.-M. Bokhorst, J. van der Laak, and G. Litjens. Residual cyclegan for robust domain transformation of
histopathological tissue slides. Medical Image Analysis, 70:102004, 2021.
"""
from logging import info
from math import isnan
from pprint import pformat
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import StepLR

from staintrans.discriminator import DiscriminatorLoss
from staintrans.generator import ConvBlock
from staintrans.utils import (
    ReplayBuffer,
    IntermediateOutputVisualizer,
    WeightInitializer,
)
from utils import compute_metrics, confusion_matrix


class CycleGan(LightningModule):
    def __init__(
        self,
        *,
        # pylint: disable=unused-argument
        # These are accessed via self.hparams.
        gen_lr: float = 1e-4,
        dis_lr: float = 1e-4,
        beta_1: float = 0.5,
        beta_2: float = 0.999,
        cycle_lambda: float = 5.0,
        cycle_mask_quantile: float = 0.1,
        generator: Literal["debel2019", "debel2021"] = "debel2021",
        segmentation_model: Optional[LightningModule] = None,
    ):
        """
        Default values correspond to the specifications in the original paper.
        """
        super().__init__()

        self.save_hyperparameters(ignore="segmentation_model")
        info("Model hparams:\n" + pformat(self.hparams))

        # Generators
        gen_weight_init = WeightInitializer(std=0.01, truncate=True)
        if generator == "debel2019":
            gen_create_fn = _GeneratorDeBel2019
        elif generator == "debel2021":
            gen_create_fn = _Generator
        else:
            raise ValueError(generator)
        self.gen_A2B = gen_weight_init(gen_create_fn())
        self.gen_B2A = gen_weight_init(gen_create_fn())
        self.gen_loss = _GeneratorLoss(
            cycle_lambda,
            cycle_mask=True,
            cycle_mask_quantile=cycle_mask_quantile,
        )

        # Discriminators
        dis_weight_init = WeightInitializer(std=0.02, truncate=True)
        self.dis_A = dis_weight_init(_Discriminator())
        self.dis_B = dis_weight_init(_Discriminator())
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
        first tensor from A to B and the second tensor from B to A. The returned tensors are _not_ clipped to a
        specific range. (Due to the residual nature of the generator, values may generally be outside the [-1, 1] range.)
        2. real is a single tensor and direction is a string "A2B": transfers tensor from A to B. The returned tensor is
        clipped to range [-1, 1].
        3. real is a single tensor and direction is a string "B2A": transfers tensor from B to A. The returned tensor is
        clipped to range [-1, 1].
        """
        if len(real) == 2:
            A_real, B_real = real
            return self.gen_A2B(A_real), self.gen_B2A(B_real)
        elif len(real) == 1:
            real = real[0]
            if direction == "A2B":
                fake = self.gen_A2B(real)
            elif direction == "B2A":
                fake = self.gen_B2A(real)
            else:
                raise ValueError(direction)
            return torch.clamp(fake, min=-1, max=1)
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
            # The residual setup of the generator can result in values exceeding range [-1, 1]; clip accordingly.
            B_fake = torch.clamp(B_fake, min=-1, max=1)
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

        (
            gen_total_loss,
            gen_gan_loss_A2B,
            gen_gan_loss_B2A,
            gen_cycle_loss_A2A,
            gen_cycle_loss_B2B,
        ) = self.gen_loss(
            A_real,
            B_real,
            A_discrimination_of_fake,
            B_discrimination_of_fake,
            A_cycle,
            B_cycle,
        )

        self._log_generator_losses(
            gen_total_loss,
            gen_gan_loss_A2B,
            gen_gan_loss_B2A,
            gen_cycle_loss_A2A,
            gen_cycle_loss_B2B,
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

    def _log_generator_losses(
        self,
        gen_total_loss,
        gen_gan_loss_A2B,
        gen_gan_loss_B2A,
        gen_cycle_loss_A2A,
        gen_cycle_loss_B2B,
        is_train: bool,
        batch_size: int,
    ):
        losses = {
            "gen_total_loss": gen_total_loss,
            "gen_gan_loss_A2B": gen_gan_loss_A2B,
            "gen_gan_loss_B2A": gen_gan_loss_B2A,
            "gen_cycle_loss_A2A": gen_cycle_loss_A2A,
            "gen_cycle_loss_B2B": gen_cycle_loss_B2B,
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
            "scheduler": StepLR(gen_optim, step_size=15, gamma=0.5, verbose=True),
            "name": "gen_lr",
        }

        dis_lr_sched = {
            "scheduler": StepLR(dis_optim, step_size=15, gamma=0.5, verbose=True),
            "name": "dis_lr",
        }

        return [gen_optim, dis_optim], [gen_lr_sched, dis_lr_sched]

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


class _Generator(LightningModule):
    """
    Implementation of the residual generator proposed by De Bel et al. (2021). The organization of the code follows
    Figure 6 in the paper. Note that the description of the final convolutional layer of the upwards path was somewhat
    inconsistent in the paper, as it was described as ending in leaky relu and instance norm at some locations and as
    ending in tanh at another one. We use tanh without normalization, which makes the most sense and follows widely
    adopted practices.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                _Generator._DownBlock(3, 32),
                _Generator._DownBlock(32, 64),
                _Generator._DownBlock(64, 128),
            ]
        )

        self.bottleneck = nn.Sequential(
            _Generator._ConvBlock(128, 128),
            nn.Upsample(scale_factor=2),
        )

        self.decoder = nn.ModuleList(
            [
                _Generator._UpBlock(128 + 128, 64),
                _Generator._UpBlock(64 + 64, 32),
                _Generator._UpBlock(32 + 32, 3, last=True),
            ]
        )

    def forward(self, x):
        identity = x
        from_skip = []

        for down in self.encoder:
            x, to_skip = down(x)
            from_skip.append(to_skip)

        x = self.bottleneck(x)

        for up, skip in zip(self.decoder, reversed(from_skip)):
            x = up(x, skip)

        # Make x the residual. Multiplying by factor two is necessary since both x and identity are in [-1, 1], and the
        # generator needs to be able to turn min. values from that range into max. values from that range and vice versa.
        return 2 * x + identity

    class _DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = _Generator._ConvBlock(in_channels, out_channels)

        def forward(self, x):
            x = self.conv(x)
            return F.interpolate(x, scale_factor=0.5, mode="bilinear"), x

    class _UpBlock(nn.Module):
        def __init__(self, in_channels, out_channels, last=False) -> None:
            super().__init__()
            self.conv = _Generator._ConvBlock(in_channels, out_channels, last=last)
            self.up = nn.Upsample(scale_factor=2) if not last else nn.Identity()

        def forward(self, from_below, from_skip):
            x = torch.cat((from_below, from_skip), dim=1)
            x = self.conv(x)
            return self.up(x)

    class _ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, last=False):
            super().__init__()
            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                ),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                ),
            ]
            if not last:
                layers += [
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            else:
                layers.append(nn.Tanh())
            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            return self.conv(x)


class _GeneratorDeBel2019(LightningModule):
    """
    Generator architecture from De Bel et al. (2019) but with the additional residual connection proposed by
    De Bel et al. (2021).
    """

    def __init__(self):
        super().__init__()

        padding_mode = "reflect"

        self.encoder = nn.ModuleList(
            [
                _GeneratorDeBel2019._DownBlock(3, 32, padding_mode, use_norm=False),
                _GeneratorDeBel2019._DownBlock(32, 64, padding_mode),
                _GeneratorDeBel2019._DownBlock(64, 128, padding_mode),
            ]
        )
        self.bottleneck = self.res = _GeneratorDeBel2019._ResBlock(
            128,
            256,
            padding_mode,
            use_norm=False,
        )
        self.decoder = nn.ModuleList(
            [
                _GeneratorDeBel2019._UpBlock(256, 128, 128, padding_mode),
                _GeneratorDeBel2019._UpBlock(128, 64, 64, padding_mode),
                _GeneratorDeBel2019._UpBlock(64, 32, 32, padding_mode),
            ]
        )
        self.output = ConvBlock(
            32,
            3,
            nn.Tanh(),
            kernel_size=1,
            stride=1,
            padding=0,
            padding_mode=padding_mode,
            use_norm=False,
        )

    def forward(self, x):
        identity = x
        from_skip = []

        for down in self.encoder:
            x, to_skip = down(x)
            from_skip.append(to_skip)

        x = self.bottleneck(x)

        for up, skip in zip(self.decoder, reversed(from_skip)):
            x = up(x, skip)

        x = self.output(x)

        # Make x the residual. Multiplying by factor two is necessary since both x and identity are in [-1, 1], and the
        # generator needs to be able to turn min. values from that range into max. values from that range and vice versa.
        return 2 * x + identity

    class _DownBlock(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            padding_mode,
            use_norm=True,
        ):
            super().__init__()
            self.res = _GeneratorDeBel2019._ResBlock(
                in_channels,
                out_channels,
                padding_mode,
                use_norm,
            )
            self.down = nn.MaxPool2d(kernel_size=2)

        def forward(self, x):
            x = self.res(x)
            return self.down(x), x

    class _UpBlock(nn.Module):
        def __init__(
            self,
            from_below_channels,
            from_skip_channels,
            out_channels,
            padding_mode,
        ):
            super().__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBlock(
                    from_below_channels,
                    from_below_channels // 2,
                    nn.LeakyReLU(0.2, inplace=True),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode=padding_mode,
                    use_norm=True,
                ),
            )
            self.res = _GeneratorDeBel2019._ResBlock(
                from_below_channels // 2 + from_skip_channels,
                out_channels,
                padding_mode,
                use_norm=True,
            )

        def forward(self, from_below, from_skip):
            from_below = self.up(from_below)
            x = torch.cat((from_below, from_skip), dim=1)
            return self.res(x)

    class _ResBlock(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            padding_mode,
            use_norm,
        ):
            super().__init__()
            self.conv = nn.Sequential(
                ConvBlock(
                    in_channels,
                    out_channels,
                    nn.LeakyReLU(0.2, inplace=True),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode=padding_mode,
                    use_norm=use_norm,
                ),
                ConvBlock(
                    out_channels,
                    out_channels,
                    None,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode=padding_mode,
                    use_norm=use_norm,
                ),
            )
            self.identity = ConvBlock(
                in_channels,
                out_channels,
                None,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode=padding_mode,
                use_norm=use_norm,
            )
            self.act = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            out = self.conv(x)
            identity = self.identity(x)
            out += identity
            return self.act(out)


class _GeneratorLoss:
    def __init__(
        self, cycle_lambda: float, cycle_mask: bool, cycle_mask_quantile: float
    ):
        assert 0 < cycle_lambda
        assert not cycle_mask or (0 < cycle_mask_quantile < 1)

        self._gan_loss = nn.MSELoss()
        self._cycle_loss = nn.L1Loss(reduction="none" if cycle_mask else "mean")
        self._cycle_lambda = cycle_lambda
        self._cycle_mask = cycle_mask
        self._cycle_mask_quantile = cycle_mask_quantile

    def __call__(
        self,
        A_real,
        B_real,
        A_discrimination_of_fake,
        B_discrimination_of_fake,
        A_cycle,
        B_cycle,
    ):
        # Adversarial loss
        gan_loss_A2B = self._compute_gan_loss(B_discrimination_of_fake)
        gan_loss_B2A = self._compute_gan_loss(A_discrimination_of_fake)

        # Cycle consistency loss
        cycle_loss_A2A = self._compute_cycle_loss(A_cycle, A_real)
        cycle_loss_B2B = self._compute_cycle_loss(B_cycle, B_real)
        cycle_loss_total = cycle_loss_A2A + cycle_loss_B2B

        total_loss_A2B = gan_loss_A2B + cycle_loss_total
        total_loss_B2A = gan_loss_B2A + cycle_loss_total
        total_loss = total_loss_A2B + total_loss_B2A - cycle_loss_total

        return (
            total_loss,
            gan_loss_A2B,
            gan_loss_B2A,
            cycle_loss_A2A,
            cycle_loss_B2B,
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

        The loss is masked to improve numerical stability. As per the original paper:

        The generators in the residual CycleGAN start off with performing a near-identity mapping, due to layer
        initialisation with low values and summation of input and output, which results in a near-zero cycle-loss. This
        caused the network to sometimes be numerically unstable, producing nans during back-propagation. To alleviate
        this problem, we averaged the cycle-consistency loss only over individual pixels with a loss in the
        90th-percentile.

        Implementation note: I am not entirely clear on the phrasing "[a loss] in the 90th-percentile". As far as I
        understand, "in a quantile" typically means at or below the value of the quantile, i.e. the smallest 90% of all
        pixel-wise losses in the present case. This can clearly not be meant. On the other hand, interpreting it as "at
        or above the 90th-percentile" seems excessive as this would discard 90% of all pixel-wise losses. Therefore we
        implement the masking such that the highest 90% of all pixel-wise losses are considered by default (see CycleGAN
        constructor above), i.e. we include values at or above the 10th percentile. We compute the percentile per image
        so that each image contributes equally to the loss, independent of the batch configuration.
        """
        cycle_loss = self._cycle_loss(after_cycle, real)
        if self._cycle_mask:
            # Percentile per image
            threshold = torch.quantile(
                cycle_loss.flatten(start_dim=1), self._cycle_mask_quantile, dim=1
            )
            threshold = threshold.reshape(threshold.shape[0], 1, 1, 1)
            # Masking
            cycle_loss = cycle_loss[cycle_loss >= threshold]
            # Reduction
            cycle_loss = cycle_loss.sum() / cycle_loss.numel()
        return self._cycle_lambda * cycle_loss


class _Discriminator(LightningModule):
    """
    Implementation of the discriminator proposed by De Bel et al. (2021), which is based on PatchGAN by
    Isola et al. (2017).
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            _Discriminator._ConvBlock(3, 64),
            _Discriminator._ConvBlock(64, 128),
            _Discriminator._ConvBlock(128, 256),
            _Discriminator._ConvBlock(256, 256),
            nn.Conv2d(256, 1, kernel_size=4, padding=1),
        )

    def forward(self, x):
        return self.layers(x)

    class _ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        def forward(self, x):
            return self.conv(x)
