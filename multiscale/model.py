"""
Modified implementation of the msY²-Net model from Schmitz et al. (2021). The implementation was done from scratch using
the implementation accompanying the paper (https://github.com/IPMI-ICNS-UKE/multiscale/) as reference to ensure the
soundness of the implementation (e.g. some hyperparameter values not specified in the paper were looked up there).

See LICENSES/SCHMITZ2021_LICENSE for the license of the original implementation by Schmitz et al. (2021).
"""
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics import functional as MF

from .resnet import resnet18


class MsY2Net(LightningModule):
    def __init__(self, pixel_weights: bool = False):
        super().__init__()
        self._pixel_weights = pixel_weights
        self.num_branches = 3

        self.full_encoder = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.mid_encoder = nn.ModuleList()
        self.low_encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Pre-trained ResNet18 layers for the model's encoders.
        full_layers = self._create_base_layers()
        mid_layers = self._create_base_layers()
        low_layers = self._create_base_layers()

        # Number of feature maps. Ordered from top level to bottom level.
        skip_channels = [64, 64, 64, 128, 256]
        decoder_in_channels = [128, 256, 256, 512, 512]
        decoder_out_channels = [64, 128, 256, 256, 512]

        # Encoders & skip
        for i, (full, skip, mid, low) in enumerate(
            zip(full_layers, skip_channels, mid_layers, low_layers)
        ):
            self.full_encoder.append(DownBlock(full, skip=True, first_block=i == 0))
            self.skip.append(SkipBlock(skip))
            self.mid_encoder.append(DownBlock(mid, skip=False))
            self.low_encoder.append(DownBlock(low, skip=False))

        # Multi-scale merge at bottleneck
        self.merge = MultiscaleMergeBlock(512, 16, context_scales=[4, 8])

        # Decoder
        for from_below, from_skip, to_above in zip(
            decoder_in_channels, skip_channels, decoder_out_channels
        ):
            self.decoder.append(UpBlock(from_below + from_skip, to_above))

        # Output
        self.segment = nn.Conv2d(
            decoder_out_channels[0], 1, kernel_size=1, stride=1, padding="same"
        )

    def _create_base_layers(self):
        base_layers = list(resnet18(pretrained=True).children())
        return (
            base_layers[:3],
            base_layers[3:5],
            [base_layers[5]],
            [base_layers[6]],
            [base_layers[7]],
        )

    def forward(self, x):
        x_full, x_mid, x_low = x

        from_skip = []

        # Encoders & skip
        for full, skip, mid, low in zip(
            self.full_encoder, self.skip, self.mid_encoder, self.low_encoder
        ):
            x_full, to_skip = full(x_full)
            from_skip.append(skip(to_skip))
            x_mid = mid(x_mid)
            x_low = low(x_low)

        # Multi-scale merge
        x = self.merge(x_full, x_mid, x_low)

        # Decoder
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](x, from_skip[i])

        # Output
        return self.segment(x)

    def training_step(self, batch, *args, **kwargs):
        loss, acc = self._step_with_target(batch)
        self.log_dict(
            {"loss": loss, "accuracy": acc},
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, *args, **kwargs):
        loss, acc = self._step_with_target(batch)
        self.log_dict(
            {"val_loss": loss, "val_accuracy": acc},
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _step_with_target(self, batch):
        x = batch[:-1]
        y = batch[-1]
        y_hat = self(x)

        if self._pixel_weights:
            # Weight each pixel based on its class's share in the sample.

            num_pix_per_sample = y.shape[1:].numel()
            num_pos_per_sample = y.sum(dim=(1, 2, 3))
            num_neg_per_sample = num_pix_per_sample - num_pos_per_sample

            pos_to_neg_per_sample = num_pos_per_sample / num_neg_per_sample
            # Uniform weights if no positive/negative pixels in sample.
            pos_to_neg_per_sample[
                (pos_to_neg_per_sample == float("inf")) | (pos_to_neg_per_sample == 0.0)
            ] = 1.0
            # Add channel, height, and width dimensions to allow broadcasting.
            pos_to_neg_per_sample = pos_to_neg_per_sample.view(-1, 1, 1, 1)

            neg_to_pos_per_sample = num_neg_per_sample / num_pos_per_sample
            # Uniform weights if no positive/negative pixels in sample.
            neg_to_pos_per_sample[
                (neg_to_pos_per_sample == float("inf")) | (neg_to_pos_per_sample == 0.0)
            ] = 1.0
            # Add channel, height, and width dimensions to allow broadcasting.
            neg_to_pos_per_sample = neg_to_pos_per_sample.view(-1, 1, 1, 1)

            weight_per_pixel = (
                y * neg_to_pos_per_sample + (1 - y) * pos_to_neg_per_sample
            )
        else:
            weight_per_pixel = None

        loss = F.binary_cross_entropy_with_logits(y_hat, y, weight=weight_per_pixel)
        acc = MF.accuracy(y_hat, y.int(), threshold=0.0)
        return loss, acc

    def predict_step(self, batch, *args, **kwargs):
        """
        Turn the logits into probabilities. This makes the predictions a little more intuitive to understand and
        threshold into their final classes. Alternatively, one could apply the logit function to the threshold value
        before thresholding.
        """
        batch = batch[: self.num_branches]  # Filter out targets if present.
        return torch.sigmoid(self(batch))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.5, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class DownBlock(nn.Module):
    def __init__(self, resnet_layers, skip, first_block=False):
        super().__init__()
        self.down = nn.Sequential(*resnet_layers)
        if skip:
            if first_block:
                self.skip = nn.Sequential(
                    ConvBlock(3, 64, kernel_size=3, stride=1, padding="same"),
                    ConvBlock(64, 64, kernel_size=3, stride=1, padding="same"),
                )
            else:
                self.skip = nn.Identity()
        else:
            self.skip = None

    def forward(self, x):
        for_below = self.down(x)
        if self.skip is not None:
            return for_below, self.skip(x)
        else:
            return for_below


class SkipBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.skip = ConvBlock(
            channels, channels, kernel_size=1, stride=1, padding="same"
        )

    def forward(self, x):
        return self.skip(x)


class MultiscaleMergeBlock(nn.Module):
    def __init__(
        self, channels_per_scale, full_resolution_spatial_size, context_scales
    ):
        super().__init__()
        if any(cs >= full_resolution_spatial_size for cs in context_scales):
            raise ValueError(
                "Support for very large magnification levels not implemented."
            )
        self.full_resolution_spatial_size = full_resolution_spatial_size
        self.context_scales = context_scales
        num_scales = 1 + len(context_scales)
        self.combine = ConvBlock(
            channels_per_scale * num_scales,
            channels_per_scale,
            kernel_size=1,
            stride=1,
            padding="same",
        )

    def forward(self, full_x, mid_x, low_x):
        mid_x = _crop(
            mid_x, self.full_resolution_spatial_size // self.context_scales[0]
        )
        mid_x = F.interpolate(
            mid_x, scale_factor=self.context_scales[0], mode="bilinear"
        )
        low_x = _crop(
            low_x, self.full_resolution_spatial_size // self.context_scales[1]
        )
        low_x = F.interpolate(
            low_x, scale_factor=self.context_scales[1], mode="bilinear"
        )
        x = torch.cat((full_x, mid_x, low_x), dim=1)
        return self.combine(x)


def _crop(x, size):
    assert x.shape[-2] == x.shape[-1]
    diff = x.shape[-1] - size
    crop_before = diff // 2
    crop_after = diff - crop_before
    crop_after = -crop_after if crop_after != 0 else None
    return x[..., crop_before:crop_after, crop_before:crop_after]


class UpBlock(nn.Module):
    def __init__(self, combined_in_channels, out_channels):
        super().__init__()
        self.scale = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up = ConvBlock(
            combined_in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

    def forward(self, from_below, from_skip):
        x = self.scale(from_below)
        x = torch.cat((x, from_skip), dim=1)
        return self.up(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        if padding == "same":
            padding = int((kernel_size - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, *x):
        return self.conv(*x)
