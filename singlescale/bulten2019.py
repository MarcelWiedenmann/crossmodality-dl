"""
Implementation of the six-level segmentation U-Net with residual blocks from Bulten et al. (2019). Serves as baseline
for evaluation of the multi-scale model. The implementation was done from scratch following the specifications given in
the paper.

W. Bulten, P. Bándi, J. Hoven, R. v. d. Loo, J. Lotz, N. Weiss, J. v. d. Laak, B. v. Ginneken, C. Hulsbergen-van de Kaa,
and G. Litjens. Epithelium segmentation using deep learning in h&e-stained prostate specimens with immunohistochemistry
as reference standard. Scientific reports, 9(1):1–10, 2019.
"""

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics import functional as MF


class ResUNet(LightningModule):
    def __init__(self, pixel_weights: bool = False):
        super().__init__()
        self._pixel_weights = pixel_weights

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Number of feature maps. Ordered from top level to bottom level.
        encoder_in_channels = [3, 32, 64, 128, 256]
        encoder_out_channels = [32, 64, 128, 256, 512]
        decoder_in_channels_from_below = [64, 128, 256, 512, 1024]
        decoder_out_channels = [32, 64, 128, 256, 512]

        for i, (in_channels, out_channels) in enumerate(
            zip(encoder_in_channels, encoder_out_channels)
        ):
            self.encoder.append(DownBlock(in_channels, out_channels))

        self.bottleneck = Bottleneck(
            encoder_out_channels[-1], decoder_in_channels_from_below[-1]
        )

        for i, (in_below, in_skip, out_channels) in enumerate(
            zip(
                decoder_in_channels_from_below,
                encoder_out_channels,
                decoder_out_channels,
            )
        ):
            self.decoder.append(UpBlock(in_below + in_skip, out_channels, last=i == 0))

        self.segment = nn.Conv2d(decoder_out_channels[0], 1, kernel_size=1)

        self.example_input_array = torch.zeros(1, 3, 512, 512)

    def forward(self, x, only_extract_features: bool = False):
        from_skip = []
        # Required for the task-specific feature losses when training the stain transfer model.
        encoder_features = []

        for down in self.encoder:
            x, to_skip = down(x)
            from_skip.append(to_skip)
            encoder_features.append(x)

        x, bottleneck_features = self.bottleneck(x)

        if only_extract_features:
            return encoder_features, bottleneck_features

        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](x, from_skip[i])

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
        x, y = batch
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
        # Filter out targets if present.
        if not isinstance(batch, torch.Tensor):
            batch = batch[0]
        return torch.sigmoid(self(batch))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, betas=(0.99, 0.99))
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=9, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels)
        self.down = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.res(x)
        return self.down(x), x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels)
        self.scale = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        x = self.res(x)
        return self.scale(x), x


class UpBlock(nn.Module):
    def __init__(self, combined_in_channels, out_channels, last):
        super().__init__()
        self.res = ResBlock(combined_in_channels, out_channels)
        if not last:
            self.scale = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            self.scale = None

    def forward(self, from_below, from_skip):
        x = torch.cat((from_below, from_skip), dim=1)
        x = self.res(x)
        if self.scale is not None:
            x = self.scale(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3),
            ConvBlock(out_channels, out_channels, kernel_size=3, act=False),
        )
        self.identity = ConvBlock(in_channels, out_channels, kernel_size=1, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        identity = self.identity(x)
        out += identity
        return self.relu(out)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=int((kernel_size - 1) / 2),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, *x):
        return self.conv(*x)
