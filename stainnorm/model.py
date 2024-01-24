"""
Implementation of the stain color normalization U-Net from Tellez et al. (2019). The implementation was done from
scratch following the specifications given in the paper.

D. Tellez, G. Litjens, P. Bándi, W. Bulten, J.-M. Bokhorst, F. Ciompi, and J. Van Der Laak. Quantifying the effects of
data augmentation and stain color normalization in convolutional neural networks for computational pathology. Medical
image analysis, 58:101544, 2019.
"""
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F

# I would actually prefer to use valid convolutions, but given that stain normalization is an (optional) preprocessing
# step, this would make plugging the model into the overall pipeline cumbersome as it would reduce the downstream tile
# size.
_padding = "same"


class StainNormalizationNet(LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                DownBlock(3, 32),
                DownBlock(32, 64),
                DownBlock(64, 128),
                DownBlock(128, 256),
            ]
        )

        self.bottleneck = nn.Sequential(
            ConvBlock(
                256,
                512,
                kernel_size=3,
                stride=2,
                padding=_padding,
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBlock(
                    512,
                    256,
                    kernel_size=3,
                    stride=1,
                    padding=_padding,
                ),
            ),
        )

        self.decoder = nn.ModuleList(
            [
                UpBlock(256 + 256, 128),
                UpBlock(128 + 128, 64),
                UpBlock(64 + 64, 32),
                UpBlock(32 + 32, 3, last=True),
            ]
        )

    def forward(self, x):
        # HACK: [0, 1] --> [-1, 1]
        x = (x - 0.5) * 2

        from_skip = []

        for down in self.encoder:
            x, to_skip = down(x)
            from_skip.append(to_skip)

        x = self.bottleneck(x)

        for up, skip in zip(self.decoder, reversed(from_skip)):
            x = up(x, skip)

        # HACK: [-1, 1] --> [0, 1]
        x = x / 2 + 0.5
        return x

    def training_step(self, batch, *args, **kwargs):
        loss = self._step_with_target(batch)
        self.log("loss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        loss = self._step_with_target(batch)
        self.log(
            "val_loss", loss, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )
        return loss

    def _step_with_target(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def predict_step(self, batch, *args, **kwargs):
        # Filter out targets if present.
        if not isinstance(batch, torch.Tensor):
            batch = batch[0]
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, min_lr=1e-5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=_padding,
        )

    def forward(self, x):
        x = self.down(x)
        return x, x


class UpBlock(nn.Module):
    def __init__(self, combined_in_channels, out_channels, last=False):
        super().__init__()
        layers = []
        layers.append(nn.Upsample(scale_factor=2))
        if not last:
            conv = ConvBlock(
                combined_in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=_padding,
            )
        else:
            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=combined_in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=_padding,
                ),
                nn.Tanh(),
            )
        layers.append(conv)
        self.up = nn.Sequential(*layers)

    def forward(self, from_below, from_skip):
        x = torch.cat((from_below, from_skip), dim=1)
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
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, *x):
        return self.conv(*x)
