"""
Single-scale U-Net serving as baseline for evaluation of the multi-scale model. The implementation was modified from
https://github.com/usuyama/pytorch-unet/blob/a0ec1374e06ca9161c21781302328b074f19391b/pytorch_resnet18_unet.ipynb which
had also been used by Schmitz et al. (2021) as baseline.

See LICENSES/PYTORCHUNET_LICENSE for the license of the original file.
"""
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics import functional as MF
from torchvision import models


class ResNetUNet(LightningModule):
    def __init__(self):
        super().__init__()

        base_model = models.resnet18(pretrained=True)
        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = _convblock(64, 64, 1, 0)
        self.layer1 = nn.Sequential(
            *self.base_layers[3:5]
        )  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = _convblock(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = _convblock(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = _convblock(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = _convblock(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_up3 = _convblock(256 + 512, 512, 3, 1)
        self.conv_up2 = _convblock(128 + 512, 256, 3, 1)
        self.conv_up1 = _convblock(64 + 256, 256, 3, 1)
        self.conv_up0 = _convblock(64 + 256, 128, 3, 1)

        self.conv_original_size0 = _convblock(3, 64, 3, 1)
        self.conv_original_size1 = _convblock(64, 64, 3, 1)
        self.conv_original_size2 = _convblock(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, inp):
        x_original = self.conv_original_size0(inp)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(inp)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

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
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = MF.accuracy(y_hat, y.int(), threshold=0.0)
        return loss, acc

    def predict_step(self, batch, *args, **kwargs):
        # Filter out targets if present.
        if not isinstance(batch, torch.Tensor):
            batch = batch[0]
        return torch.sigmoid(self(batch))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.5, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


def _convblock(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
