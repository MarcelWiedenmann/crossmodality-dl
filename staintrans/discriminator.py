"""
Implementation of the PatchGAN discriminator as proposed by Isola et al. (2017) and as parameterized by Zhu et al.
(2017). Also, implementation of a feature-level discriminator based on PatchGAN, and a standard LSGAN discriminator
loss.

P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial networks. In
Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134, 2017.

J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial
networks. In Proceedings of the IEEE international conference on computer vision, pages 2223–2232, 2017.
"""
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F


class Discriminator(LightningModule):
    def __init__(self, padding_mode: str = "zeros"):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode
            ),
            nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(64, 128, padding_mode=padding_mode),
            ConvBlock(128, 256, padding_mode=padding_mode),
            ConvBlock(256, 512, stride=1, padding_mode=padding_mode),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, padding_mode=padding_mode),
        )

        self.example_input_array = torch.zeros(1, 3, 512, 512)

    def forward(self, x):
        return self.layers(x)


class FeatureDiscriminator(LightningModule):
    def __init__(self):
        super().__init__()
        self.reduce_fms = nn.ModuleList(
            [
                FeatureDiscriminator.Conv1x1Block(32, 2),
                FeatureDiscriminator.Conv1x1Block(64, 2),
                FeatureDiscriminator.Conv1x1Block(128, 4),
                FeatureDiscriminator.Conv1x1Block(256, 8),
                FeatureDiscriminator.Conv1x1Block(512, 16),
            ]
        )
        self.patch_gan = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.Conv2d(256, 1, kernel_size=4, padding=1),
        )

    def forward(self, fms):
        common_fm_size = fms[0].shape[-2:]
        aligned_fms = []
        for fm, reduce_fm in zip(fms, self.reduce_fms):
            reduced_fm = reduce_fm(fm)
            aligned_fm = F.interpolate(
                reduced_fm, size=common_fm_size, mode="bilinear", align_corners=True
            )
            aligned_fms.append(aligned_fm)
        aligned_fms = torch.cat(aligned_fms, dim=1)
        return self.patch_gan(aligned_fms)

    class Conv1x1Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )

        def forward(self, x):
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        padding_mode="zeros",
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DiscriminatorLoss:
    def __init__(self):
        self.loss = nn.MSELoss()

    def __call__(self, dis_of_real, dis_of_fake):
        """
        Adversarial loss: real and fake images should be detected as such, respectively.
        """
        target_dis_of_real = torch.ones_like(dis_of_real, requires_grad=False)
        target_dis_of_fake = torch.zeros_like(dis_of_fake, requires_grad=False)

        loss_real = self.loss(dis_of_real, target_dis_of_real)
        loss_fake = self.loss(dis_of_fake, target_dis_of_fake)
        return (loss_real + loss_fake) * 0.5
