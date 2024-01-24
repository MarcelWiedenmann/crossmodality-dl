"""
Implementations of U-Net based generators. Also, a generator loss implementation that allows to combine different
auxiliary loss terms.
"""
from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch import nn


class Generator(LightningModule):
    """
    Standard U-Net architecture with some modifications such as using upsamling instead of transposed convolutions in
    the upward path, instance normalization, leaky ReLU, and "same" padding in the convolutions. Largely corresponds to
    the "unet_256" generator from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.

    The original CycleGAN implementation stems from this paper:

    J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation using cycle-consistent
    adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 2223–2232, 2017.

    See LICENSES/CYCLEGAN_LICENSE for the license of the original CycleGAN implementation.
    """

    def __init__(self, padding_mode: str):
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                Generator.DownBlock(3, 64, padding_mode, use_norm=False),
                Generator.DownBlock(64, 128, padding_mode),
                Generator.DownBlock(128, 256, padding_mode),
                Generator.DownBlock(256, 512, padding_mode),
                Generator.DownBlock(512, 512, padding_mode),
                Generator.DownBlock(512, 512, padding_mode),
                Generator.DownBlock(512, 512, padding_mode),
            ]
        )
        self.bottleneck = nn.Sequential(
            ConvBlock(
                512,
                512,
                nn.ReLU(inplace=True),
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
                use_norm=False,
            ),
            nn.Upsample(scale_factor=2),
            ConvBlock(
                512,
                512,
                nn.ReLU(inplace=True),
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
                use_norm=True,
            ),
        )
        self.decoder = nn.ModuleList(
            [
                Generator.UpBlock(512, 512, 512, padding_mode),
                Generator.UpBlock(512, 512, 512, padding_mode),
                Generator.UpBlock(512, 512, 512, padding_mode),
                Generator.UpBlock(512, 512, 256, padding_mode),
                Generator.UpBlock(256, 256, 128, padding_mode),
                Generator.UpBlock(128, 128, 64, padding_mode),
                Generator.UpBlock(64, 64, 32, padding_mode),
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

        self.example_input_array = torch.zeros(1, 3, 256, 256)

    def forward(self, x):
        from_skip = []

        for down in self.encoder:
            x, to_skip = down(x)
            from_skip.append(to_skip)

        x = self.bottleneck(x)

        for up, skip in zip(self.decoder, reversed(from_skip)):
            x = up(x, skip)

        return self.output(x)

    class DownBlock(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            padding_mode,
            use_norm=True,
        ):
            super().__init__()
            self.down = nn.Sequential(
                ConvBlock(
                    in_channels,
                    out_channels,
                    nn.LeakyReLU(0.2, inplace=True),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    padding_mode=padding_mode,
                    use_norm=use_norm,
                ),
            )

        def forward(self, x):
            x = self.down(x)
            return x, x

    class UpBlock(nn.Module):
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
                    from_below_channels + from_skip_channels,
                    out_channels,
                    nn.ReLU(inplace=True),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode=padding_mode,
                    use_norm=True,
                ),
            )

        def forward(self, from_below, from_skip):
            x = torch.cat((from_below, from_skip), dim=1)
            return self.up(x)


class GeneratorDeBel2019(LightningModule):
    """
    Generator architecture as proposed by De Bel et al. (2019):

    T. de Bel, M. Hermsen, J. Kers, J. van der Laak, and G. Litjens. Stain-transforming cycle-consistent generative
    adversarial networks for improved segmentation of renal histopathology. In Proceedings of the 2nd International
    Conference on Medical Imaging with Deep Learning, Proceedings of Machine Learning Research, pages 151–163, 2019.
    """

    def __init__(self, padding_mode: str = "zeros"):
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                GeneratorDeBel2019.DownBlock(3, 32, padding_mode, use_norm=False),
                GeneratorDeBel2019.DownBlock(32, 64, padding_mode),
                GeneratorDeBel2019.DownBlock(64, 128, padding_mode),
            ]
        )
        self.bottleneck = self.res = GeneratorDeBel2019.ResBlock(
            128,
            256,
            padding_mode,
            use_norm=False,
        )
        self.decoder = nn.ModuleList(
            [
                GeneratorDeBel2019.UpBlock(256, 128, 128, padding_mode),
                GeneratorDeBel2019.UpBlock(128, 64, 64, padding_mode),
                GeneratorDeBel2019.UpBlock(64, 32, 32, padding_mode),
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

        self.example_input_array = torch.zeros(1, 3, 512, 512)

    def forward(self, x):
        from_skip = []

        for down in self.encoder:
            x, to_skip = down(x)
            from_skip.append(to_skip)

        x = self.bottleneck(x)

        for up, skip in zip(self.decoder, reversed(from_skip)):
            x = up(x, skip)

        return self.output(x)

    class DownBlock(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            padding_mode,
            use_norm=True,
        ):
            super().__init__()
            self.res = GeneratorDeBel2019.ResBlock(
                in_channels,
                out_channels,
                padding_mode,
                use_norm,
            )
            self.down = nn.MaxPool2d(kernel_size=2)

        def forward(self, x):
            x = self.res(x)
            return self.down(x), x

    class UpBlock(nn.Module):
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
            self.res = GeneratorDeBel2019.ResBlock(
                from_below_channels // 2 + from_skip_channels,
                out_channels,
                padding_mode,
                use_norm=True,
            )

        def forward(self, from_below, from_skip):
            from_below = self.up(from_below)
            x = torch.cat((from_below, from_skip), dim=1)
            return self.res(x)

    class ResBlock(nn.Module):
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


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        kernel_size,
        stride,
        padding,
        padding_mode,
        use_norm,
    ):
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
            )
        )
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        if activation is not None:
            layers.append(activation)
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


def create_generator(generator: str, padding_mode):
    if generator == "vanilla":
        create_fn = Generator
    elif generator == "debel":
        create_fn = GeneratorDeBel2019
    else:
        raise ValueError(generator)

    return create_fn(padding_mode)


class GeneratorLoss:
    def __init__(
        self,
        target_identity_loss: bool,
        source_identity_loss: bool,
        saliency_loss: bool,
        bottleneck_loss: bool,
        feature_loss: bool,
        cycle_lambda: float,
        source_identity_loss_end_epoch: Optional[int] = None,
        saliency_rho: float = 0.5,
        bottleneck_lambda: float = 0.5,
        feature_lambda: float = 0.5,
    ):
        assert 0 < cycle_lambda
        assert not source_identity_loss or (
            source_identity_loss_end_epoch is None or 0 < source_identity_loss_end_epoch
        )
        assert not saliency_loss or 0 < saliency_rho
        assert not bottleneck_loss or 0 < bottleneck_lambda
        assert not feature_loss or 0 < feature_lambda

        self._gan_loss = nn.MSELoss()

        self._cycle_loss = nn.L1Loss()
        self._cycle_lambda = cycle_lambda

        self._use_target_identity_loss = target_identity_loss
        self._target_identity_loss = nn.L1Loss()
        self._target_identity_lambda = self._cycle_lambda * 0.5

        self._use_source_identity_loss = source_identity_loss
        self._source_identity_loss = nn.L1Loss()
        self._source_identity_lambda = self._cycle_lambda * 0.5
        self._source_identity_lambda_decay = (
            self._source_identity_lambda / source_identity_loss_end_epoch
            if source_identity_loss_end_epoch is not None
            else 0
        )

        self._use_saliency_loss = saliency_loss
        self._saliency_loss = nn.L1Loss()
        self._saliency_rho = saliency_rho

        self._use_bottleneck_loss = bottleneck_loss
        self._bottleneck_loss = nn.L1Loss()
        self._bottleneck_lambda = bottleneck_lambda

        self._use_feature_loss = feature_loss
        self._feature_loss = nn.MSELoss()
        self._feature_lambda = feature_lambda

    def __call__(
        self,
        A_real,
        B_real,
        A_fake,
        B_fake,
        A_dis_of_fake,
        B_dis_of_fake,
        A_cycle,
        B_cycle,
        A_identity,
        B_identity,
        A_real_mask,
        B_real_mask,
        A_fake_mask,
        B_fake_mask,
        A_real_bottleneck,
        B_real_bottleneck,
        A_fake_bottleneck,
        B_fake_bottleneck,
        A_dis_of_fake_feat,
        B_dis_of_fake_feat,
        epoch,
    ):
        # Adversarial loss
        gan_loss_A2B = self._compute_gan_loss(B_dis_of_fake)
        gan_loss_B2A = self._compute_gan_loss(A_dis_of_fake)

        # Cycle consistency loss
        cycle_loss_A2A = self._compute_cycle_loss(A_cycle, A_real)
        cycle_loss_B2B = self._compute_cycle_loss(B_cycle, B_real)
        cycle_loss_total = cycle_loss_A2A + cycle_loss_B2B

        total_loss_A2B = gan_loss_A2B + cycle_loss_total
        total_loss_B2A = gan_loss_B2A + cycle_loss_total

        # Optional auxiliary losses

        if self._use_target_identity_loss:
            # Identity loss from Zhu et al. (2017)
            target_identity_loss_B_A2B = self._compute_target_identity_loss(
                B_identity, B_real
            )
            target_identity_loss_A_B2A = self._compute_target_identity_loss(
                A_identity, A_real
            )
            total_loss_A2B += target_identity_loss_B_A2B
            total_loss_B2A += target_identity_loss_A_B2A
        else:
            target_identity_loss_B_A2B = None
            target_identity_loss_A_B2A = None

        if self._use_source_identity_loss:
            # Identity loss from De Bel et al. (2019).
            source_identity_loss_A_A2B = self._compute_source_identity_loss(
                B_fake, A_real, epoch
            )
            source_identity_loss_B_B2A = self._compute_source_identity_loss(
                A_fake, B_real, epoch
            )
            total_loss_A2B += source_identity_loss_A_A2B
            total_loss_B2A += source_identity_loss_B_B2A
        else:
            source_identity_loss_A_A2B = None
            source_identity_loss_B_B2A = None

        if self._use_saliency_loss:
            # Saliency loss from Li et al. (2021)
            saliency_loss_A2B = self._compute_saliency_loss(B_fake_mask, A_real_mask)
            saliency_loss_B2A = self._compute_saliency_loss(A_fake_mask, B_real_mask)
            total_loss_A2B += saliency_loss_A2B
            total_loss_B2A += saliency_loss_B2A
        else:
            saliency_loss_A2B = None
            saliency_loss_B2A = None

        if self._use_bottleneck_loss:
            # Task-specific feature loss that encourages aligned feature maps at the bottleneck of the segmentation
            # model for weakly paired data, where each pair consists of a generated tile and a real tile of the same
            # domain (e.g. a terminal H&E tile generated from a serial H&E tile and a real terminal H&E tile that is
            # weakly paired with the serial H&E tile).
            bottleneck_loss_A2B = self._compute_bottleneck_loss(
                B_fake_bottleneck, B_real_bottleneck
            )
            bottleneck_loss_B2A = self._compute_bottleneck_loss(
                A_fake_bottleneck, A_real_bottleneck
            )
            total_loss_A2B += bottleneck_loss_A2B
            total_loss_B2A += bottleneck_loss_B2A
        else:
            bottleneck_loss_A2B = None
            bottleneck_loss_B2A = None

        if self._use_feature_loss:
            # Task-specific feature loss that penalizes the generator if its generated image is mapped to features in
            # the encoder of the segmentation model that a feature-level discriminator considers fake.
            feature_loss_A2B = self._compute_feature_loss(B_dis_of_fake_feat)
            feature_loss_B2A = self._compute_feature_loss(A_dis_of_fake_feat)
            total_loss_A2B += feature_loss_A2B
            total_loss_B2A += feature_loss_B2A
        else:
            feature_loss_A2B = None
            feature_loss_B2A = None

        total_loss = total_loss_A2B + total_loss_B2A - cycle_loss_total

        return (
            total_loss,
            gan_loss_A2B,
            gan_loss_B2A,
            cycle_loss_A2A,
            cycle_loss_B2B,
            target_identity_loss_B_A2B,
            target_identity_loss_A_B2A,
            source_identity_loss_A_A2B,
            source_identity_loss_B_B2A,
            saliency_loss_A2B,
            saliency_loss_B2A,
            bottleneck_loss_A2B,
            bottleneck_loss_B2A,
            feature_loss_A2B,
            feature_loss_B2A,
        )

    def _compute_gan_loss(self, dis_of_fake):
        """
        Adversarial loss: Fake images should be considered real by the discriminator.
        """

        target_dis_of_fake = torch.ones_like(dis_of_fake, requires_grad=False)
        return self._gan_loss(dis_of_fake, target_dis_of_fake)

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

    def _compute_source_identity_loss(self, target_fake, source_real, epoch):
        """
        Identity loss: Applying A->B to A should result in something similar to A (same for B->A and B).
        """
        source_identity_loss = self._source_identity_loss(target_fake, source_real)
        source_identity_lambda = max(
            0, self._source_identity_lambda - self._source_identity_lambda_decay * epoch
        )
        return source_identity_lambda * source_identity_loss

    def _compute_saliency_loss(self, mask_fake, mask_real):
        """
        Saliency loss: Applying A->B to A should result in something whose foreground-background structure is similar to
        A's structure (same for B->A and B).

        The idea of a saliency loss is adopted from:

        X. Li, G. Zhang, H. Qiao, F. Bao, Y. Deng, J. Wu, Y. He, J. Yun, X. Lin, H. Xie, et al. Unsupervised content-
        preserving transformation for optical microscopy. Light: Science & Applications, 10(1):1–11, 2021.
        """
        saliency_loss = self._saliency_loss(mask_fake, mask_real)
        # TODO: Support for exponential decay of saliency rho.
        return self._saliency_rho * saliency_loss

    def _compute_bottleneck_loss(self, bottleneck_fake, bottleneck_real):
        bottleneck_loss = self._bottleneck_loss(bottleneck_fake, bottleneck_real)
        return self._bottleneck_lambda * bottleneck_loss

    def _compute_feature_loss(self, dis_of_fake_feat):
        target_dis_of_fake_feat = torch.ones_like(dis_of_fake_feat, requires_grad=False)
        feature_loss = self._feature_loss(dis_of_fake_feat, target_dis_of_fake_feat)
        return self._feature_lambda * feature_loss
