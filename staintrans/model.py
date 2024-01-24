"""
Implementation of a CycleGAN architecture for stain transfer.
"""
from logging import info
from math import isnan
from pprint import pformat
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from staintrans.discriminator import (
    Discriminator,
    DiscriminatorLoss,
    FeatureDiscriminator,
)
from staintrans.generator import GeneratorLoss, create_generator
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
        target_identity_loss: bool = True,
        source_identity_loss: bool = False,
        saliency_loss: bool = False,
        bottleneck_loss: bool = False,
        feature_loss: bool = False,
        cycle_lambda: float = 10.0,
        source_identity_loss_end_epoch: int = 20,
        saliency_rho: float = 0.5,
        bottleneck_lambda: float = 0.5,
        feature_lambda: float = 0.5,
        padding_mode: str = "zeros",
        generator: str = "vanilla",
        segmentation_model: Optional[LightningModule] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore="segmentation_model")
        info("Model hparams:\n" + pformat(self.hparams))

        weight_init = WeightInitializer(std=0.02)

        # Generators
        self.gen_A2B = weight_init(
            create_generator(generator, padding_mode=padding_mode)
        )
        self.gen_B2A = weight_init(
            create_generator(generator, padding_mode=padding_mode)
        )
        self.gen_loss = GeneratorLoss(
            target_identity_loss=target_identity_loss,
            source_identity_loss=source_identity_loss,
            saliency_loss=saliency_loss,
            bottleneck_loss=bottleneck_loss,
            feature_loss=feature_loss,
            cycle_lambda=cycle_lambda,
            source_identity_loss_end_epoch=source_identity_loss_end_epoch,
            saliency_rho=saliency_rho,
            bottleneck_lambda=bottleneck_lambda,
            feature_lambda=feature_lambda,
        )

        # Image discriminators
        self.dis_A = weight_init(Discriminator(padding_mode=padding_mode))
        self.dis_B = weight_init(Discriminator(padding_mode=padding_mode))

        # Feature discriminators
        if feature_loss:
            self.dis_feat_A = weight_init(FeatureDiscriminator())
            self.dis_feat_B = weight_init(FeatureDiscriminator())
        else:
            self.dis_feat_A = None
            self.dis_feat_B = None

        self.dis_loss = DiscriminatorLoss()

        self.fake_buffer_A = ReplayBuffer(capacity=50)
        self.fake_buffer_B = ReplayBuffer(capacity=50)

        self._A_real_feat = None
        self._B_real_feat = None

        self.intermediate_output_visualizer = IntermediateOutputVisualizer()

        self.segm_model = segmentation_model

        self.example_input_array = (
            torch.zeros(1, 3, 512, 512),
            torch.zeros(1, 3, 512, 512),
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
        If using a segmentation model for validation (and not for computing task-specific losses during training), move
        it to the CPU before training to maximize the available space on the GPU. If using it for task-specific losses,
        make sure it is in inference mode.
        """
        task_specific_loss = (
            self.hparams.saliency_loss
            or self.hparams.bottleneck_loss
            or self.hparams.feature_loss
        )
        if self.segm_model is not None:
            if task_specific_loss:
                self.segm_model.freeze()
            else:
                self.segm_model.cpu()
        elif self.segm_model is None and task_specific_loss:
            raise ValueError(
                "Cannot use task-specific loss(es) without a segmentation model."
            )

    def training_step(self, batch, _, optimizer_idx):
        A_real, B_real = batch["A"], batch["B"]

        if optimizer_idx == 0:
            # Generators:

            real_mask = batch.get("SM")
            (
                A_fake,
                B_fake,
                gen_total_loss,
                *_,
                A_real_feat,
                B_real_feat,
                A_fake_feat,
                B_fake_feat,
                _,
                _,
            ) = self._train_val_step_generators(
                A_real, B_real, real_mask, is_train=True
            )

            # Detach and store generated outputs for discrimination.
            self.fake_buffer_A.push(A_fake, A_fake_feat)
            self.fake_buffer_B.push(B_fake, B_fake_feat)

            self._set_real_feat(A_real_feat, B_real_feat)

            return gen_total_loss

        if optimizer_idx == 1:
            A_fake, A_fake_feat = self.fake_buffer_A.pop()
            B_fake, B_fake_feat = self.fake_buffer_B.pop()

            # Image Discriminators:

            dis_A_loss = self._train_step_discriminator(self.dis_A, A_real, A_fake, "A")
            dis_B_loss = self._train_step_discriminator(self.dis_B, B_real, B_fake, "B")
            dis_total_loss = dis_A_loss + dis_B_loss

            # Feature discriminators:

            if self.hparams.feature_loss:
                A_real_feat, B_real_feat = self._get_and_clear_real_feat(A_real, B_real)

                dis_feat_A_loss = self._train_step_discriminator(
                    self.dis_feat_A, A_real_feat, A_fake_feat, "feat_A"
                )
                dis_feat_B_loss = self._train_step_discriminator(
                    self.dis_feat_B, B_real_feat, B_fake_feat, "feat_B"
                )
                dis_total_loss += dis_feat_A_loss + dis_feat_B_loss

            return dis_total_loss

    def on_validation_epoch_start(self) -> None:
        """
        See on_train_epoch_start. Make sure that the segmentation model is moved back to the GPU. (Maybe this is already
        done by Lightning.)
        """
        task_specific_loss = (
            self.hparams.saliency_loss
            or self.hparams.bottleneck_loss
            or self.hparams.feature_loss
        )
        if self.segm_model is not None and not task_specific_loss:
            self.segm_model.to(self.device)

    def validation_step(self, batch, batch_idx, **__):
        A_real, B_real = batch["A"], batch["B"]

        # Generators:

        real_mask = batch.get("SM")
        (
            A_fake,
            B_fake,
            _,
            A_dis_of_fake,
            B_dis_of_fake,
            A_cycle,
            B_cycle,
            A_real_feat,
            B_real_feat,
            *_,
            A_dis_of_fake_feat,
            B_dis_of_fake_feat,
        ) = self._train_val_step_generators(A_real, B_real, real_mask, is_train=False)

        # Image discriminators:

        A_dis_of_real = self.dis_A(A_real)
        B_dis_of_real = self.dis_B(B_real)
        dis_A_loss = self.dis_loss(A_dis_of_real, A_dis_of_fake)
        dis_B_loss = self.dis_loss(B_dis_of_real, B_dis_of_fake)

        self.log_dict(
            {"dis_A_loss_val": dis_A_loss, "dis_B_loss_val": dis_B_loss},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=A_real.size(0),
        )

        # Feature discriminators:

        if self.hparams.feature_loss:
            if A_real_feat is None:
                # HACK: Rescale from [-1, 1] to [0, 1] which is the expected input domain of the segmentation network.
                A_real_feat, _ = self.segm_model(
                    A_real / 2 + 0.5, only_extract_features=True
                )
            if B_real_feat is None:
                # HACK: Rescale from [-1, 1] to [0, 1] which is the expected input domain of the segmentation network.
                B_real_feat, _ = self.segm_model(
                    B_real / 2 + 0.5, only_extract_features=True
                )

            A_dis_of_real_feat = self.dis_feat_A(A_real_feat)
            B_dis_of_real_feat = self.dis_feat_B(B_real_feat)
            dis_feat_A_loss = self.dis_loss(A_dis_of_real_feat, A_dis_of_fake_feat)
            dis_feat_B_loss = self.dis_loss(B_dis_of_real_feat, B_dis_of_fake_feat)

            self.log_dict(
                {
                    "dis_feat_A_loss_val": dis_feat_A_loss,
                    "dis_feat_B_loss_val": dis_feat_B_loss,
                },
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
            B_dis_of_fake,
            A_cycle,
            # B->A
            batch["B_paths"],
            B_real,
            A_fake,
            A_dis_of_fake,
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

    def _train_val_step_generators(self, A_real, B_real, real_mask, is_train: bool):
        B_fake, A_fake = self(A_real, B_real)

        # Adversarial
        self.dis_A.requires_grad_(requires_grad=False)
        self.dis_B.requires_grad_(requires_grad=False)
        A_dis_of_fake = self.dis_A(A_fake)
        B_dis_of_fake = self.dis_B(B_fake)

        # Cycle consistency
        A_cycle, B_cycle = self._step_cycle(A_fake, B_fake)

        # Target identity
        A_identity, B_identity = self._step_identity(A_real, B_real)

        # Saliency
        A_fake_mask, B_fake_mask = self._step_saliency(A_fake, B_fake)

        # Segmentation bottleneck and encoder feature losses
        (
            A_real_bottleneck,
            B_real_bottleneck,
            A_fake_bottleneck,
            B_fake_bottleneck,
            A_real_feat,
            B_real_feat,
            A_fake_feat,
            B_fake_feat,
            A_dis_of_fake_feat,
            B_dis_of_fake_feat,
        ) = self._step_features(A_real, B_real, A_fake, B_fake)

        (
            gen_total_loss,
            gen_gan_loss_A2B,
            gen_gan_loss_B2A,
            gen_cycle_loss_A2A,
            gen_cycle_loss_B2B,
            gen_target_identity_loss_B_A2B,
            gen_target_identity_loss_A_B2A,
            gen_source_identity_loss_A_A2B,
            gen_source_identity_loss_B_B2A,
            gen_saliency_loss_A2B,
            gen_saliency_loss_B2A,
            gen_bottleneck_loss_A2B,
            gen_bottleneck_loss_B2A,
            gen_feature_loss_A2B,
            gen_feature_loss_B2A,
        ) = self.gen_loss(
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
            real_mask,
            real_mask,
            A_fake_mask,
            B_fake_mask,
            A_real_bottleneck,
            B_real_bottleneck,
            A_fake_bottleneck,
            B_fake_bottleneck,
            A_dis_of_fake_feat,
            B_dis_of_fake_feat,
            self.current_epoch,
        )

        self._log_generator_losses(
            gen_total_loss,
            gen_gan_loss_A2B,
            gen_gan_loss_B2A,
            gen_cycle_loss_A2A,
            gen_cycle_loss_B2B,
            gen_target_identity_loss_B_A2B,
            gen_target_identity_loss_A_B2A,
            gen_source_identity_loss_A_A2B,
            gen_source_identity_loss_B_B2A,
            gen_saliency_loss_A2B,
            gen_saliency_loss_B2A,
            gen_bottleneck_loss_A2B,
            gen_bottleneck_loss_B2A,
            gen_feature_loss_A2B,
            gen_feature_loss_B2A,
            is_train=is_train,
            batch_size=A_real.size(0),
        )

        return (
            A_fake,
            B_fake,
            gen_total_loss,
            A_dis_of_fake,
            B_dis_of_fake,
            A_cycle,
            B_cycle,
            A_real_feat,
            B_real_feat,
            A_fake_feat,
            B_fake_feat,
            A_dis_of_fake_feat,
            B_dis_of_fake_feat,
        )

    def _step_cycle(self, A_fake, B_fake):
        A_cycle = self.gen_B2A(B_fake)
        B_cycle = self.gen_A2B(A_fake)
        return A_cycle, B_cycle

    def _step_identity(self, A_real, B_real):
        if self.hparams.target_identity_loss:
            A_identity = self.gen_B2A(A_real)
            B_identity = self.gen_A2B(B_real)
        else:
            A_identity = None
            B_identity = None
        return A_identity, B_identity

    def _step_saliency(self, A_fake, B_fake):
        if self.hparams.saliency_loss:
            # HACK: Rescale from [-1, 1] to [0, 1] which is the expected input domain of the segmentation network.
            A_fake_mask = self.segm_model(A_fake / 2 + 0.5)
            # TODO: The original paper probably specified pixel values in range [0, 255], so factor 100 is likely not
            # ideal for our range of [0, 1]. Increase factor to e.g. 100 * 255? Or simply use a binary cross-entropy
            # loss on the raw predictions as we do it for training the segmentation model?
            A_fake_mask = torch.sigmoid(100 * (A_fake_mask - 0.5))
            B_fake_mask = self.segm_model(B_fake / 2 + 0.5)
            B_fake_mask = torch.sigmoid(100 * (B_fake_mask - 0.5))
        else:
            A_fake_mask = None
            B_fake_mask = None
        return A_fake_mask, B_fake_mask

    def _step_features(self, A_real, B_real, A_fake, B_fake):
        if self.hparams.bottleneck_loss or self.hparams.feature_loss:
            # HACK: Rescale from [-1, 1] to [0, 1] which is the expected input domain of the segmentation network.
            A_fake_feat, A_fake_bottleneck = self.segm_model(
                A_fake / 2 + 0.5, only_extract_features=True
            )
            B_fake_feat, B_fake_bottleneck = self.segm_model(
                B_fake / 2 + 0.5, only_extract_features=True
            )

        if self.hparams.bottleneck_loss:
            # HACK: Rescale from [-1, 1] to [0, 1] which is the expected input domain of the segmentation network.
            A_real_feat, A_real_bottleneck = self.segm_model(
                A_real / 2 + 0.5, only_extract_features=True
            )
            B_real_feat, B_real_bottleneck = self.segm_model(
                B_real / 2 + 0.5, only_extract_features=True
            )
        else:
            A_real_bottleneck = None
            B_real_bottleneck = None
            A_fake_bottleneck = None
            B_fake_bottleneck = None
            A_real_feat = None
            B_real_feat = None
        if self.hparams.feature_loss:
            self.dis_feat_A.requires_grad_(requires_grad=False)
            self.dis_feat_B.requires_grad_(requires_grad=False)
            A_dis_of_fake_feat = self.dis_feat_A(A_fake_feat)
            B_dis_of_fake_feat = self.dis_feat_B(B_fake_feat)
        else:
            A_fake_feat = None
            B_fake_feat = None
            A_dis_of_fake_feat = None
            B_dis_of_fake_feat = None

        return (
            A_real_bottleneck,
            B_real_bottleneck,
            A_fake_bottleneck,
            B_fake_bottleneck,
            A_real_feat,
            B_real_feat,
            A_fake_feat,
            B_fake_feat,
            A_dis_of_fake_feat,
            B_dis_of_fake_feat,
        )

    def _log_generator_losses(
        self,
        gen_total_loss,
        gen_gan_loss_A2B,
        gen_gan_loss_B2A,
        gen_cycle_loss_A2A,
        gen_cycle_loss_B2B,
        gen_target_identity_loss_B_A2B,
        gen_target_identity_loss_A_B2A,
        gen_source_identity_loss_A_A2B,
        gen_source_identity_loss_B_B2A,
        gen_saliency_loss_A2B,
        gen_saliency_loss_B2A,
        gen_bottleneck_loss_A2B,
        gen_bottleneck_loss_B2A,
        gen_feature_loss_A2B,
        gen_feature_loss_B2A,
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
            "gen_source_identity_loss_A_A2B": gen_source_identity_loss_A_A2B,
            "gen_source_identity_loss_B_B2A": gen_source_identity_loss_B_B2A,
            "gen_saliency_loss_A2B": gen_saliency_loss_A2B,
            "gen_saliency_loss_B2A": gen_saliency_loss_B2A,
            "gen_bottleneck_loss_A2B": gen_bottleneck_loss_A2B,
            "gen_bottleneck_loss_B2A": gen_bottleneck_loss_B2A,
            "gen_feature_loss_A2B": gen_feature_loss_A2B,
            "gen_feature_loss_B2A": gen_feature_loss_B2A,
        }
        for k, v in losses.items():
            if v is not None:
                self.log(
                    k if is_train else k + "_val",
                    v,
                    prog_bar=False,
                    logger=True,
                    on_step=is_train,
                    on_epoch=True,
                    batch_size=batch_size,
                )

    def _train_step_discriminator(self, dis, real, fake, name: str):
        dis.requires_grad_(requires_grad=True)
        dis_of_real = dis(real)
        dis_of_fake = dis(fake)
        dis_loss = self.dis_loss(dis_of_real, dis_of_fake)
        self.log(
            f"dis_{name}_loss",
            dis_loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=dis_of_real.size(0),
        )
        return dis_loss

    def _set_real_feat(self, A_real_feat, B_real_feat):
        self._A_real_feat = A_real_feat
        self._B_real_feat = B_real_feat

    def _get_and_clear_real_feat(self, A_real, B_real):
        A_real_feat = self._A_real_feat
        self._A_real_feat = None
        if A_real_feat is None:
            # HACK: Rescale from [-1, 1] to [0, 1] which is the expected input domain of the segmentation network.
            A_real_feat, _ = self.segm_model(
                A_real / 2 + 0.5, only_extract_features=True
            )

        B_real_feat = self._B_real_feat
        self._B_real_feat = None
        if B_real_feat is None:
            # HACK: Rescale from [-1, 1] to [0, 1] which is the expected input domain of the segmentation network.
            B_real_feat, _ = self.segm_model(
                B_real / 2 + 0.5, only_extract_features=True
            )

        return A_real_feat, B_real_feat

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
        if self.hparams.feature_loss:
            dis_feat_A_params = list(self.dis_feat_A.parameters())
            dis_feat_B_params = list(self.dis_feat_B.parameters())
            dis_params += dis_feat_A_params + dis_feat_B_params
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

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hack to remove saved weights of segmentation model when loading the overall CycleGAN for e.g. inference.
        This hook was added for older models for which on_save_checkpoint below was not yet implemented when their
        checkpoints were saved.
        """
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("segm_model."):
                del state_dict[k]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hack to keep weights of segmentation model from being saved.
        """
        if self.segm_model is not None:
            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                if k.startswith("segm_model."):
                    del state_dict[k]
