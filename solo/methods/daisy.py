# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.daisy import conditional_entropy, multiview_conditional_entropy_loss
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseMethod
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.utils.misc import omegaconf_select
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import wandb
from torchmetrics.classification import MulticlassConfusionMatrix

class Daisy(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements Daisy method

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                proj_output_dim (int): number of dimensions of projected features.
                shared_dim (int): number of dimensions of the shared space.
                exclusive_dim (int): number of dimensions of the exclusive space.
        """
        super().__init__(cfg)

        # self.extender_output_dim: int = cfg.method_kwargs.extender_output_dim
        # self.extender_hidden_dim: int = cfg.method_kwargs.extender_output_dim
        self.proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        self.proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        self.shared_dim: int = cfg.method_kwargs.proj_output_dim // 2
        self.exclusive_dim: int = cfg.method_kwargs.proj_output_dim // 2

        self.conde_loss_weight: float = cfg.method_kwargs.conde_loss_weight
        self.exclusive_loss_weight: float = cfg.method_kwargs.exclusive_loss_weight
        self.invariance_loss_weight: float = cfg.method_kwargs.invariance_loss_weight

        self.kernel_type: str = cfg.method_kwargs.kernel_type
        self.alpha: float = cfg.method_kwargs.alpha

        # # extender
        # if cfg.method_kwargs.extender_type == "mlp":
        #     self.extender = nn.Sequential(
        #         nn.Linear(self.features_dim, self.extender_hidden_dim),
        #         nn.BatchNorm1d(self.extender_hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(self.extender_hidden_dim, self.extender_hidden_dim),
        #         nn.BatchNorm1d(self.extender_hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(self.extender_hidden_dim, self.extender_output_dim),
        #     )
        # elif cfg.method_kwargs.extender_type == "identity":
        #     self.extender_output_dim = self.features_dim
        #     self.shared_dim = self.features_dim//2
        #     self.exclusive_dim = self.features_dim//2
        #     self.extender = nn.Identity()

        # elif cfg.method_kwargs.extender_type == "linear":
        #     self.extender = nn.Sequential(
        #         nn.Linear(self.features_dim, self.extender_output_dim),
        #     )

        # projector
        self.projector = nn.Sequential(
           nn.Linear(self.features_dim, self.proj_hidden_dim),
           nn.BatchNorm1d(self.proj_hidden_dim),
           nn.ReLU(),
           nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim),
           nn.BatchNorm1d(self.proj_hidden_dim),
           nn.ReLU(),
           nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        )

        # augmentation predictor
        self.augmentation_regressor = nn.Sequential(
            nn.Linear(self.exclusive_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 13),
        )

        #classifier
        self.classifier_shared = nn.Linear(self.shared_dim, self.num_classes)
        self.classifier_exclusive = nn.Linear(self.exclusive_dim, self.num_classes)
        self.classifier_both = nn.Linear(self.proj_output_dim, self.num_classes)
        self.classifier_feats = nn.Linear(self.features_dim, self.num_classes)

        del self.classifier # remove the parent classifier that was instantiated in the superclass

        # for analysis of training outputs
        self.return_train_outputs = cfg.return_train_outputs

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(Daisy, Daisy).add_and_assert_specific_cfg(cfg)

        # assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.extender_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        # assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.extender_type")

        # conde arguments
        cfg.method_kwargs.kernel_type = omegaconf_select(cfg, "method_kwargs.kernel_type", "gaussian")
        cfg.method_kwargs.alpha = omegaconf_select(cfg, "method_kwargs.alpha", 1.0)

        # loss tradeoffs
        cfg.method_kwargs.conde_loss_weight = omegaconf_select(cfg, "method_kwargs.conde_loss_weight", 1.0)
        cfg.method_kwargs.exclusive_loss_weight = omegaconf_select(cfg, "method_kwargs.exclusive_loss_weight", 1.0)
        cfg.method_kwargs.invariance_loss_weight = omegaconf_select(cfg, "method_kwargs.invariance_loss_weight", 1.0)

        # # extender arguments
        # cfg.method_kwargs.extender_type = omegaconf_select(cfg, "method_kwargs.extender_type", "mlp")
        # cfg.method_kwargs.extender_output_dim = omegaconf_select(cfg, "method_kwargs.extender_output_dim", 1024)
        # cfg.method_kwargs.extender_hidden_dim = omegaconf_select(cfg, "method_kwargs.extender_hidden_dim", 512)
        # assert cfg.method_kwargs.extender_type in ["mlp", "identity", "linear"]

        # # projector arguments
        cfg.method_kwargs.projector_type = omegaconf_select(cfg, "method_kwargs.projector_type", "mlp")
        cfg.method_kwargs.proj_output_dim = omegaconf_select(cfg, "method_kwargs.proj_output_dim", 1024)
        cfg.method_kwargs.proj_hidden_dim = omegaconf_select(cfg, "method_kwargs.proj_hidden_dim", 2048)
        assert cfg.method_kwargs.projector_type in ["mlp", "linear", "identity"]

        cfg.return_train_outputs = omegaconf_select(cfg, "return_train_outputs", False)
        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds mlp parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        base_learnable_params = [{"name": "backbone", "params": self.backbone.parameters()},
                                 {"name": "augmentation_predictor", "params": self.augmentation_regressor.parameters()},
                                 {"name": "projector", "params": self.projector.parameters()}]
        
        classifier_learnable_params = [{"name": "classifier_shared", "params": self.classifier_shared.parameters(),  "lr": self.classifier_lr, "weight_decay": 0},
                                       {"name": "classifier_exclusive", "params": self.classifier_exclusive.parameters(),  "lr": self.classifier_lr, "weight_decay": 0},
                                       {"name": "classifier_both", "params": self.classifier_both.parameters(),  "lr": self.classifier_lr, "weight_decay": 0},
                                       {"name": "classifier_feats", "params": self.classifier_feats.parameters(),  "lr": self.classifier_lr, "weight_decay": 0}]

        return base_learnable_params + classifier_learnable_params

    def forward(self, X):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs
        """
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        X = X.cuda()
        feats = self.backbone(X)
        feats_logits = self.classifier_feats(feats)
        projections = self.projector(feats)

        shared_dims, exclusive_dims = torch.split(projections, [self.shared_dim, self.exclusive_dim], dim=1)
        augmentation_predictions = self.augmentation_regressor(exclusive_dims)

        shared_logits = self.classifier_shared(shared_dims.detach())
        exclusive_logits = self.classifier_exclusive(exclusive_dims.detach())
        both_logits = self.classifier_both(projections.detach())

        return {"shared_logits": shared_logits, 
                "exclusive_logits": exclusive_logits,
                "both_logits": both_logits,
                "feats_logits": feats_logits,
                "feats": feats,
                "shared_dims": shared_dims,
                "exclusive_dims": exclusive_dims,
                "augmentation_predictions": augmentation_predictions
                }
    
    # OVERRIDES SHARED_STEP IN THE PARENT CLASS
    def _base_shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        def update_specificed_online_classifier(logits, classifier_name: str) -> None:
            loss = F.cross_entropy(logits, targets, ignore_index=-1)

            top_k_max = min(5, logits.size(1))
            acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))

            loss_name = f"{classifier_name}_loss"
            acc1_name = f"{classifier_name}_acc1"
            acc5_name = f"{classifier_name}_acc5"
            out.update({loss_name: loss, acc1_name: acc1, acc5_name: acc5})

        out = self(X)
        targets = targets.cuda()
        
        update_specificed_online_classifier(out["shared_logits"], "class_shared")
        update_specificed_online_classifier(out["exclusive_logits"], "class_exclusive")
        update_specificed_online_classifier(out["both_logits"], "class_both")
        update_specificed_online_classifier(out["feats_logits"], "class_feats")
        
        return out
    
    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """
        def log_classifier_loss(outs, name: str) -> None:
            loss_name = f"{name}_loss"
            acc1_name = f"{name}_acc1"
            acc5_name = f"{name}_acc5"

            # average over loss and stats
            outs[loss_name] = sum(outs[loss_name]) / self.num_large_crops
            outs[acc1_name] = sum(outs[acc1_name]) / self.num_large_crops
            outs[acc5_name] = sum(outs[acc5_name]) / self.num_large_crops

            metrics = {
                f"train_{name}_loss": outs[loss_name],
                f"train_{name}_acc1": outs[acc1_name],
                f"train_{name}_acc5": outs[acc5_name],
            }

            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            return outs


        sample_indices, data_and_params, targets = batch
        X = [data_and_params[idx][0] for idx in range(self.num_crops)]
        params_targets = [data_and_params[idx][1] for idx in range(self.num_crops)]
        
        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.num_crops

        outs = [self.base_training_step(x, targets) for x in X[: self.num_large_crops]]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        outs, conde_loss, exclusive_loss, shared_loss = self.compute_ssl_losses(outs, params_targets)

        self.log("train_conde_loss", conde_loss, sync_dist=True)
        self.log("train_aug_mseloss", exclusive_loss,  sync_dist=True)
        self.log("train_invariance_mseloss", shared_loss, sync_dist=True)
        
        # CLASSIFIER LOSSES
        outs = log_classifier_loss(outs, "class_shared")
        outs = log_classifier_loss(outs, "class_exclusive")
        outs = log_classifier_loss(outs, "class_both")
        outs = log_classifier_loss(outs, "class_feats")

        class_loss = outs["class_shared_loss"] + outs["class_exclusive_loss"] + outs["class_both_loss"] + outs["class_feats_loss"]

        full_loss = class_loss + \
                    -(self.conde_loss_weight * conde_loss) + \
                    self.exclusive_loss_weight * exclusive_loss + \
                    self.invariance_loss_weight * shared_loss

        should_adjust_params = self.current_epoch > self.cfg.data.adjustable_dataloader.epochs_before_adjusting
        self.trainer.train_dataloader.dataset.update_augmentation_parameters(
            sample_indices.tolist(), 
            outs["augmentation_errors"].mean(dim=0).tolist(), 
            should_adjust=should_adjust_params
        )

        if self.return_train_outputs:
            return full_loss, outs
        else:
            return full_loss
    
    def compute_ssl_losses(self, outs: Dict[str, Any], params_targets: torch.Tensor):
        conde_loss = multiview_conditional_entropy_loss(
                        outs["shared_dims"], 
                        outs["exclusive_dims"], 
                        self.kernel_type, 
                        self.alpha
                    )

        shared_invariance_loss = self.multiview_shared_invariance_loss(outs["shared_dims"])
        
        exclusive_invariance_loss, diffs_per_batch, phats_per_batch = self.multiview_exclusive_invariance_loss(
                                        outs["exclusive_dims"],
                                        params_targets
                                    )
        
        outs.update({"augmentation_errors": torch.stack(diffs_per_batch)})
        outs.update({"phats": phats_per_batch})

        return outs, conde_loss, exclusive_invariance_loss, shared_invariance_loss

        
    def compute_exclusive_invariance_loss(self, e_i, e_j, p_i, p_j):
        e_diff = e_i - e_j
        p_diff_hat = self.augmentation_regressor(e_diff)

        p_i = torch.stack(p_i).T.half()
        p_j = torch.stack(p_j).T.half()
        p_diff = p_i - p_j
        p_diff = p_diff.cuda()

        diffs_per_batch = (p_diff_hat - p_diff).detach()

        exclusive_loss = F.l1_loss(p_diff_hat, p_diff)
        return exclusive_loss, diffs_per_batch, p_diff_hat.detach()


    def multiview_exclusive_invariance_loss(self, exclusive_dims, params_targets: torch.Tensor):
        total_exclusive_loss = 0
        diffs_per_batch = []
        phats_per_batch = []
        for i in range(len(exclusive_dims)):
            for j in range(i+1, len(exclusive_dims)):
                exclusive_loss, batch_diffs, p_diff_hat = self.compute_exclusive_invariance_loss(
                                            exclusive_dims[i],
                                            exclusive_dims[j],
                                            params_targets[i],
                                            params_targets[j]
                                        )
                diffs_per_batch.append(batch_diffs)
                phats_per_batch.append(p_diff_hat)
                total_exclusive_loss += exclusive_loss
        total_exclusive_loss /= len(exclusive_dims)
        
        return total_exclusive_loss, diffs_per_batch, phats_per_batch

    def multiview_shared_invariance_loss(self, shared_dims):
        average_embedding = torch.mean(torch.stack(shared_dims), dim=0)

        total_shared_loss = 0
        for i in range(len(shared_dims)):
            total_shared_loss += F.mse_loss(shared_dims[i], average_embedding)
        total_shared_loss /= len(shared_dims)

        return total_shared_loss


    def validation_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = None,
        update_validation_step_outputs: bool = True,
    ) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.
            update_validation_step_outputs (bool): whether or not to append the
                metrics to validation_step_outputs

        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """
        def log_classifier_loss(outs, name: str) -> None:
            loss_name = f"{name}_loss"
            acc1_name = f"{name}_acc1"
            acc5_name = f"{name}_acc5"

            metrics = {
                f"val_{name}_loss": outs[loss_name],
                f"val_{name}_acc1": outs[acc1_name],
                f"val_{name}_acc5": outs[acc5_name],
            }
            return metrics
        
        (X, params), targets = batch
        batch_size = targets.size(0)

        out = self.base_validation_step(X, targets)

        if self.knn_eval and not self.trainer.sanity_checking:
            self.knn(test_features=out.pop("feats").detach(), test_targets=targets.detach())

        # Update CLASSIFIER LOSSES
        metrics = log_classifier_loss(out, "class_shared")
        metrics.update(log_classifier_loss(out, "class_exclusive"))
        metrics.update(log_classifier_loss(out, "class_both"))
        metrics.update(log_classifier_loss(out, "class_feats"))
        metrics.update({"batch_size": batch_size})
        metrics.update({"shared_logits": out["shared_logits"]})
        metrics.update({"exclusive_logits": out["exclusive_logits"]})
        metrics.update({"feats_logits": out["feats_logits"]})
        metrics.update({"targets": targets})

        if update_validation_step_outputs:
            self.validation_step_outputs.append(metrics)
        return metrics
    
    def on_validation_epoch_end(self):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        """

        acc1s = []
        for loss_type in ["class_shared", "class_exclusive", "class_both", "class_feats"]:
            val_loss = weighted_mean(self.validation_step_outputs, f"val_{loss_type}_loss", "batch_size")
            val_acc1 = weighted_mean(self.validation_step_outputs, f"val_{loss_type}_acc1", "batch_size")
            val_acc5 = weighted_mean(self.validation_step_outputs, f"val_{loss_type}_acc5", "batch_size")

            log = {f"val_{loss_type}_loss": val_loss, f"val_{loss_type}_acc1": val_acc1, f"val_{loss_type}_acc5": val_acc5}
            self.log_dict(log, sync_dist=True)
            acc1s.append(val_acc1)

        # save current augmentation npy on last epoch
        if self.current_epoch == self.trainer.max_epochs - 1 or self.current_epoch == self.trainer.max_epochs:
            self.trainer.train_dataloader.dataset.save_augmentation_parameters()

        # log confusion matrices on last epoch
        if self.current_epoch == self.trainer.max_epochs - 1 or self.current_epoch == self.trainer.max_epochs:
            y_true = torch.tensor([t for out in self.validation_step_outputs for t in out["targets"]]).cuda().float()
            y_shared_pred = torch.concat([torch.argmax(out["shared_logits"], dim=1) for out in self.validation_step_outputs]).cuda().float()
            y_exclusive_pred = torch.concat([torch.argmax(out["exclusive_logits"], dim=1) for out in self.validation_step_outputs]).cuda().float()
            y_feats_pred = torch.concat([torch.argmax(out["feats_logits"], dim=1) for out in self.validation_step_outputs]).cuda().float()

            # shared
            conf_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize='true').to(y_true)
            conf_matrix.update(y_shared_pred, y_true)
            fig_shared, ax_ = conf_matrix.plot(labels=self.trainer.val_dataloaders.dataset.classes)

            #exclusive
            conf_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize='true').to(y_true)
            conf_matrix.update(y_exclusive_pred, y_true)
            fig_exclusive, ax_ = conf_matrix.plot(labels=self.trainer.val_dataloaders.dataset.classes)

            # feats
            conf_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize='true').to(y_true)
            conf_matrix.update(y_feats_pred, y_true)
            fig_feats, ax_ = conf_matrix.plot(labels=self.trainer.val_dataloaders.dataset.classes)

            self.logger.log_image("confusion_matrix", 
                                    images=[wandb.Image(fig_shared), wandb.Image(fig_exclusive), wandb.Image(fig_feats)], 
                                    caption=[f"Shared Confusion {acc1s[0]}", f"Exclusive Confusion {acc1s[1]}", f"Feats Confusion {acc1s[2]}"])

        self.validation_step_outputs.clear()