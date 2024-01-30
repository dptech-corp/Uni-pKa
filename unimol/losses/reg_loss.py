# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("finetune_mse")
class FinetuneMSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output_a, batch_a = model(
            sample["net_input_a"],
            classification_head_name=self.args.classification_head_name,
            features_only = True,
        )
        net_output_b, batch_b = model(
            sample["net_input_b"],
            classification_head_name=self.args.classification_head_name,
            features_only = True,
        )

        loss, predict = self.compute_loss(model, net_output_a, net_output_b, batch_a, batch_b, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=predict.device)
                targets_std = torch.tensor(self.task.std, device=predict.device)
                predict = predict * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": predict.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, self.args.num_classes)
                .data,
                "smi_name": sample["id"],
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output_a, net_output_b, batch_a, batch_b, sample, reduce=True):
        free_energy_a = net_output_a.view(-1, self.args.num_classes).float()
        free_energy_b = net_output_b.view(-1, self.args.num_classes).float()
        if not self.training:
            def compute_agg_free_energy(free_energy, batch):
                split_tensor_list = torch.split(free_energy, self.args.conf_size, dim=0)
                mean_tensor_list = [torch.mean(x, dim=0, keepdim=True) for x in split_tensor_list]
                agg_free_energy = torch.cat(mean_tensor_list, dim=0)
                agg_batch = [x//self.args.conf_size for x in batch]
                return agg_free_energy, agg_batch
            free_energy_a, batch_a = compute_agg_free_energy(free_energy_a, batch_a)
            free_energy_b, batch_b = compute_agg_free_energy(free_energy_b, batch_b)

        free_energy_a_padded = torch.nn.utils.rnn.pad_sequence(
            torch.split(free_energy_a, batch_a),
            padding_value=float("inf")
        )
        free_energy_b_padded = torch.nn.utils.rnn.pad_sequence(
            torch.split(free_energy_b, batch_b),
            padding_value=float("inf")
        )
        predicts = (
            torch.logsumexp(-free_energy_a_padded, dim=0)-
            torch.logsumexp(-free_energy_b_padded, dim=0)
        ) /  torch.log(torch.tensor([10.0])).item()

        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.mse_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss, predicts

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task, add aggregate acc and loss score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                        "smi": smi_list,
                    }
                )
                mae = np.abs(df["predict"] - df["target"]).mean()
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                metrics.log_scalar(f"{split}_mae", mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(
                    f"{split}_rmse", np.sqrt(mse), sample_size, round=4
                )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("infer_free_energy")
class InferFreeEnergyLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            classification_head_name=self.args.classification_head_name,
            features_only=True,
        )
        reg_output = net_output[0]
        loss = torch.tensor([0.01], device=sample["target"]["finetune_target"].device)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, self.args.num_classes)
                .data,
                "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
