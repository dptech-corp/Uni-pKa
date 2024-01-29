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


@register_loss("pretrain_mlm")
class PretrainMLMLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.charge_padding_idx = task.charge_dictionary.pad()
        self.seed = task.seed
        self.dist_mean = 6.174412864984603
        self.dist_std = 216.17030997643033

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens_a = sample["net_input_a"][-1].ne(self.padding_idx)
        all_output_a = model(
            sample["net_input_a"],
            classification_head_name=self.args.classification_head_name,
            encoder_masked_tokens=masked_tokens_a
        )
        masked_tokens_b = sample["net_input_b"][-1].ne(self.padding_idx)
        all_output_b = model(
            sample["net_input_b"],
            classification_head_name=self.args.classification_head_name,
            encoder_masked_tokens=masked_tokens_b
        )
        net_output_a, batch_a = all_output_a[:2]
        net_output_b, batch_b = all_output_b[:2]

        loss, predict = self.compute_loss(model, net_output_a, net_output_b, batch_a, batch_b, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=predict.device)
                targets_std = torch.tensor(self.task.std, device=predict.device)
                predict = predict * targets_std + targets_mean
            logging_output = {
                "pka_loss": loss.data,
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
                "pka_loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }

        loss, logging_output = self.compute_mlm_loss(loss, all_output_a, masked_tokens_a, logging_output, reduce= reduce)
        loss, logging_output = self.compute_mlm_loss(loss, all_output_b, masked_tokens_b, logging_output, reduce= reduce)
        logging_output['loss'] = loss.data

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

    def compute_mlm_loss(self, loss, all_output, masked_tokens, logging_output, reduce= True):
        (_, _,
        logits_encoder, charge_logits, encoder_distance, encoder_coord, x_norm, delta_encoder_pair_rep_norm,
        token_targets, charge_targets, coord_targets, dist_targets) = all_output
        sample_size = masked_tokens.long().sum()

        if self.args.masked_token_loss > 0:
            target = token_targets
            if masked_tokens is not None:
                target = target[masked_tokens]
            masked_token_loss = F.nll_loss(
                F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
                target,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )
            masked_pred = logits_encoder.argmax(dim=-1)
            masked_hit = (masked_pred == target).long().sum()
            masked_cnt = sample_size
            if 'masked_token_loss' in logging_output:
                logging_output['seq_len'] += token_targets.size(1) * token_targets.size(0)
                logging_output['masked_token_loss'] += masked_token_loss.data
                logging_output['masked_token_hit'] += masked_hit.data
                logging_output['masked_token_cnt'] += masked_cnt
            else:
                logging_output['seq_len'] = token_targets.size(1) * token_targets.size(0)
                logging_output['masked_token_loss'] = masked_token_loss.data
                logging_output['masked_token_hit'] = masked_hit.data
                logging_output['masked_token_cnt'] = masked_cnt
            loss += masked_token_loss * self.args.masked_token_loss

        if self.args.masked_charge_loss > 0:
            target = charge_targets
            if masked_tokens is not None:
                target = target[masked_tokens]
            masked_charge_loss = F.nll_loss(
                F.log_softmax(charge_logits, dim=-1, dtype=torch.float32),
                target,
                ignore_index=self.charge_padding_idx,
                reduction="sum" if reduce else "none",
            )
            masked_pred = charge_logits.argmax(dim=-1)
            masked_hit = (masked_pred == target).long().sum()
            masked_cnt = sample_size
            if 'masked_charge_loss' in logging_output:
                logging_output['masked_charge_loss'] += masked_charge_loss.data
                logging_output['masked_charge_hit'] += masked_hit.data
                logging_output['masked_charge_cnt'] += masked_cnt
            else:
                logging_output['masked_charge_loss'] = masked_charge_loss.data
                logging_output['masked_charge_hit'] = masked_hit.data
                logging_output['masked_charge_cnt'] = masked_cnt
            loss += masked_charge_loss * self.args.masked_charge_loss

        if self.args.masked_coord_loss > 0:
            # real = mask + delta
            masked_coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_targets[masked_tokens].view(-1, 3),
                reduction="sum" if reduce else "none",
                beta=1.0,
            )
            loss = loss + masked_coord_loss * self.args.masked_coord_loss
            # restore the scale of loss for displaying
            if 'masked_coord_loss' in logging_output:
                logging_output["masked_coord_loss"] += masked_coord_loss.data
            else:
                logging_output["masked_coord_loss"] = masked_coord_loss.data

        if self.args.masked_dist_loss > 0:
            dist_masked_tokens = masked_tokens
            masked_dist_loss = self.cal_dist_loss(
                encoder_distance, dist_masked_tokens, dist_targets, reduce=reduce, normalize=True, 
            )
            loss = loss + masked_dist_loss * self.args.masked_dist_loss
            if 'masked_dist_loss' in logging_output:
                logging_output["masked_dist_loss"] += masked_dist_loss.data
            else:
                logging_output["masked_dist_loss"] = masked_dist_loss.data

        if self.args.x_norm_loss > 0 and x_norm is not None:
            loss = loss + self.args.x_norm_loss * x_norm
            if 'x_norm_loss' in logging_output:
                logging_output["x_norm_loss"] += x_norm.data
            else:
                logging_output["x_norm_loss"] = x_norm.data

        if (
            self.args.delta_pair_repr_norm_loss > 0
            and delta_encoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            )
            if 'delta_pair_repr_norm_loss' in logging_output:
                logging_output[
                    "delta_pair_repr_norm_loss"
                ] += delta_encoder_pair_rep_norm.data
            else:
                logging_output[
                    "delta_pair_repr_norm_loss"
                ] = delta_encoder_pair_rep_norm.data

        return loss, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        if "valid" in split or "test" in split:
            sample_size *= logging_outputs[0].get("conf_size",0)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)
        pka_loss = sum(log.get("pka_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "pka_loss", pka_loss / sample_size, sample_size, round=3
        )
        masked_token_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        if masked_token_loss >0:
            metrics.log_scalar(
                "masked_token_loss", masked_token_loss / sample_size, sample_size, round=3
            )
            masked_acc = sum(
                log.get("masked_token_hit", 0) for log in logging_outputs
            ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
            metrics.log_scalar("masked_token_acc", masked_acc, sample_size, round=3)

        masked_charge_loss = sum(log.get("masked_charge_loss", 0) for log in logging_outputs)
        if masked_charge_loss >0:
            metrics.log_scalar(
                "masked_charge_loss", masked_charge_loss / sample_size, sample_size, round=3
            )
            masked_acc = sum(
                log.get("masked_charge_hit", 0) for log in logging_outputs
            ) / sum(log.get("masked_charge_cnt", 0) for log in logging_outputs)
            metrics.log_scalar("masked_charge_acc", masked_acc, sample_size, round=3)

        masked_coord_loss = sum(
            log.get("masked_coord_loss", 0) for log in logging_outputs
        )
        if masked_coord_loss > 0:
            metrics.log_scalar(
                "masked_coord_loss",
                masked_coord_loss / sample_size,
                sample_size,
                round=3,
            )

        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3
            )

        x_norm_loss = sum(log.get("x_norm_loss", 0) for log in logging_outputs)
        if x_norm_loss > 0:
            metrics.log_scalar(
                "x_norm_loss", x_norm_loss / sample_size, sample_size, round=3
            )

        delta_pair_repr_norm_loss = sum(
            log.get("delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "delta_pair_repr_norm_loss",
                delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )

        if "valid" in split or "test" in split:
            sample_size //= logging_outputs[0].get("conf_size",0)
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

    def cal_dist_loss(self, dist, dist_masked_tokens, dist_targets, reduce= True, normalize=False):
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = dist_targets[dist_masked_tokens]
        non_pad_pos = masked_distance_target > 0
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[non_pad_pos].view(-1).float(),
            masked_distance_target[non_pad_pos].view(-1),
            reduction="sum" if reduce else "none",
            beta=1.0,
        )
        return masked_dist_loss
