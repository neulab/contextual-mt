import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import numpy as np


@dataclass
class AttentionLossConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("attention_loss", dataclass=AttentionLossConfig)
class AttentionLoss(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.regularize_heads = task.args.regularize_heads
        self.lamb = task.args.kl_lambda
        self.reg_attn = task.args.regularize_attention
        self.enc_alignment_layer = task.args.enc_alignment_layer
        self.cross_alignment_layer = task.args.cross_alignment_layer
        self.self_alignment_layer = task.args.self_alignment_layer
        self.dec_alignment_layer = task.args.dec_alignment_layer

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            enc_alignment_layer=self.enc_alignment_layer,
            dec_alignment_layer=self.dec_alignment_layer,
            self_alignment_layer=self.self_alignment_layer,
            cross_alignment_layer=self.cross_alignment_layer
        )
        output_features = net_output[1]
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        cross_attn_loss = 0
        self_attn_loss = 0
        enc_attn_loss = 0
        if "highlights" in sample:

            src_highlights = torch.cat(
                [
                    sample["highlights"]["src_ctx_highlights"],
                    sample["highlights"]["source_highlights"],
                ],
                axis=1,
            )
            tgt_highlights = torch.cat(
                [
                    sample["highlights"]["tgt_ctx_highlights"],
                    sample["highlights"]["target_highlights"],
                ],
                axis=1,
            )
            src_words = torch.cat(
                [
                    torch.zeros_like(sample["highlights"]["src_ctx_highlights"]),
                    sample["highlights"]["src_words"],
                ],
                axis=1,
            )
            tgt_words = torch.cat(
                [
                    torch.zeros_like(sample["highlights"]["tgt_ctx_highlights"]),
                    sample["highlights"]["tgt_words"],
                ],
                axis=1,
            )

            cross_attn = output_features["attn"][
                0
            ]  # batchsize x tgt_seq_len (query) x src_seq_len (value)
            self_attn = output_features["attn"][
                1
            ]  # batchsize x tgt_seq_len (query) x tgt_seq_len (value)
            enc_self_attn = output_features["encoder_out"][
                "enc_self_attn"
            ]  # num_layers x num_heads x batchsize x src_seq_len (query) x src_seq_len (value)

            if self.regularize_heads < 0:
                src_highlights = src_highlights.add(src_words)
                src_highlights = torch.where(
                    src_highlights > 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()
                )
                tgt_highlights = tgt_highlights.add(tgt_words)
                tgt_highlights = torch.where(
                    tgt_highlights > 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()
                )

            # normalize the highlights
            updated_tags = src_highlights + 1e-9
            normalizing_const = torch.sum(updated_tags, dim=1)
            src_normalized_tags = torch.einsum(
                "ij,i->ij", updated_tags, 1.0 / normalizing_const
            )

            updated_tags = tgt_highlights + 1e-9
            normalizing_const = torch.sum(updated_tags, dim=1)
            tgt_normalized_tags = torch.einsum(
                "ij,i->ij", updated_tags, 1.0 / normalizing_const
            )

            kld_loss = nn.KLDivLoss(reduction="sum")

            cross_attn_mean = []
            self_attn_mean = []
            enc_attn_mean = []

            for batch_id, words in enumerate(tgt_words):
                for i in np.where(words.cpu() == 1)[0]:
                    if "self" in self.reg_attn:
                        for self_a in self_attn:
                            if self_a.dim() == 4:
                                if self.regularize_heads > -1:
                                    self_a = self_a[self.regularize_heads]
                                else:
                                    self_a = self_a.mean(dim=0)
                            self_attn_loss += kld_loss(
                                F.log_softmax(self_a[batch_id, i], -1),
                                tgt_normalized_tags[batch_id],
                            )

                        self_attn_mean.append(
                            torch.sum(
                                torch.mul(
                                    self_a[batch_id, i], tgt_normalized_tags[batch_id]
                                )
                            )
                            / torch.sum(tgt_normalized_tags[batch_id])
                        )
                    if "cross" in self.reg_attn:
                        for cross_a in cross_attn:
                            if cross_a.dim() == 4:
                                if self.regularize_heads > -1:
                                    cross_a = cross_a[self.regularize_heads]
                                else:
                                    cross_a = cross_a.mean(dim=0)
                            cross_attn_loss += kld_loss(
                                F.log_softmax(cross_a[batch_id, i], -1),
                                src_normalized_tags[batch_id],
                            )

                            cross_attn_mean.append(
                                torch.sum(
                                    torch.mul(
                                        cross_a[batch_id, i],
                                        src_normalized_tags[batch_id],
                                    )
                                )
                                / torch.sum(src_normalized_tags[batch_id])
                            )

            if "enc" in self.reg_attn:
                for batch_id, words in enumerate(src_words):
                    for i in np.where(words.cpu() == 1)[0]:
                        for enc_a in enc_self_attn:
                            if enc_a.dim() == 4:
                                if self.regularize_heads > -1:
                                    enc_a = enc_a[self.regularize_heads]
                                else:
                                    enc_a = enc_a.mean(dim=0)
                            enc_attn_loss += kld_loss(
                                F.log_softmax(enc_a[batch_id, i], -1),
                                src_normalized_tags[batch_id],
                            )

                            enc_attn_mean.append(
                                torch.sum(
                                    torch.mul(
                                        enc_a[batch_id, i],
                                        src_normalized_tags[batch_id],
                                    )
                                )
                                / torch.sum(src_normalized_tags[batch_id])
                            )

            if len(cross_attn_mean) > 0:
                cross_attn_mean = sum(cross_attn_mean) / len(cross_attn_mean)
            if len(self_attn_mean) > 0:
                self_attn_mean = sum(self_attn_mean) / len(self_attn_mean)
            if len(enc_attn_mean) > 0:
                enc_attn_mean = sum(enc_attn_mean) / len(enc_attn_mean)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        attn_loss = 0

        if cross_attn_loss != 0:
            logging_output["cross_attn_mean"] = cross_attn_mean.data
            logging_output["cross_attn_loss"] = cross_attn_loss.data
            attn_loss += self.lamb * cross_attn_loss

        if self_attn_loss != 0:
            logging_output["self_attn_mean"] = self_attn_mean.data
            logging_output["self_attn_loss"] = self_attn_loss.data
            attn_loss += self.lamb * self_attn_loss

        if enc_attn_loss != 0:
            logging_output["enc_attn_mean"] = enc_attn_mean.data
            logging_output["enc_attn_loss"] = enc_attn_loss.data

            attn_loss += self.lamb * enc_attn_loss

        logging_output["attn_loss"] = attn_loss.data if attn_loss != 0 else 0
        loss += attn_loss

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        attn_loss_sum = sum(log.get("attn_loss", 0) for log in logging_outputs)
        cross_attn_loss_sum = sum(
            log.get("cross_attn_loss", 0) for log in logging_outputs
        )
        self_attn_loss_sum = sum(
            log.get("self_attn_loss", 0) for log in logging_outputs
        )
        enc_attn_loss_sum = sum(log.get("enc_attn_loss", 0) for log in logging_outputs)
        cross_attn_sum = sum(log.get("cross_attn_mean", 0) for log in logging_outputs)
        self_attn_sum = sum(log.get("self_attn_mean", 0) for log in logging_outputs)
        enc_attn_sum = sum(log.get("enc_attn_mean", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        num_attn = sum(
            [
                1
                for log in logging_outputs
                if (
                    "cross_attn_mean" in log
                    or "self_attn_mean" in log
                    or "enc_attn_mean" in log
                )
            ]
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )

        if num_attn > 0:
            metrics.log_scalar(
                "enc_loss",
                enc_attn_loss_sum / num_attn / math.log(2),
                num_attn,
                round=3,
            )
            metrics.log_scalar(
                "cross_loss",
                cross_attn_loss_sum / num_attn / math.log(2),
                num_attn,
                round=3,
            )
            metrics.log_scalar(
                "self_loss",
                self_attn_loss_sum / num_attn / math.log(2),
                num_attn,
                round=3,
            )
            metrics.log_scalar(
                "enc_attn", enc_attn_sum / num_attn / math.log(2), num_attn, round=3
            )
            metrics.log_scalar(
                "cross_attn", cross_attn_sum / num_attn / math.log(2), num_attn, round=3
            )
            metrics.log_scalar(
                "self_attn", self_attn_sum / num_attn / math.log(2), num_attn, round=3
            )
            metrics.log_scalar(
                "attn_loss", attn_loss_sum / num_attn / math.log(2), num_attn, round=3
            )

        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
