# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import torch.nn.functional as F
import numpy as np


@register_criterion("deer_pruning_loss")
class DEERPruningCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
    ):
        super().__init__(task)
        self.eps = 0.1
        self.dict = task.tgt_dict

        self.pad = self.dict.pad()
        self.bos = self.dict.bos()
        self.eos = self.dict.eos()
        self.blank_id = self.dict.unk()
        self.ctc_loss = torch.nn.CTCLoss(blank=self.blank_id, reduction='none', zero_infinity=True)

    def _random_delete(self, target_tokens):
        max_len = target_tokens.size(1)
        target_mask = target_tokens.eq(self.pad)
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(target_tokens.eq(self.bos) | target_tokens.eq(self.eos), 0.0)
        target_score.masked_fill_(target_mask, 1)
        target_score, target_rank = target_score.sort(1)
        target_length = target_mask.size(1) - target_mask.float().sum(1, keepdim=True)
        
        target_cutoff = (2+ ((target_length - 2) * target_score.new_zeros(target_score.size(0), 1).uniform_()).long())
        target_cutoff = target_score.sort(1)[1] >= target_cutoff

        prev_target_tokens = (
            target_tokens.gather(1, target_rank)
            .masked_fill_(target_cutoff, self.pad)
            .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
        )
        prev_target_tokens = prev_target_tokens[:, : prev_target_tokens.ne(self.pad).sum(1).max()]

        return prev_target_tokens

    def build_input_data(self, sample):
        
        model_input ={
            "src_tokens": sample["net_input"]["src_tokens"],
            "src_lengths": sample["net_input"]["src_lengths"],
            "decoder_input": self._random_delete(sample["target"]),
            "tgt_tokens": sample["target"],
            "sampling_ratio": sample['sampling_ratio'] if 'sampling_ratio' in sample else 0.5,
        }
        return model_input

    def forward(self, model, sample, reduce=True):
        
        model.random_threshold()
        net_input = self.build_input_data(sample)
        model_out = model.forward_iter(**net_input)
        
        # mask_ins
        mask_ins_mask = model_out['mask_ins']['mask']
        mask_ins_out = model_out['mask_ins']['out'][mask_ins_mask]
        mask_ins_tgt = model_out['mask_ins']['tgt'][mask_ins_mask]
        mask_ins_loss = self.label_smooth_loss(mask_ins_out, mask_ins_tgt, 0.01)
        # word_ins
        word_ins_mask = model_out['word_ins']['mask']
        word_ins_out = model_out['word_ins']['out'][word_ins_mask]
        word_ins_tgt = model_out['word_ins']['tgt'][word_ins_mask]
        word_ins_loss = self.label_smooth_loss(word_ins_out, word_ins_tgt, self.eps)
        # word_del
        word_del_mask = model_out['word_del']['mask']
        word_del_out = model_out['word_del']['out'][word_del_mask]
        word_del_tgt = model_out['word_del']['tgt'][word_del_mask]
        del_loss = self.label_smooth_loss(word_del_out, word_del_tgt)
        
        ctc_out = model_out['ctc']['out']
        ctc_target = net_input['tgt_tokens']
        ctc_mask = model_out['ctc']['mask']
        output_lens = (~ctc_mask).sum(-1)
        target_lens = ctc_target.ne(self.pad).sum(-1)
        ctc_loss = self.ctc_label_smooth_loss(ctc_out, ctc_target, output_lens, target_lens)
        
        lr_loss = 30 * model.reg_threshold()
        loss = ctc_loss + mask_ins_loss + word_ins_loss + del_loss + lr_loss
        
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "mask_ins_loss": mask_ins_loss.data,
            "word_ins_loss": word_ins_loss.data,
            "del_loss": del_loss.data,
            "ctc_loss": ctc_loss.data,
            "lr_loss": lr_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def ctc_label_smooth_loss(self, net_out, net_target, output_lens, target_lens):
        net_logits = F.log_softmax(net_out, dim=-1).float().transpose(0, 1)
        nll_loss = self.ctc_loss(net_logits, net_target, output_lens, target_lens).sum() / (output_lens.sum().float())
        loss = nll_loss * (1. - self.eps) - net_logits.float().mean() * self.eps
        return loss

    def label_smooth_loss(self, net_out, net_target, ls=0.0):
        net_logits = F.log_softmax(net_out, dim=-1)
        nll_loss = F.nll_loss(net_logits, net_target, reduction="none").float().mean()
        loss = nll_loss * (1. - ls) - net_logits.float().mean() * ls
        return loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        mask_ins_loss_sum = sum(log.get("mask_ins_loss", 0) for log in logging_outputs)
        word_ins_loss_sum = sum(log.get("word_ins_loss", 0) for log in logging_outputs)
        del_loss_sum = sum(log.get("del_loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        lr_loss_sum = sum(log.get("lr_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "mask_ins_loss", mask_ins_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "word_ins_loss", word_ins_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "del_loss", del_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "lr_loss", lr_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
