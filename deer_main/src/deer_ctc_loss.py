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


@register_criterion("deer_ctc_loss")
class DEERCTCCrossEntropyCriterion(FairseqCriterion):
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

    def build_input_data(self, sample):
        
        model_input ={
            "src_tokens": sample["net_input"]["src_tokens"],
            "src_lengths": sample["net_input"]["src_lengths"],
            "tgt_tokens": sample["target"],
        }
        return model_input

    def forward(self, model, sample, reduce=True):

        net_input = self.build_input_data(sample)
        model_out = model(**net_input)
        
        ctc_out = model_out['ctc']['out']
        ctc_target = net_input['tgt_tokens']
        ctc_mask = model_out['ctc']['mask']
        output_lens = (~ctc_mask).sum(-1)
        target_lens = ctc_target.ne(self.pad).sum(-1)
        ctc_loss, nll_loss = self.ctc_label_smooth_loss(ctc_out, ctc_target, output_lens, target_lens)

        loss = ctc_loss
        
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "ctc_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def ctc_label_smooth_loss(self, net_out, net_target, output_lens, target_lens):
        net_logits = F.log_softmax(net_out, dim=-1).float().transpose(0, 1)
        nll_loss = self.ctc_loss(net_logits, net_target, output_lens, target_lens).sum() / (output_lens.sum().float())
        loss = nll_loss * (1. - self.eps) - net_logits.float().mean() * self.eps
        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
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
