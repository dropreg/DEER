# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    PositionalEmbedding,
    LayerNorm,
    SinusoidalPositionalEmbedding,
)
from fairseq.models.transformer import TransformerModel, TransformerDecoder, TransformerEncoder
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import torch.nn.functional as F
from .iterative_refinement_generator import DecoderOut
from fairseq.utils import new_arange
from .deer_transformer_layers import TransformerSharedLayer
import numpy as np


from torch_imputer.imputer import best_alignment
from fairseq.models.nat.levenshtein_utils import (
    _apply_del_words,
    _apply_ins_masks,
    _apply_ins_words,
    _fill,
    _get_del_targets,
    _get_ins_targets,
    _skip,
    _skip_encoder_out,
)

def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats

def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t

def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@register_model("deer_transformer")
class DEERTransformerModel(BaseFairseqModel):

    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = None

        self.bos = encoder.dictionary.bos()
        self.eos = encoder.dictionary.eos()
        self.pad = encoder.dictionary.pad()
        self.unk = encoder.dictionary.unk()

    def init_threshold(self):
        return 
    
    def inf_by_threshold(self):
        return

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        parser.add_argument('--offload-activations', action='store_true',
                            help='checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # fmt: on
        parser.add_argument("--apply-bert-init", action="store_true",
                            help="use custom param initialization for BERT",)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        # decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, None)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = RobertaEncoder(args, src_dict, embed_tokens)
        return encoder


    def forward(self, src_tokens, src_lengths, tgt_tokens, sampling_ratio, **kwargs):
        
        assert tgt_tokens is not None, "forward function only supports training."
        bsz, target_len = tgt_tokens.size()
        
        with torch.no_grad():
            ctc_out, ctc_padding_mask, encoder_out, ctc_tokens = self.encoder.forward_ctc(            
                normalize=False,
                src_tokens=src_tokens,
            )
            pred_ctc_tokens = ctc_out.argmax(-1)
            ctc_out_lprobs = F.log_softmax(ctc_out, dim=-1, dtype=torch.float32)
            seq_lens = (pred_ctc_tokens.ne(self.pad)).sum(1)
            target_lens = tgt_tokens.ne(self.pad).sum(1)
            best_aligns = best_alignment(ctc_out_lprobs.transpose(0, 1), tgt_tokens, seq_lens, target_lens, self.unk, zero_infinity=True)
            best_aligns_pad = torch.tensor([a + [0] * (ctc_out.shape[1] - len(a)) for a in best_aligns],
                                            device=ctc_out.device, dtype=tgt_tokens.dtype)
            oracle_pos = (best_aligns_pad // 2).clip(max=tgt_tokens.shape[1] - 1)
            oracle = tgt_tokens.gather(-1, oracle_pos)
            oracle_empty = oracle.masked_fill(best_aligns_pad % 2 == 0, self.unk)

            same_num = ((pred_ctc_tokens == oracle_empty) & (~ctc_padding_mask)).sum(1)
            keep_prob = ((seq_lens - same_num) / seq_lens * sampling_ratio).unsqueeze(-1)
            # keep: True, drop: False
            keep_word_mask = (torch.rand(ctc_tokens.shape, device=ctc_out.device) < keep_prob).bool()
            ctc_tokens = ctc_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)
        
        # for ctc prediction:
        ctc_out, ctc_padding_mask, encoder_out, _ = self.encoder.forward_ctc(            
            normalize=False,
            src_tokens=src_tokens,
            pesudo_tokens=ctc_tokens,
        )

        return {
            "ctc": {
                "out": ctc_out,
                "mask": ctc_padding_mask,
            },
        }

    def forward_iter(self, src_tokens, src_lengths, decoder_input, tgt_tokens, sampling_ratio, **kwargs):
        
        assert tgt_tokens is not None, "forward function only supports training."
        
        with torch.no_grad():
            ctc_out, ctc_padding_mask, encoder_out, ctc_tokens = self.encoder.forward_ctc(            
                normalize=False,
                src_tokens=src_tokens,
            )
            pred_ctc_tokens = ctc_out.argmax(-1)
            ctc_out_lprobs = F.log_softmax(ctc_out, dim=-1, dtype=torch.float32)
            seq_lens = (pred_ctc_tokens.ne(self.pad)).sum(1)
            target_lens = tgt_tokens.ne(self.pad).sum(1)
            best_aligns = best_alignment(ctc_out_lprobs.transpose(0, 1), tgt_tokens, seq_lens, target_lens, self.unk, zero_infinity=True)
            best_aligns_pad = torch.tensor([a + [0] * (ctc_out.shape[1] - len(a)) for a in best_aligns],
                                            device=ctc_out.device, dtype=tgt_tokens.dtype)
            oracle_pos = (best_aligns_pad // 2).clip(max=tgt_tokens.shape[1] - 1)
            oracle = tgt_tokens.gather(-1, oracle_pos)
            oracle_empty = oracle.masked_fill(best_aligns_pad % 2 == 0, self.unk)

            same_num = ((pred_ctc_tokens == oracle_empty) & (~ctc_padding_mask)).sum(1)
            keep_prob = ((seq_lens - same_num) / seq_lens * sampling_ratio).unsqueeze(-1)
            # keep: True, drop: False
            keep_word_mask = (torch.rand(ctc_tokens.shape, device=ctc_out.device) < keep_prob).bool()
            ctc_tokens = ctc_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)
        
        # for ctc prediction:
        ctc_out, ctc_padding_mask, encoder_out, _ = self.encoder.forward_ctc(            
            normalize=False,
            src_tokens=src_tokens,
            pesudo_tokens=ctc_tokens,
        )
        
        # generate training labels for insertion
        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
            decoder_input, tgt_tokens, self.pad, self.unk
        )
        mask_ins_targets = mask_ins_targets.clamp(min=0, max=511)  # for safe prediction
        mask_ins_masks = decoder_input[:, 1:].ne(self.pad)

        mask_ins_out = self.encoder.forward_mask_ins(
            normalize=False,
            decoder_input=decoder_input,
            encoder_out=encoder_out,
        )
        word_ins_out = self.encoder.forward_word_ins(
            normalize=False,
            decoder_input=masked_tgt_tokens,
            encoder_out=encoder_out,
        )

        word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]
        word_predictions.masked_scatter_(~masked_tgt_masks, tgt_tokens[~masked_tgt_masks])

        # generate training labels for deletion
        word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
        word_del_out = self.encoder.forward_word_del(
            normalize=False,
            decoder_input=word_predictions,
            encoder_out=encoder_out,
        )
        word_del_masks = word_predictions.ne(self.pad)

        # import pdb; pdb.set_trace()
        return {
            "ctc": {
                "out": ctc_out,
                "mask": ctc_padding_mask,
            },
            "mask_ins": {
                "out": mask_ins_out,
                "tgt": mask_ins_targets,
                "mask": mask_ins_masks,
            },
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": masked_tgt_masks,
            },
            "word_del": {
                "out": word_del_out,
                "tgt": word_del_targets,
                "mask": word_del_masks,
            },
        }

    def forward_correct(self, src_tokens, src_lengths, decoder_input, tgt_tokens, sampling_ratio, **kwargs):
        
        assert tgt_tokens is not None, "forward function only supports training."
        
        with torch.no_grad():
            ctc_out, ctc_padding_mask, encoder_out, ctc_tokens = self.encoder.forward_ctc(            
                normalize=False,
                src_tokens=src_tokens,
            )
            pred_ctc_tokens = ctc_out.argmax(-1)
            ctc_out_lprobs = F.log_softmax(ctc_out, dim=-1, dtype=torch.float32)
            seq_lens = (pred_ctc_tokens.ne(self.pad)).sum(1)
            target_lens = tgt_tokens.ne(self.pad).sum(1)
            best_aligns = best_alignment(ctc_out_lprobs.transpose(0, 1), tgt_tokens, seq_lens, target_lens, self.unk, zero_infinity=True)
            best_aligns_pad = torch.tensor([a + [0] * (ctc_out.shape[1] - len(a)) for a in best_aligns],
                                            device=ctc_out.device, dtype=tgt_tokens.dtype)
            oracle_pos = (best_aligns_pad // 2).clip(max=tgt_tokens.shape[1] - 1)
            oracle = tgt_tokens.gather(-1, oracle_pos)
            oracle_empty = oracle.masked_fill(best_aligns_pad % 2 == 0, self.unk)

            same_num = ((pred_ctc_tokens == oracle_empty) & (~ctc_padding_mask)).sum(1)
            keep_prob = ((seq_lens - same_num) / seq_lens * sampling_ratio).unsqueeze(-1)
            # keep: True, drop: False
            keep_word_mask = (torch.rand(ctc_tokens.shape, device=ctc_out.device) < keep_prob).bool()
            ctc_tokens = ctc_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)
        
        # for ctc prediction:
        ctc_out, ctc_padding_mask, encoder_out, _ = self.encoder.forward_ctc(            
            normalize=False,
            src_tokens=src_tokens,
            pesudo_tokens=ctc_tokens,
        )

        output_tokens, _ = self.ectract_seq(ctc_out, ctc_padding_mask)
        # pesudo target
        if output_tokens.size(1) > tgt_tokens.size(1):
            new_tgt_tokens = output_tokens.new_zeros(output_tokens.size(0), output_tokens.size(1)).fill_(self.pad)
            new_tgt_tokens[:, :tgt_tokens.size(1)] = tgt_tokens
            tgt_tokens = new_tgt_tokens
            decoder_del_input = output_tokens
        else:
            new_decoder_input = tgt_tokens.new_zeros(tgt_tokens.size(0), tgt_tokens.size(1)).fill_(self.pad)
            new_decoder_input[:, :output_tokens.size(1)] = output_tokens      
            decoder_del_input = new_decoder_input

        word_del_out = self.encoder.forward_word_del(
            normalize=False,
            decoder_input=decoder_del_input,
            encoder_out=encoder_out,
        )
        word_del_masks = decoder_del_input.ne(self.pad)
        can_del_word = decoder_del_input.ne(self.pad).sum(1) > 2
        word_del_targets = _get_del_targets(decoder_del_input, tgt_tokens, self.pad)

        _tokens, _, _ = _apply_del_words(
            decoder_del_input.clone()[can_del_word],
            None,
            None,
            word_del_targets[can_del_word].bool(),
            self.pad,
            self.bos,
            self.eos,
        )
        decoder_input = _fill(decoder_del_input.clone(), can_del_word, _tokens, self.pad)

        # generate training labels for insertion
        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
            decoder_input, tgt_tokens, self.pad, self.unk
        )
        mask_ins_targets = mask_ins_targets.clamp(min=0, max=511)  # for safe prediction
        mask_ins_masks = decoder_input[:, 1:].ne(self.pad)

        mask_ins_out = self.encoder.forward_mask_ins(
            normalize=False,
            decoder_input=decoder_input,
            encoder_out=encoder_out,
        )
        word_ins_out = self.encoder.forward_word_ins(
            normalize=False,
            decoder_input=masked_tgt_tokens,
            encoder_out=encoder_out,
        )
        return {
            "ctc": {
                "out": ctc_out,
                "mask": ctc_padding_mask,
            },
            "mask_ins": {
                "out": mask_ins_out,
                "tgt": mask_ins_targets,
                "mask": mask_ins_masks,
            },
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": masked_tgt_masks,
            },
            "word_del": {
                "out": word_del_out,
                "tgt": word_del_targets,
                "mask": word_del_masks,
            },
        }

    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)

    @property
    def allow_length_beam(self):
        return True
    
    def ectract_seq(self, ctc_output, ctc_padding_mask):
        _scores, _tokens = ctc_output.max(-1)
        _tokens = _tokens.masked_fill(ctc_padding_mask, self.pad)
        output_tokens = torch.zeros_like(_tokens).fill_(self.pad)
        output_scores = torch.zeros_like(_scores)
        batch_size, seq_len = _tokens.size()
        
        _tokens[:, 0] = self.bos
        _tokens[:,-1] = self.eos

        rep_mask = _tokens != torch.cat([_tokens[:,1:], _tokens[:,0:1]], dim=-1)
        unk_mask = _tokens.ne(self.bos) & _tokens.ne(self.eos) & _tokens.ne(self.pad) & _tokens.ne(self.unk)
        all_mask = rep_mask & unk_mask

        length_tgt = all_mask.sum(1) + 1
        idx_length = utils.new_arange(_tokens, _tokens.size(1))
        order_mask = torch.zeros_like(_tokens).bool()
        order_mask.masked_fill_(idx_length[None, :] < length_tgt[:, None], True)
        order_mask[:, 0] = False
        output_tokens.masked_scatter_(order_mask, _tokens.masked_select(all_mask))
        output_tokens[:, 0] = self.bos
        output_tokens.scatter_(1, length_tgt[:, None], self.eos)

        output_scores.masked_scatter_(order_mask, _scores.masked_select(all_mask))
        output_scores[:, 0] = 0.0
        output_scores.scatter_(1, length_tgt[:, None], 0.0)
        return output_tokens, output_scores

    def forward_decoder(
        self, decoder_out, src_tokens, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs
    ):  
        step = decoder_out.step
        max_step = decoder_out.max_step
        attn = decoder_out.attn
        history = decoder_out.history

        if step == 0:
            ctc_output, ctc_padding_mask, encoder_out, _ = self.encoder.forward_ctc(            
                normalize=True,
                src_tokens=src_tokens,
            )
            output_tokens, output_scores = self.ectract_seq(ctc_output, ctc_padding_mask)
        else:
            output_tokens = decoder_out.output_tokens
            output_scores = decoder_out.output_scores
            bsz = output_tokens.size(0)
            if max_ratio is None:
                max_lens = torch.zeros_like(output_tokens).fill_(511)
            else:
                if not encoder_out["encoder_padding_mask"]:
                    max_src_len = encoder_out["encoder_out"].size(0)
                    src_lens = encoder_out["encoder_out"].new(bsz).fill_(max_src_len)
                else:
                    src_lens = (~encoder_out["encoder_padding_mask"][0]).sum(1)
                max_lens = (src_lens * max_ratio).clamp(min=10).long()

            # delete words
            # do not delete tokens if it is <s> </s>
            can_del_word = output_tokens.ne(self.pad).sum(1) > 2
            if can_del_word.sum() != 0:  # we cannot delete, skip
                word_del_score = self.encoder.forward_word_del(
                    normalize=True,
                    decoder_input=_skip(output_tokens, can_del_word),
                    encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_del_word),
                )
                word_del_pred = word_del_score.max(-1)[1].bool()

                _tokens, _scores, _attn = _apply_del_words(
                    output_tokens[can_del_word],
                    output_scores[can_del_word],
                    None,
                    word_del_pred,
                    self.pad,
                    self.bos,
                    self.eos,
                )
                output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
                output_scores = _fill(output_scores, can_del_word, _scores, 0)
                attn = _fill(attn, can_del_word, _attn, 0.0)

                if history is not None:
                    history.append(output_tokens.clone())
            # insert placeholders
            can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
            if can_ins_mask.sum() != 0:
                mask_ins_score = self.encoder.forward_mask_ins(
                    normalize=True,
                    decoder_input=_skip(output_tokens, can_ins_mask),
                    encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_mask),
                )
                if eos_penalty > 0.0:
                    mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
                mask_ins_pred = mask_ins_score.max(-1)[1]
                mask_ins_pred = torch.min(
                    mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
                )

                _tokens, _scores = _apply_ins_masks(
                    output_tokens[can_ins_mask],
                    output_scores[can_ins_mask],
                    mask_ins_pred,
                    self.pad,
                    self.unk,
                    self.eos,
                )
                output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
                output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

                if history is not None:
                    history.append(output_tokens.clone())

            # insert words
            can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
            if can_ins_word.sum() != 0:
                word_ins_score = self.encoder.forward_word_ins(
                    normalize=True,
                    decoder_input=_skip(output_tokens, can_ins_word),
                    encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_word),
                )
                word_ins_score, word_ins_pred = word_ins_score.max(-1)
                _tokens, _scores = _apply_ins_words(
                    output_tokens[can_ins_word],
                    output_scores[can_ins_word],
                    word_ins_pred,
                    word_ins_score,
                    self.unk,
                )

                output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
                output_scores = _fill(output_scores, can_ins_word, _scores, 0)
                attn = _fill(attn, can_ins_word, None, 0.0)

                if history is not None:
                    history.append(output_tokens.clone())
        
        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]
        
        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn,
            history=history,
        ), encoder_out


    def initialize_output_tokens(self, src_tokens):
        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens[:, 1] = self.eos

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(src_tokens)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, beam_size)
            - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.mask_idx
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )

    def upgrade_state_dict_named(self, state_dict, name):

        for k in list(state_dict.keys()):
            if "sentence_encoder.embed_tokens.weight" in k:
                state_dict['encoder.embed_tokens.weight'] = state_dict[k]
                del state_dict[k]
                # self defined
                state_dict["encoder.embed_mask_ins.weight"] = self.encoder.embed_mask_ins.weight
                state_dict["encoder.embed_word_del.weight"] = self.encoder.embed_word_del.weight

            elif "sentence_encoder.embed_positions.weight" in k:
                state_dict['encoder.embed_positions.weight'] = state_dict[k]
                del state_dict[k]
            elif "sentence_encoder.layers." in k:
                replace_encoder_key = k.replace("decoder.sentence_encoder", "encoder")
                state_dict[replace_encoder_key] = state_dict[k]
                del state_dict[k]
            elif "sentence_encoder.emb_layer_norm" in k:
                if "weight" in k:
                    state_dict["encoder.emb_layer_norm.weight"] = state_dict[k]
                if "bias" in k:
                    state_dict["encoder.emb_layer_norm.bias"] = state_dict[k]
                del state_dict[k]
            elif "decoder.lm_head" in k:
                replace_encoder_key = k.replace("decoder", "encoder")
                state_dict[replace_encoder_key] = state_dict[k]
                del state_dict[k]
            if 'encoder.sentence_encoder.version' in k:
                del state_dict[k]
        super().upgrade_state_dict_named(state_dict, name)


class RobertaEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        
        self.embed_dim = embed_tokens.embedding_dim
        self.embed_positions = (
            PositionalEmbedding(
                512,
                self.embed_dim,
                padding_idx=self.padding_idx,
                learned=True,
            )
        )
        self.emb_layer_norm = LayerNorm(self.embed_dim)

        ###### Decoding
        self.lm_head = self.build_lm_head(
            embed_dim=self.embed_dim,
            output_dim=len(dictionary),
            activation_fn="gelu",
            weight=embed_tokens.weight,
        )        
        self.embed_mask_ins = Embedding(512, self.embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.embed_dim, None)
        
        self.embed_mask_ins.weight.data.normal_(mean=0.0, std=0.02)
        self.embed_word_del.weight.data.normal_(mean=0.0, std=0.02)
        ################################################################
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()
        self.pad = dictionary.pad()
        self.unk = dictionary.unk()
        self.dynamic_ratio = 1.0

    def output_layer(self, features):
        return self.lm_head(features)

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    def build_encoder_layer(self, args):
        layer = TransformerSharedLayer(args)
        return layer

    def forward_embedding(self, src_tokens):
        padding_mask = src_tokens.eq(self.padding_idx)
        token_embedding = self.embed_tokens(src_tokens)
        x = embed = token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        # roberta layer norm
        x = self.emb_layer_norm(x)
        
        x = self.dropout_module(x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        return x, embed, padding_mask

    def forward_ctc(self, normalize, src_tokens, pesudo_tokens=None, **unused):
        tokens, features, tgt_padding_mask, src_out = self.extract_features_ctc(src_tokens, pesudo_tokens)
        out = self.output_layer(features)
        if normalize:
            return F.log_softmax(out, -1),  tgt_padding_mask, src_out, tokens
        else:
            return out, tgt_padding_mask, src_out, tokens
        
    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(~tgt_masks, 0)
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(*mapped_inputs.size(), src_embeds.size(-1)),
        )
        return copied_embedding
        
    def initialize_ctc_input(self, src_tokens):
        # NMT alpha = 2 Summerization alpha = 1
        alpha = 2
        src_length = src_tokens.ne(self.pad).sum(-1) * alpha
        if (src_length > 511).sum() > 0:
            src_length_max = src_length.clone()
            src_length_max[:] = 511
            length_tgt = src_length_max
        else:
            length_tgt = src_length
        
        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)
        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), max_length).fill_(self.pad)
        initial_output_tokens.masked_fill_(idx_length[None, :] < length_tgt[:, None], self.unk)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        return initial_output_tokens

    def extract_features_ctc(
        self,
        src_tokens,
        pesudo_tokens=None,
        **unused
    ):  
        src_x, src_embed, src_padding_mask = self.forward_embedding(src_tokens)
        if pesudo_tokens is not None:
            ctc_tokens = pesudo_tokens
        else:
            ctc_tokens = self.initialize_ctc_input(src_tokens)
        tgt_x, tgt_embed, tgt_padding_mask = self.forward_embedding(ctc_tokens)

        # copy source embedding to target
        # tgt_padding_mask = ctc_tokens.eq(self.pad)
        # tgt_embed = self.forward_copying_source(src_embed, ~src_padding_mask, ~tgt_padding_mask)
        # if self.embed_positions is not None:
        #     tgt_x = tgt_embed + self.embed_positions(ctc_tokens)
        # tgt_x = self.emb_layer_norm(tgt_x)
        # tgt_x = self.dropout_module(tgt_x)
        # tgt_x = tgt_x * (1 - tgt_padding_mask.unsqueeze(-1).type_as(tgt_x))

        tgt_start = src_x.size(1)
        x = torch.cat([src_x, tgt_x], dim=1)
        padding_mask = torch.cat([src_padding_mask, tgt_padding_mask], dim=-1)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        src_hidden_states = []
        attn_mask = self.make_attention_mask_cuda(x, src_padding_mask, tgt_padding_mask)
        for i, layer in enumerate(self.layers):
            src_hidden_states.append(x[:tgt_start, :, :])
            x = layer.forward_encoder(
                x,
                encoder_padding_mask=padding_mask,
                attn_mask=attn_mask,
            )
        out_x = x[tgt_start:, :, :]
        # T x B x C -> B x T x C

        src_out = {
            "encoder_out": [x[:tgt_start, :, :]],  # T x B x C
            "encoder_padding_mask": [src_padding_mask],  # B x T
            "encoder_embedding": [src_embed],  # B x T x C
            "encoder_states": src_hidden_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }
        out_x = out_x.transpose(0, 1)
        return ctc_tokens, out_x, tgt_padding_mask, src_out
        
    def make_attention_mask_cuda(self, x_tensor, src_padding_mask, tgt_padding_mask):
        bsz, src_len = src_padding_mask.size()
        _, tgt_len = tgt_padding_mask.size()
        
        src_left = torch.zeros((src_len, src_len))
        src_right = torch.zeros((src_len, tgt_len)).fill_(float("-inf"))
        src_mask = torch.cat([src_left, src_right], dim=-1)
        tgt_mask = torch.zeros((tgt_len, src_len + tgt_len))
        return torch.cat([src_mask, tgt_mask], dim=0).to(x_tensor)
        
    def extract_features(
        self,
        tgt_tokens,
        encoder_out,
        **unused
    ):
        
        src_hidden_states = encoder_out["encoder_states"]
        src_padding_mask = encoder_out["encoder_padding_mask"][0]

        x, _, tgt_padding_mask = self.forward_embedding(tgt_tokens)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for i, layer in enumerate(self.layers):
            
            x = layer.forward_decoder(
                x,
                src_hidden_states[i],
                encoder_padding_mask=src_padding_mask,
                decoder_padding_mask=tgt_padding_mask,
            )
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        return x, tgt_padding_mask

    def forward_mask_ins(self, normalize, decoder_input, encoder_out, **unused):
        features, extra = self.extract_features(
            decoder_input,
            encoder_out,
        )
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        out_feat_size, in_feat_size = self.embed_mask_ins.weight.size()
        decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
        
        if normalize:
            return F.log_softmax(decoder_out, -1)
        return decoder_out

    def forward_word_ins(self, normalize, decoder_input, encoder_out, **unused):
        features, extra = self.extract_features(
            decoder_input,
            encoder_out,
        )
        decoder_out = self.output_layer(features)
        if normalize:
            return F.log_softmax(decoder_out, -1)
        return decoder_out
        
    def forward_word_del(self, normalize, decoder_input, encoder_out, **unused):
        features, extra = self.extract_features(
            decoder_input,
            encoder_out,
        )
        out_feat_size, in_feat_size = self.embed_word_del.weight.size()
        decoder_out = F.linear(features, self.embed_word_del.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1)
        return decoder_out

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }


    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        new_encoder_states = []
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                # encoder_states[idx] = state.index_select(1, new_order)
                new_encoder_states.append(state.index_select(1, new_order))

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": new_encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)
        
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("deer_transformer", "deer_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


@register_model_architecture("deer_transformer", "deer_transformer_large")
def large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)
