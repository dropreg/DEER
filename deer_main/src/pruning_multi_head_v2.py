# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
import numbers
from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise
from fairseq.models.fairseq_incremental_decoder import FairseqIncrementalDecoder
from torch import autograd
import torch


class ThresholdBinarizer(autograd.Function):
    
    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float):
        mask = (torch.sigmoid(inputs) >= threshold).type(inputs.type())
        return mask
        
    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None, None

class PruningLinear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, in_pruning = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PruningLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # choose the input pruning or output pruning for fc1 (output) and fc2 (output)
        self.in_pruning = in_pruning
        self.param_threshold = 0.0
        if self.in_pruning:
            self.parm_mask_scores = torch.nn.Parameter(torch.Tensor(self.out_features), requires_grad=True)
        else:
            self.parm_mask_scores = torch.nn.Parameter(torch.Tensor(self.in_features), requires_grad=True)
        torch.nn.init.constant_(self.parm_mask_scores, val=0.0)
        self.pruned_model = False

    def pruning_parameter(self):
        if not self.pruned_model:
            weight_mask, bias_mask = self.get_mask()
            
            if self.in_pruning:
                new_weight = torch.masked_select(self.weight, weight_mask.bool()).view(-1, self.in_features)
            else:
                new_weight = torch.masked_select(self.weight, weight_mask.bool()).view(self.out_features, -1)
            new_bias = torch.masked_select(self.bias, bias_mask.bool())
            
            self.weight = torch.nn.Parameter(new_weight)
            self.bias = torch.nn.Parameter(new_bias)
            self.pruned_model = True
        
    def set_threshold(self, param_threshold):
        self.param_threshold = param_threshold

    def get_param_score(self):
        if self.in_pruning:
            unroll_parm_mask_scores = torch.repeat_interleave(self.parm_mask_scores.unsqueeze(-1), self.in_features, dim=-1)
        else:
            unroll_parm_mask_scores = torch.repeat_interleave(self.parm_mask_scores.unsqueeze(0), self.out_features, dim=0)            
        return unroll_parm_mask_scores.type(self.weight.type())

    def get_param_number(self):
        return self.get_mask()

    def get_mask(self):
        _param_mask = ThresholdBinarizer.apply(self.parm_mask_scores, self.param_threshold)
        
        if self.in_pruning:
            _mask = torch.repeat_interleave(_param_mask.unsqueeze(-1), self.in_features, dim=-1)
        else:
            _mask = torch.repeat_interleave(_param_mask.unsqueeze(0), self.out_features, dim=0)
        
        if self.in_pruning:
            _out_mask = _param_mask
        else:
            _out_mask = torch.ones(self.out_features).to(_param_mask)
        return _mask.type(self.parm_mask_scores.type()), _out_mask.type(self.parm_mask_scores.type())

    def get_weight_bias(self):
        if self.pruned_model:
            return self.weight, self.bias
        else:
            weight_mask, bias_mask = self.get_mask()
            _weight = self.weight * weight_mask
            _bias = self.bias * bias_mask
            return _weight, _bias

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        _weight, _bias = self.get_weight_bias()
        return F.linear(input, _weight, _bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def upgrade_state_dict_named(self, state_dict, name):
        
        prefix = name + ".parm_mask_scores"
        if prefix not in state_dict:
            state_dict[prefix] = self.parm_mask_scores

class PruningHead(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PruningHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # pruning
        self.block_size = 32
        col_size = self.out_features // self.block_size
        row_size = self.in_features // self.block_size
        self.head_mask_scores = torch.nn.Parameter(torch.Tensor(col_size, row_size), requires_grad=True)
        torch.nn.init.constant_(self.head_mask_scores, val=0.0)

        self.head_threshold = 0.0
        self.pruned_model = False

    def set_threshold(self, head_threshold):
        self.head_threshold = head_threshold

    def pruning_parameter(self):
        if not self.pruned_model:
            weight_mask = self.get_mask()
            self.weight = torch.nn.Parameter(self.weight * weight_mask.bool())
            self.pruned_model = True

    def get_head_score(self):
        _head_out_score = torch.repeat_interleave(self.head_mask_scores, self.block_size, dim=0)
        _head_in_score = torch.repeat_interleave(_head_out_score, self.block_size, dim=-1)
        return _head_in_score.type(self.weight.type())
        
    def get_mask(self):
        _head_mask = ThresholdBinarizer.apply(self.head_mask_scores, self.head_threshold)

        _head_out_mask = torch.repeat_interleave(_head_mask, self.block_size, dim=0)
        _head_in_mask = torch.repeat_interleave(_head_out_mask, self.block_size, dim=-1)
        _mask = _head_in_mask
        return _mask.type(self.head_mask_scores.type())
        
    def get_weight_bias(self):
        if self.pruned_model:
            return self.weight, self.bias
        else:
            weight_mask = self.get_mask()
            _weight = self.weight * weight_mask
            return _weight, self.bias

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input: Tensor) -> Tensor:
        _weight, _bias = self.get_weight_bias()
        return F.linear(input, _weight, _bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + ".head_mask_scores"
        if prefix not in state_dict:
            state_dict[prefix] = self.head_mask_scores


class PruningMultiheadAttention(FairseqIncrementalDecoder):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        dictionary=None,
        q_noise=0.0,
        qn_block_size=8,
        # TODO: pass in config rather than string.
        # config defined in xformers.components.attention.AttentionConfig
        xformers_att_config: Optional[str] = None,
        xformers_blocksparse_layout: Optional[
            torch.Tensor
        ] = None,  # This should be part of the config
        xformers_blocksparse_blocksize: Optional[
            int
        ] = 16,  # This should be part of the config
    ):
        super().__init__(dictionary)

        xformers_att_config = utils.eval_str_dict(xformers_att_config)
        self.use_xformers = xformers_att_config is not None
        if self.use_xformers and not _xformers_available:
            raise ImportError("\n\n  Please install xFormers.")
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = PruningHead(self.kdim, embed_dim, bias=bias)
        self.v_proj = PruningHead(self.vdim, embed_dim, bias=bias)
        self.q_proj = PruningHead(embed_dim, embed_dim, bias=bias)
        self.out_proj = PruningHead(embed_dim, embed_dim, bias=bias)
        
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.beam_size = 1
        self.reset_parameters()

        self.onnx_trace = False
        self.skip_embed_dim_check = False

        self.pruned_model = False
        self.cache = {}

    def get_param_number(self):
        return {"k_proj": self.k_proj.get_mask(),
                "v_proj": self.v_proj.get_mask(),
                "q_proj": self.q_proj.get_mask(),
                "out_proj": self.out_proj.get_mask(),
        }

    def pruning_parameter(self):
        if not self.pruned_model:
            self.k_proj.pruning_parameter()
            self.v_proj.pruning_parameter()
            self.q_proj.pruning_parameter()
            self.out_proj.pruning_parameter()
            self.pruned_model = True

    def set_head_threshold(self, head_threshold):
        self.k_proj.set_threshold(head_threshold)
        self.v_proj.set_threshold(head_threshold)
        self.q_proj.set_threshold(head_threshold)
        self.out_proj.set_threshold(head_threshold)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        if not self.num_heads:
            return  query, None

        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        if not self.skip_embed_dim_check:
            assert (
                embed_dim == self.embed_dim
            ), f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert src_len, key_bsz == value.shape[:2]

        q_weight, q_bias = self.q_proj.get_weight_bias()
        k_weight, k_bias = self.k_proj.get_weight_bias()
        v_weight, v_bias = self.v_proj.get_weight_bias()
        out_weight, out_bias = self.out_proj.get_weight_bias() 

        return F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            torch.empty([0]),
            torch.cat((q_bias, k_bias, v_bias)),
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout_module.p,
            out_weight,
            out_bias,
            self.training or self.dropout_module.apply_during_inference,
            key_padding_mask,
            need_weights,
            attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=q_weight,
            k_proj_weight=k_weight,
            v_proj_weight=v_weight,
        )


    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
        

