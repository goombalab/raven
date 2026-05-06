# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules.feature_map import ReLUFeatureMap, SwishFeatureMap, T2RFeatureMap
from fla.modules.layernorm import rms_norm_linear
from fla.ops.gsa import chunk_gsa as chunk_raven, fused_recurrent_gsa as fused_recurrent_raven


from fla.ops.utils.index import prepare_lens_from_mask


from fla.modules import FusedRMSNormGated , RMSNorm, RotaryEmbedding  #Added for RoPE

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache

def _max_offset(seqlen_offset):
    if seqlen_offset is None:
        return 0
    if isinstance(seqlen_offset, int):
        # scalar offset, nothing fancy
        return seqlen_offset
    if isinstance(seqlen_offset, torch.Tensor):
        # tensor of offsets -> take max
        return int(seqlen_offset.max().item())
    # list/tuple/other iterables
    return max(seqlen_offset)


class RavenAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 1.,
        expand_v: float = 1.,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        num_slots: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 8,
        feature_map: str = 'swish',
        use_output_gate: bool = False,
        use_norm: bool = True,
        layer_idx: Optional[int] = None,
        scale: Optional[float] = 1.,
        decay_type: str = 'Mamba2',
        topk: int = 32,
        bias_rmm: bool = False,
        add_gumbel_noise: bool = True,
        router_score: str = 'sigmoid',
        router_type: str = 'lin',
        use_rope: bool = False,
        **kwargs
    ) -> RavenAttention:
        super().__init__()

        self.mode = mode
        self.decay_type = decay_type
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        self.topk = topk
        self.use_output_gate = use_output_gate
        self.use_rope = use_rope
        self.gate_logit_normalizer = gate_logit_normalizer
        self.use_norm = use_norm
        self.scale = scale
        self.rope_theta = 10000.

        ##  For Router Design
        self.bias_rmm = bias_rmm  # For no gumbel router with bias
        self.add_gumbel_noise = add_gumbel_noise  # For no gumbel router with bias
        self.router_score = router_score
        self.router_type = router_type

        if num_slots is None:
            num_slots = self.head_k_dim
        self.num_slots = num_slots

        self.layer_idx = layer_idx

        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.register_module('feature_map', None)
        if feature_map == 'swish':
            self.feature_map = SwishFeatureMap()
        elif feature_map == 'relu':
            self.feature_map = ReLUFeatureMap()
        elif feature_map == 't2r':
            self.feature_map = T2RFeatureMap(self.head_k_dim, self.head_k_dim)
        else:
            raise NotImplementedError(f"Feature map `{feature_map}` is not supported now.")

        ## ===== QKV Proj =====
        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim_per_group, bias=False)

        ## ===== Forget Gate/ Decay =====
        if self.decay_type == 'Mamba2':
            # Decay of Mamba2
            self.a_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
            A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True
            # hard coded for now
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias._no_weight_decay = True

        elif self.decay_type == 'GLA':
            self.f_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.num_slots, bias=False)

        ## ===== Router  =====
        if self.bias_rmm:
            self.r_bias =  nn.Parameter(torch.empty( ( self.num_heads , self.num_slots) , dtype=torch.float32))

        if  self.router_type == 'lin':
            self.r_proj = nn.Linear(self.hidden_size, self.num_heads * self.num_slots , bias=False)

        elif  self.router_type == 'mlp':
            self.r_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.num_heads * self.num_slots, bias=False) )

        self.score_fn = (
            (lambda x: torch.sigmoid(x))
            if self.router_score == "sigmoid"
            else (lambda x: torch.softmax(x, dim=-1))
        )


        ## ===== RoPE  =====
        if self.use_rope:
            self.rotary = RotaryEmbedding(dim=self.head_k_dim, base=self.rope_theta)   # Added for RoPE

        ## ===== QK Norm  =====
        self.q_norm = RMSNorm(self.head_k_dim, elementwise_affine, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_k_dim, elementwise_affine, eps=norm_eps)

        ## ===== Output Layer  =====
        if self.use_output_gate:
            self.o_gate_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.g_norm = RMSNorm(self.hidden_size, elementwise_affine, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        seqlen_offset, max_seqlen = 0, q_len # Added for RoPE

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)   # Added for RoPE
            last_state = past_key_values[self.layer_idx]
            max_seqlen = q_len + seqlen_offset  # Added for RoPE

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0) # to deliminate the offsets of padding tokens
            seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]   # Added for RoPE
            max_seqlen = max(q_len, q_len + _max_offset(seqlen_offset))    # Added for RoPE

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_k_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
        router = rearrange(self.r_proj(hidden_states) , '... (h m) -> ... h m', m=self.num_slots)

        if self.decay_type == 'Mamba2':
            # Build Mamba2 Decay
            f = (- self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)).unsqueeze(-1)
        elif self.decay_type == 'GLA':
            f = rearrange(self.f_proj(hidden_states) , '... (h m) -> ... h m', m=self.num_slots)
            f =  F.logsigmoid(f) / self.gate_logit_normalizer

        if self.feature_map is not None:
            q, k = map(lambda x: self.feature_map(x), (q, k))

        # QK Norm
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rope:
            assert batch_size == 1, "RoPE is not supported for batch size > 1"
            max_seqlen = max(max_seqlen, 8192)  # Added for RoPE
            q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)  #Added for RoPE

        # V Feature map
        v = F.silu(v)

        # Build RMM Router
        if self.add_gumbel_noise:
            if self.training:
                router = router - torch.empty_like(router).exponential_().log()

        orig_scores = self.score_fn(router)
        if self.bias_rmm:
            scores = orig_scores + self.r_bias.float()
        else:
            scores = orig_scores

        route_idx = scores.topk(self.topk, dim=-1).indices
        topk_weights =  torch.gather(orig_scores, dim=-1, index=route_idx)

        if self.router_score == 'sigmoid':
            topk_weights /= (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        s_multihot = torch.zeros_like(router).scatter_(-1, route_idx, topk_weights.to(router.dtype))

        f = (f*s_multihot).to(q.dtype)
        s = (1-f.exp()).to(q.dtype)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if self.num_kv_groups > 1:
            k, v, f, s = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_kv_groups), (k, v, f, s))

        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_raven(
                q=q,
                k=k,
                v=v,
                s=s,
                g=f,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                scale=self.scale,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_raven(
                q=q,
                k=k,
                v=v,
                s=s,
                g=f,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                scale=self.scale,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=None,
                layer_idx=self.layer_idx,
                offset=q_len
            )


        if self.use_output_gate:
            gate_out = rearrange(self.o_gate_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(F.silu(o), gate_out)
            o = rearrange(o, '... h d -> ... (h d)')
            o = self.o_proj(o)
        else:
            o = rearrange(o, '... h d -> ... (h d)')
            o = rms_norm_linear(F.silu(o), self.g_norm.weight, self.g_norm.bias, self.o_proj.weight, self.o_proj.bias)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
