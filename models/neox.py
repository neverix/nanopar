import os
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from typing import Optional, Tuple
from socket import gethostname

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from apex.transformer import parallel_state
from apex.transformer import tensor_parallel
from apex.transformer.pipeline_parallel import get_forward_backward_func, build_model
from apex.transformer.pipeline_parallel.utils import (
    average_losses_across_data_parallel_group,
    setup_microbatch_calculator,
    _reconfigure_microbatch_calculator,
)

from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
from apex.optimizers.fused_adam import FusedAdam


import torch._dynamo

torch._dynamo.allow_in_graph(rearrange)


def identity(x):
    return x


torch._dynamo.config.cache_size_limit = 1000


@dataclass
class NeoXArgs:
    hidden_size: int = 512
    vocab_size: int = 50304
    layer_norm_eps: float = 1e-6
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    rotary_pct: float = 0.125
    max_position_embeddings: int = 2048
    rotary_emb_base: float = 10000.0
    intermediate_size: int = 2048
    hidden_act: str = "gelu"
    use_parallel_residual: bool = True
    device: Optional[str] = None
    use_sp: bool = False


def precompute_freqs(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(freqs_cis)


def reshape_for_broadcast(freqs, x_shape):
    ndim = len(x_shape)
    assert 0 <= 1 < ndim
    assert freqs.shape == (
        x_shape[0],
        x_shape[-2],
        x_shape[-1],
    ), f"{freqs.shape=} not compatible with {x_shape=}"
    shape = [d if i == 0 or i >= ndim - 2 else 1 for i, d in enumerate(x_shape)]
    return freqs.view(*shape)


def cmul(x, y):
    return torch.stack(
        [
            x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1],
            x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0],
        ],
        dim=-1,
    )


@torch.compile
def apply_rotary_emb(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:    
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_out = cmul(x_, freqs).flatten(3)
    return x_out.type_as(x)


def add_bias(x: Tuple[torch.tensor, Optional[torch.Tensor]]):
    x, bias = x
    if bias is not None:
        x = x + bias
    return x


class GPTNeoXAttention(nn.Module):
    def __init__(self, args: NeoXArgs, dtype: torch.dtype = torch.float32):
        super().__init__()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert args.num_attention_heads % tp_size == 0
        self.n_local_heads = args.num_attention_heads // tp_size
        self.head_dim = args.hidden_size // args.num_attention_heads

        # separate so it doesn't break with model paralellism
        # self.query_key_value = tensor_parallel.ColumnParallelLinear(
        #     args.hidden_size,
        #     args.hidden_size * 3,
        #     bias=True,
        #     gather_output=False,
        #     params_dtype=dtype,
        #     sequence_parallel_enabled=True,
        #     no_async_tensor_model_parallel_allreduce=True,
        # )
        self.query, self.key, self.value = (tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            args.hidden_size,
            bias=True,
            gather_output=False,
            params_dtype=dtype,
            sequence_parallel_enabled=True,
            no_async_tensor_model_parallel_allreduce=True,
        ) for _ in "QKV")

        self.dense = tensor_parallel.RowParallelLinear(
            args.hidden_size,
            args.hidden_size,
            bias=True,
            input_is_parallel=True,
            params_dtype=dtype,
            sequence_parallel_enabled=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        kv_freqs: torch.Tensor,
        q_freqs: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        seqlen, bsz, _ = x.shape

        x = x.contiguous()
        xq, xk, xv = map(add_bias, (self.query(x), self.key(x), self.value(x)))
        xq, xk, xv = (
            rearrange(x, "s b (nh hd) -> s b nh hd", nh=self.n_local_heads)
            for x in (xq, xk, xv))

        xk = apply_rotary_emb(xk, freqs=kv_freqs)
        xq = apply_rotary_emb(xq, freqs=q_freqs)

        xk = rearrange(xk, "s b nh hd -> b nh s hd")
        xv = rearrange(xv, "s b nh hd -> b nh s hd")
        xq = rearrange(xq, "s b nh hd -> b nh s hd")

        causal = mask is None
        with torch.backends.cuda.sdp_kernel(
            enable_math=causal, enable_flash=True, enable_mem_efficient=False
        ):
            output = F.scaled_dot_product_attention(
                xq, xk, xv, is_causal=causal, attn_mask=mask
            )
            output = rearrange(output, "b nh s hd -> s b (nh hd)").contiguous()
            output = add_bias(self.dense(output))
            return output


class GPTNeoXMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            params_dtype=dtype,
            sequence_parallel_enabled=True,
            no_async_tensor_model_parallel_allreduce=True,
        )

        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
            params_dtype=dtype,
            sequence_parallel_enabled=True,
        )

    def forward(self, x):
        return add_bias(self.dense_4h_to_h(F.gelu(add_bias(self.dense_h_to_4h(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: NeoXArgs, dtype: torch.dtype):
        super().__init__()
        self.attention = GPTNeoXAttention(args, dtype=dtype)
        self.mlp = GPTNeoXMLP(
            dim=args.hidden_size,
            hidden_dim=args.intermediate_size,
            dtype=dtype,
        )
        self.layer_id = layer_id
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps
        )
        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        kv_freqs: torch.Tensor,
        q_freqs: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        x0 = self.input_layernorm(x)
        x1 = self.post_attention_layernorm(x)
        x0_attn = self.attention(x0, start_pos, kv_freqs, q_freqs, mask)
        x1_mlp = self.mlp(x1)
        return (
            x
            + x0_attn
            + x1_mlp
        )


class SplitNeoX(nn.Module):
    def __init__(self, args: NeoXArgs, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.pp_world = parallel_state.get_pipeline_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.tp_world = parallel_state.get_tensor_model_parallel_world_size()

        curr_rank_layers = args.num_hidden_layers // self.pp_world
        start_layer = self.pp_rank * curr_rank_layers

        self.layers = nn.ModuleList(
            [
                TransformerBlock(i + start_layer, args, dtype)
                for i in range(curr_rank_layers)
            ]
        )
        self.freqs = precompute_freqs(args.hidden_size // args.num_attention_heads, args.max_position_embeddings * 2)

        if self.pp_rank == 0:
            self.embed_in = tensor_parallel.VocabParallelEmbedding(
                args.vocab_size, args.hidden_size, params_dtype=dtype
            )

        if self.pp_rank == self.pp_world - 1:
            self.embed_out = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                args.vocab_size,
                bias=False,
                params_dtype=dtype,
                gather_output=False,
                sequence_parallel_enabled=True,
                no_async_tensor_model_parallel_allreduce=True,
            )
            self.final_layer_norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

        self.args = args

    # factored out for torch.compile
    # @torch.compile
    def transformer_block(self, x, start_pos, kv_freqs, q_freqs, mask):
        for layer in self.layers:
            x = layer(x, start_pos, kv_freqs, q_freqs, mask)
        return x

    def forward(self, tokens_or_hidden_state: torch.Tensor, start_pos: int = 0):
        if self.pp_rank == 0:
            x = self.embed_in(tokens_or_hidden_state)
            x = rearrange(x, "b s d -> s b d")
            x = tensor_parallel.mappings.scatter_to_sequence_parallel_region(x)
        else:
            x = tokens_or_hidden_state

        seq_len, batch_size, _ = x.shape
        total_seq_len = seq_len * self.tp_world

        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(x)

        kv_freqs = self.freqs[start_pos : start_pos + total_seq_len].to(x.device)
        sp_n_queries = seq_len // self.tp_world
        q_freqs = kv_freqs

        n_heads = self.args.num_attention_heads
        head_dim = self.args.hidden_size // n_heads
        kv_shape = (total_seq_len, batch_size, n_heads, head_dim // 2, 2)
        q_shape = (total_seq_len, batch_size, n_heads, head_dim // 2, 2)
        kv_freqs = reshape_for_broadcast(kv_freqs, kv_shape).to(x.device)
        q_freqs = reshape_for_broadcast(q_freqs, q_shape).to(x.device)

        x = self.transformer_block(x, start_pos, kv_freqs, q_freqs, mask=None)

        if self.pp_rank == self.pp_world - 1:
            x = self.final_layer_norm(x)
            x = add_bias(self.embed_out(x))
            return x
        else:
            return x


class PipelineStage(nn.Module):
    input_tensors: Optional[List[torch.Tensor]] = None

    def __init__(self, module):
        super().__init__()
        self.input_tensors = None
        self.wrapped = module

    def set_input_tensor(self, tensor: List[torch.Tensor]):
        self.input_tensors = tensor

    def forward(self, *x, **kwargs):
        if parallel_state.is_pipeline_first_stage():
            inputs = x
        else:
            inputs = self.input_tensors
        return self.wrapped(*inputs, **kwargs)


def neox_model_provider(args, **_):
    return PipelineStage(SplitNeoX(args,
                                   dtype=torch.bfloat16))
