from apex.normalization import FusedRMSNorm
from apex.transformer import tensor_parallel
from apex.transformer import parallel_state
from dataclasses import dataclass
from typing import Optional, List, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # Later set in the build method
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: Optional[str] = None
    use_sp: bool = False


class ColumnParallelLinear(tensor_parallel.ColumnParallelLinear):
    def __init__(self, in_dim, out_dim, dtype=torch.float32, use_sp: bool = False):
        super().__init__(
            in_dim,
            out_dim,
            bias=False,
            gather_output=False,
            params_dtype=dtype,
            sequence_parallel_enabled=use_sp,
            no_async_tensor_model_parallel_allreduce=use_sp,
        )
    
    def forward(self, x):
        # ignore bias
        result, _ = super().forward(x)
        return result


class RowParallelLinear(tensor_parallel.RowParallelLinear):
    def __init__(self, in_dim, out_dim, dtype=torch.float32, use_sp: bool = False):
        super().__init__(
            in_dim,
            out_dim,
            bias=False,
            input_is_parallel=True,
            params_dtype=dtype,
            sequence_parallel_enabled=use_sp
        )
    
    def forward(self, x):
        # ignore bias
        result, _ = super().forward(x.contiguous())
        return result


class RMSNorm(FusedRMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6,
                 dtype: torch.dtype = torch.bfloat16, use_sp: bool = False):
        super().__init__(dim, eps=eps)
        self.weight.data = self.weight.data.to(dtype)
        if use_sp:
            self.weight.sequence_parallel_enabled = use_sp

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return super().forward(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real n umber
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs, dtype: torch.dtype,
                 use_sp: bool=True, use_sdp: bool = True):
        super().__init__()
        
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim, self.n_heads_q * self.head_dim, dtype=dtype, use_sp=use_sp)
        self.wk = ColumnParallelLinear(
            args.dim, self.n_kv_heads * self.head_dim, dtype=dtype, use_sp=use_sp)
        self.wv = ColumnParallelLinear(
            args.dim, self.n_kv_heads * self.head_dim, dtype=dtype, use_sp=use_sp)
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim, args.dim, dtype=dtype, use_sp=use_sp)
        
        self.use_sdp = use_sdp

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None
    ):
        # TODO fix this
        seq_len, batch_size, _ = x.shape  # (Seq_Len, B, Dim)

        # (Seq_Len, B, Dim) -> (Seq_Len, B, H_Q * Head_Dim)
        xq = self.wq(x)
        # (Seq_Len, B, Dim) -> (Seq_Len, B, H_KV * Head_Dim)
        xk = self.wk(x)
        # (Seq_Len, B, Dim) -> (Seq_Len, B, H_KV * Head_Dim)
        xv = self.wv(x)
        # Dimension gets multiplied by 2 during SP?..
        seq_len = xq.shape[0]
        
        # (Seq_Len, B, H_Q * Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim)
        xq = xq.transpose(0, 1).reshape(batch_size, seq_len, self.n_heads_q // self.tp_size, self.head_dim)
        # (Seq_Len, B, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xk = xk.transpose(0, 1).reshape(batch_size, seq_len, self.n_kv_heads // self.tp_size, self.head_dim)
        # (Seq_Len, B, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xv = xv.transpose(0, 1).reshape(batch_size, seq_len, self.n_kv_heads // self.tp_size, self.head_dim)

        # (B, Seq_Len, H_Q, Head_Dim) --> (B, Seq_Len, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, Seq_Len, H_KV, Head_Dim) --> (B, Seq_Len, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        if kv_cache is None:
            # (B, Seq_Len_KV, H_KV, Head_Dim)
            keys = xk
            # (B, Seq_Len_KV, H_KV, Head_Dim)
            values = xv
        else:
            if len(kv_cache) == 0:
                # (B, Seq_Len_KV)
                index = torch.zeros((batch_size, self.n_heads_q // self.tp_size, seq_len),
                                                 dtype=torch.long, device=x.device)
                # (B, Seq_Len_KV, H_KV, Head_Dim)
                # (B, Seq_Len_KV, H_KV, Head_Dim)
                # (B, Seq_Len_KV)
                kv_cache = (xk, xv, index)
                keys, values = xk, xv
            else:
                index = kv_cache[2]
                index += 1
                keys = kv_cache[0]
                keys.scatter_(1, index[:, :, None, None], xk)
                values = kv_cache[1]
                values.scatter_(1, index[:, :, None, None], xv)

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # (B, Seq_Len, H_Q, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        if (not self.use_sdp) or kv_cache is not None:
            if not self.use_sdp:
                raise NotImplementedError("Stop using vanilla attention. Get some help.")
            
            # (B, H_Q, Seq_Len, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, Seq_Len, Seq_Len_KV)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores - (index.unsqueeze(-1).to(scores)
                               >= torch.arange(scores.shape[-1]).unsqueeze(-2).to(scores)) * torch.inf
            # (B, H_Q, Seq_Len, Seq_Len_KV) -> (B, H_Q, Seq_Len, Seq_Len_KV)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)

            # (B, H_Q, Seq_Len, Seq_Len_KV) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
            output = torch.matmul(scores, values)
        else:
            with torch.backends.cuda.sdp_kernel(
                enable_math=False, enable_flash=True, enable_mem_efficient=False
            ):
                output = F.scaled_dot_product_attention(xq, keys, values, is_causal=True)
        # (B, H_Q, Seq_Len, Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim) -> (B, Seq_Len, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        # (B, Seq_Len, Dim) -> ((Seq_Len, B, Dim)
        output = output.transpose(0, 1)
        return self.wo(output) # (Seq_Len, B, Dim) -> (Seq_Len, B, Dim)


class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        dtype: torch.dtype,
        use_sp: bool
    ):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = ColumnParallelLinear(args.dim, hidden_dim, dtype=dtype, use_sp=use_sp)
        self.w2 = RowParallelLinear(hidden_dim, args.dim, dtype=dtype, use_sp=use_sp)
        self.w3 = ColumnParallelLinear(args.dim, hidden_dim, dtype=dtype, use_sp=use_sp)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs, dtype: torch.dtype, use_sp: bool = False):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args, dtype, use_sp)
        self.feed_forward = FeedForward(args, dtype, use_sp)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps,
                                      dtype=dtype, use_sp=use_sp)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps,
                                dtype=dtype, use_sp=use_sp)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor,
                kv_cache: Optional[Dict[str, torch.Tensor]] = None):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        attn_out = self.attention(
            self.attention_norm(x), start_pos, freqs_complex,
            kv_cache=kv_cache
        )
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + attn_out
        
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        mlp_out = self.feed_forward(self.ffn_norm(h))
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + mlp_out
        return out

    
class Transformer(nn.Module):

    def __init__(self, args: ModelArgs, dtype: torch.dtype, use_sp: bool):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"
        self.use_sp = use_sp
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.pp_world = parallel_state.get_pipeline_model_parallel_world_size()
        self.tp_world = parallel_state.get_tensor_model_parallel_world_size()

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        if self.pp_rank == 0:
            self.tok_embeddings = tensor_parallel.VocabParallelEmbedding(
                self.vocab_size, args.dim, params_dtype=dtype)

        self.layers = nn.ModuleDict()
        n_layers = args.n_layers
        layers_per_pp = n_layers // self.pp_world
        self.layers_start = layers_per_pp * self.pp_rank
        self.layers_end = layers_per_pp * (self.pp_rank + 1)
        for layer_id in range(self.layers_start, self.layers_end):
            self.layers[str(layer_id)] = EncoderBlock(args, dtype, use_sp=use_sp)

        if self.pp_rank == self.pp_world - 1:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=torch.float32, use_sp=use_sp)
            self.output = ColumnParallelLinear(
                args.dim, self.vocab_size, dtype=torch.float32, use_sp=use_sp)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len * 2
                                                              * (self.tp_world if use_sp else 1),
                                                              device=self.args.device)

    def forward(self,
                tokens: torch.Tensor,
                start_pos: int = 0,
                kv_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None):
        if self.pp_rank == 0:
            # (B, Seq_Len) -> (B, Seq_Len, Dim)
            h = self.tok_embeddings(tokens)
            # (B, Seq_Len, Dim) -> (Seq_Len, B, Dim)
            h = h.transpose(0, 1)
            if self.use_sp:
                h = tensor_parallel.scatter_to_sequence_parallel_region(h)
        else:
            h = tokens
        
        # (B, Seq_Len)
        seq_len, batch_size, *_ = h.shape

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos
                                           + seq_len
                                           * (self.tp_world if self.use_sp else 1)]
        
        # Consecutively apply all the encoder layers
        for layer_id in range(self.layers_start, self.layers_end):
            if kv_cache is None:
                cache = None
            else:
                i = layer_id - self.layers_start
                kv_cache = kv_cache + [tuple()]
                cache = kv_cache[i]
            layer_id = str(layer_id)
            layer = self.layers[layer_id]
            h = layer(h, start_pos, freqs_complex,
                      kv_cache=cache)
        if self.pp_rank == self.pp_world - 1:
            h = self.norm(h.float())
            output = self.output(h.float())
        else:
            output = h
        return output


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
        return self.wrapped(*inputs, **kwargs).contiguous()


def llama_model_provider(args, **_):
    return PipelineStage(Transformer(args,
                                     dtype=torch.bfloat16,
                                    use_sp=args.use_sp))
