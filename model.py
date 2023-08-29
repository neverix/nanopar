from dataclasses import dataclass
from typing import Optional
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
    vocab_size: int = -1
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_frequencies(dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    theta_numerator = torch.arange(0, dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / dim)).to(device)
    # Construct the positions (the "m" parameter)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def reshape_for_broadcasting(freqs_complex: torch.Tensor, x: torch.Tensor):
    # freqs_complex: (Seq_Len, Head_Dim/2)
    # X: (B, Seq_Len, H, Head_Dim/2)
    # So we need to add the batch dimension and the head dimension
    return freqs_complex.unsqueeze(0).unsqueeze(2)


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = reshape_for_broadcasting(freqs_complex, x_complex)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
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
        x[:, :, :, None, :]  # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        .expand(
            batch_size, seq_len, n_kv_heads, n_rep, head_dim
        )  # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .reshape(
            batch_size, seq_len, n_kv_heads * n_rep, head_dim
        )  # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        ).to(args.device)
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        ).to(args.device)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        batch_size, seq_len, _ = x.shape  # (B, Seq_Len, Dim)

        xq = self.wq(x)  # (B, Seq_Len, Dim) -> (B, Seq_Len, H_Q * Head_Dim)
        xk = self.wk(x)  # (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        xv = self.wv(x)  # (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)

        xq = xq.view(
            batch_size, seq_len, self.n_heads_q, self.head_dim
        )  # (B, Seq_Len, H_Q * Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim)
        xk = xk.view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        )  # (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xv = xv.view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        )  # (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)

        # (B, Seq_Len, H_Q, Head_Dim) --> (B, Seq_Len, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, Seq_Len, H_KV, Head_Dim) --> (B, Seq_Len, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        keys = self.cache_k[
            :batch_size, : start_pos + seq_len
        ]  # (B, Seq_Len, H_KV, Head_Dim)
        values = self.cache_v[
            :batch_size, : start_pos + seq_len
        ]  # (B, Seq_Len, H_KV, Head_Dim)

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.
        keys = repeat_kv(
            keys, self.n_rep
        )  # (B, Seq_Len, H_KV, Head_Dim) --> (B, Seq_Len, H_Q, Head_Dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (B, Seq_Len, H_KV, Head_Dim) --> (B, Seq_Len, H_Q, Head_Dim)

        xq = xq.transpose(
            1, 2
        )  # (B, Seq_Len, H_Q, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        keys = keys.transpose(
            1, 2
        )  # (B, Seq_Len, H_Q, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        values = values.transpose(
            1, 2
        )  # (B, Seq_Len, H_Q, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)

        # (B, H_Q, Seq_Len, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len) -> (B, H_Q, Seq_Len, Seq_Len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply the mask if specified
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(
            scores, values
        )  # (B, H_Q, Seq_Len, Seq_Len) @ (B, H_Q, Seq_Len, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )  # (B, H_Q, Seq_Len, Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim) -> (B, Seq_Len, Dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

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


class TransformerBlock(nn.Module):

    def __init__(self, layer_id: int, args: ModelArgs):
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = Attention(args)

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )

        self.layer_id = layer_id
        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor, mask: Optional[torch.Tensor]):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex, mask
        )
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    
class Transformer(nn.Module):

    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, self.vocab_size, bias=False)

        self.freqs_comlex = precompute_frequencies(self.params.dim // self.params.n_heads, self.params.max_seq_len* 2, device=self.params.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        mask = None
        if seq_len > 1:
            # Only applied to the prompt, because the successive tokens will be predicted using KV-Cache, so no need to mask
            mask = torch.full(
                (1, 1, seq_len, seq_len), float("-inf"), device=h.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output