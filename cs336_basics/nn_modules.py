import torch
import torch.nn as nn
import math
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weight = nn.Parameter(torch.empty(
            out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        self.W = nn.init.trunc_normal_(
            tensor=weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        emb = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.emb = nn.init.trunc_normal_(tensor=emb, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        rms = einsum(
            torch.pow(x, 2), 'batch seq_len dim -> batch seq_len') / x.size(-1) + self.eps
        return torch.sqrt(rms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = self._rms(x).unsqueeze(-1)
        x = x * self.g / rms
        result = x
        return result.to(in_dtype)


def silu(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.W1 = nn.Parameter(torch.empty(
            self.d_ff, self.d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(
            self.d_model, self.d_ff, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(
            self.d_ff, self.d_model, device=device, dtype=dtype))
        std = math.sqrt(2 / (self.d_ff + self.d_model))
        self.W1 = nn.init.trunc_normal_(
            tensor=self.W1, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W2 = nn.init.trunc_normal_(
            tensor=self.W2, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W3 = nn.init.trunc_normal_(
            tensor=self.W3, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_x = silu(x @ self.W1.T)
        w3_x = x @ self.W3.T
        return (silu_x * w3_x) @ self.W2.T


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        half_d = d_k // 2
        inv_freq = 1.0 / \
            (theta ** (torch.arange(0, half_d, dtype=dtype) / half_d))
        positions = torch.arange(max_seq_len, dtype=dtype).unsqueeze(1)
        angles = positions * inv_freq.unsqueeze(0)  # (max_seq_len, half_d)

        sin = torch.sin(angles).repeat_interleave(
            2, dim=-1)  # (max_seq_len, d_k)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)

        self.register_buffer("sin", sin.to(device))
        self.register_buffer("cos", cos.to(device))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_positions = token_positions.squeeze()
        selected_sin = self.sin.index_select(
            0, token_positions).view(*token_positions.shape, self.d_k)
        selected_cos = self.cos.index_select(
            0, token_positions).view(*token_positions.shape, self.d_k)

        x1, x2 = x[..., ::2], x[..., 1::2]
        sin, cos = selected_sin[..., ::2], selected_cos[..., ::2]

        rot_x = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return rot_x.flatten(-2)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max = x.max(dim=dim, keepdim=True).values
    x = x - max
    exp_x = torch.exp(x)
    exp_sum = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / exp_sum


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    d_k = K.shape[-1]
    attn_score = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        attn_score = attn_score.masked_fill(~mask, float('-inf'))
    attn_probs = torch.softmax(attn_score, dim=-1)
    return attn_probs @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 max_seq_len: int = 1000, with_rope: bool = False,
                 theta: float = 0, token_positions: torch.Tensor | None = None,
                 device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_kv = d_model
        self.W_q = nn.Parameter(torch.empty(
            self.d_kv, self.d_model, device=device, dtype=dtype))
        self.W_k = nn.Parameter(torch.empty(
            self.d_kv, self.d_model, device=device, dtype=dtype))
        self.W_v = nn.Parameter(torch.empty(
            self.d_kv, self.d_model, device=device, dtype=dtype))
        self.W_o = nn.Parameter(torch.empty(
            d_model, d_model, device=device, dtype=dtype))
        std = math.sqrt(2 / (self.d_model + self.d_kv))
        self.W_q = nn.init.trunc_normal_(
            tensor=self.W_q, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W_k = nn.init.trunc_normal_(
            tensor=self.W_k, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W_v = nn.init.trunc_normal_(
            tensor=self.W_v, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W_o = nn.init.trunc_normal_(
            tensor=self.W_o, mean=0, std=std, a=-3 * std, b=3 * std)
        self.device = device
        if with_rope:
            self.rope = RotaryPositionalEmbedding(
                theta, self.head_dim, max_seq_len, device, dtype)
            self.token_positions = token_positions
        else:
            self.rope = None

    def forward(self, x: torch.Tensor):
        q = x @ self.W_q.T
        k = x @ self.W_k.T
        v = x @ self.W_v.T

        B, N, _ = x.shape

        def split_heads(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        if self.rope:
            token_positions = self.token_positions if self.token_positions is not None else torch.arange(
                x.size(1), device=x.device)
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        mask = ~torch.triu(torch.ones(
            N, N, device=self.device, dtype=torch.bool), diagonal=1)

        attn = scaled_dot_product_attention(q, k, v, mask)
        attn_output = rearrange(
            attn, "batch num_head len head_dim -> batch len (num_head head_dim)")
        output = attn_output @ self.W_o.T
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_head: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model, num_head, max_seq_len, True, theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor):
        x_norm = self.ln1(x)
        attn = self.attn(x_norm)
        x = x + attn
        x_norm = self.ln2(x)
        ffn = self.ffn(x_norm)
        return x + ffn


class Transformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device=None,
            dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff,
                context_length, rope_theta,
                device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x