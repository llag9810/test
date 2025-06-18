import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        self.W = nn.init.trunc_normal_(tensor=weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        emb = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
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
        rms = einsum(torch.pow(x, 2), 'batch seq_len dim -> batch seq_len') / x.size(-1) + self.eps
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
        self.W1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        std = math.sqrt(2 / (self.d_ff + self.d_model))
        self.W1 = nn.init.trunc_normal_(tensor=self.W1, mean=0, std=std, a=-3 * std, b = 3 * std)
        self.W2 = nn.init.trunc_normal_(tensor=self.W2, mean=0, std=std, a=-3 * std, b = 3 * std)
        self.W3 = nn.init.trunc_normal_(tensor=self.W3, mean=0, std=std, a=-3 * std, b = 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_x = silu(x @ self.W1.T)
        w3_x = x @ self.W3.T
        return (silu_x * w3_x) @ self.W2.T
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        half_d = d_k // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half_d, dtype=torch.float32) / half_d))
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        angles = positions * inv_freq.unsqueeze(0)  # (max_seq_len, half_d)

        sin = torch.sin(angles).repeat_interleave(2, dim=-1)  # (max_seq_len, d_k)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)

        self.register_buffer("sin", sin.to(device))
        self.register_buffer("cos", cos.to(device))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        selected_sin = self.sin.index_select(0, token_positions).view(*token_positions.shape, self.d_k)
        selected_cos = self.cos.index_select(0, token_positions).view(*token_positions.shape, self.d_k)

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

