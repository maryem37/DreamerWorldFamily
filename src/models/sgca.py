"""State-Grounded Cross-Attention for Adaptive Text Dreamer."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGCALayer(nn.Module):
    """Single cosine-similarity cross-attention layer."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        self.norm_q = nn.LayerNorm(self.head_dim)
        self.norm_k = nn.LayerNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Nq, D = queries.shape
        _, Nk, _ = keys.shape
        Q = self.W_Q(queries).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(keys).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(values).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        Q_norm = F.normalize(self.norm_q(Q), dim=-1)
        K_norm = F.normalize(self.norm_k(K), dim=-1)
        attn = torch.matmul(Q_norm, K_norm.transpose(-2, -1))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, Nq, D)
        out = self.layer_norm(queries + self.W_O(out))
        out = out + self.ffn(out)
        return out, attn.detach()


class StateGroundedCrossAttention(nn.Module):
    """Multi-layer SGCA stack."""

    def __init__(self, hidden_dim: int, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(SGCALayer(hidden_dim, num_heads) for _ in range(num_layers))

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = queries
        attention_maps = []
        for layer in self.layers:
            x, attn = layer(x, keys, values)
            attention_maps.append(attn)
        return x, attention_maps
