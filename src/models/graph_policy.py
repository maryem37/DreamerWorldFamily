"""Graph-based navigation policy inspired by DUET."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAwareSelfAttention(nn.Module):
    """Self-attention with spatial distance bias."""

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_e = nn.Linear(1, num_heads, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor, distances: Optional[torch.Tensor]) -> torch.Tensor:
        B, K, D = nodes.shape
        H = self.num_heads
        Q = self.W_q(nodes).view(B, K, H, self.head_dim).transpose(1, 2)
        K_ = self.W_k(nodes).view(B, K, H, self.head_dim).transpose(1, 2)
        V = self.W_v(nodes).view(B, K, H, self.head_dim).transpose(1, 2)
        attn = torch.matmul(Q, K_.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if distances is not None:
            attn = attn + self.W_e(distances.unsqueeze(-1)).permute(0, 3, 1, 2)
        out = torch.matmul(F.softmax(attn, dim=-1), V).transpose(1, 2).contiguous().view(B, K, D)
        return self.norm(nodes + self.W_o(out))


class GraphNavigationPolicy(nn.Module):
    """Instruction-conditioned graph policy that scores candidate nodes."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        instr_dim: Optional[int] = None,
    ):
        super().__init__()
        instr_dim = instr_dim or hidden_dim
        self.instr_proj = nn.Linear(instr_dim, hidden_dim) if instr_dim != hidden_dim else nn.Identity()
        self.cross_attn_layers = nn.ModuleList(
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(num_layers)
        )
        self.cross_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.gasa_layers = nn.ModuleList(GraphAwareSelfAttention(hidden_dim, num_heads) for _ in range(num_layers))
        self.ffns = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_layers)
        )
        self.action_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(
        self,
        node_features: torch.Tensor,
        instruction_embeds: torch.Tensor,
        graph_edges: torch.Tensor,
        graph_distances: torch.Tensor,
    ) -> torch.Tensor:
        del graph_edges
        nodes = node_features
        instr = self.instr_proj(instruction_embeds)
        for i, cross_attn in enumerate(self.cross_attn_layers):
            attended, _ = cross_attn(nodes, instr, instr)
            nodes = self.cross_norms[i](nodes + attended)
            nodes = self.gasa_layers[i](nodes, graph_distances)
            nodes = nodes + self.ffns[i](nodes)
        return self.action_head(nodes).squeeze(-1)
