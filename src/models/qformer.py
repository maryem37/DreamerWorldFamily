"""Q-Former: compact learned query tokens for vision-language fusion."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class QFormerLayer(nn.Module):
    """Single Q-Former layer with query self-attention and cross-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.visual_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        visual_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q)

        q, _ = self.visual_cross_attn(queries, visual_features, visual_features)
        queries = self.norm2(queries + q)

        if text_features is not None:
            q, _ = self.text_cross_attn(queries, text_features, text_features)
            queries = self.norm3(queries + q)

        return self.norm4(queries + self.ffn(queries))


class QFormer(nn.Module):
    """Learnable query transformer bridging visual features and text embeddings."""

    def __init__(
        self,
        num_query_tokens: int = 32,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        vis_input_dim: int = 768,
        text_input_dim: int = 2048,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, hidden_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.vis_proj = nn.Linear(vis_input_dim, hidden_dim)
        self.text_proj = nn.Linear(text_input_dim, hidden_dim)
        self.layers = nn.ModuleList(QFormerLayer(hidden_dim, num_heads) for _ in range(num_layers))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, visual_features: torch.Tensor, text_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        if visual_features.dim() == 4:
            # (B, N, P, D) -> aggregate per candidate as tokens.
            visual_features = visual_features.mean(dim=2)
        B = visual_features.shape[0]
        vis = self.vis_proj(visual_features)
        text = self.text_proj(text_embeds) if text_embeds is not None else None
        queries = self.query_tokens.expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, vis, text)
        return self.norm(queries)
