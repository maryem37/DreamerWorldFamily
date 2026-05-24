"""Adaptive Text Dreamer (ATD) for vision-and-language navigation."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_policy import GraphNavigationPolicy
from .qformer import QFormer
from .sgca import StateGroundedCrossAttention


@dataclass
class ATDConfig:
    num_query_tokens: int = 32
    qformer_hidden_dim: int = 768
    qformer_num_layers: int = 6
    qformer_num_heads: int = 12
    llm_hidden_dim: int = 2048
    llm_name: str = "google/flan-t5-xl"
    freeze_llm: bool = True
    sgca_num_layers: int = 4
    sgca_hidden_dim: int = 512
    graph_hidden_dim: int = 768
    graph_num_layers: int = 4
    batch_size: int = 2
    lr: float = 1e-5
    weight_decay: float = 0.05
    lambda_bc: float = 1.0


class LeftBrain(nn.Module):
    """State-estimation branch."""

    def __init__(self, config: ATDConfig):
        super().__init__()
        self.config = config
        self.qformer = QFormer(
            num_query_tokens=config.num_query_tokens,
            hidden_dim=config.qformer_hidden_dim,
            num_layers=config.qformer_num_layers,
            num_heads=config.qformer_num_heads,
            text_input_dim=config.llm_hidden_dim,
        )
        self.proj = nn.Linear(config.qformer_hidden_dim, config.llm_hidden_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(config.llm_hidden_dim, config.sgca_hidden_dim),
            nn.LayerNorm(config.sgca_hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        instruction_ids: Optional[torch.Tensor],
        instruction_embeds: torch.Tensor,
        llm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del instruction_ids
        q_projected = self.proj(self.qformer(visual_features=visual_features, text_embeds=instruction_embeds))
        llm_input = torch.cat([q_projected, instruction_embeds], dim=1)
        ctx = torch.no_grad() if self.config.freeze_llm else nullcontext()
        with ctx:
            hidden = llm.encoder(inputs_embeds=llm_input).last_hidden_state
        state_tokens = hidden[:, : self.config.num_query_tokens, :]
        return llm.lm_head(hidden), self.state_proj(state_tokens)


class RightBrain(nn.Module):
    """Textual-imagination branch."""

    def __init__(self, config: ATDConfig):
        super().__init__()
        self.config = config
        self.qformer = QFormer(
            num_query_tokens=config.num_query_tokens,
            hidden_dim=config.qformer_hidden_dim,
            num_layers=config.qformer_num_layers,
            num_heads=config.qformer_num_heads,
            text_input_dim=config.llm_hidden_dim,
        )
        self.proj = nn.Linear(config.qformer_hidden_dim, config.llm_hidden_dim)
        self.imagine_proj = nn.Sequential(
            nn.Linear(config.llm_hidden_dim, config.sgca_hidden_dim),
            nn.LayerNorm(config.sgca_hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        instruction_embeds: torch.Tensor,
        llm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_projected = self.proj(self.qformer(visual_features=visual_features, text_embeds=instruction_embeds))
        llm_input = torch.cat([q_projected, instruction_embeds], dim=1)
        ctx = torch.no_grad() if self.config.freeze_llm else nullcontext()
        with ctx:
            hidden = llm.encoder(inputs_embeds=llm_input).last_hidden_state
        imagine_tokens = hidden[:, : self.config.num_query_tokens, :]
        return llm.lm_head(hidden), self.imagine_proj(imagine_tokens)


class AdaptiveTextDreamer(nn.Module):
    """Full ATD model combining dual brains, SGCA, and graph policy."""

    def __init__(self, config: ATDConfig, llm: nn.Module, visual_encoder: nn.Module):
        super().__init__()
        self.config = config
        self.llm = llm
        self.visual_encoder = visual_encoder
        self.left_brain = LeftBrain(config)
        self.right_brain = RightBrain(config)
        self.sgca = StateGroundedCrossAttention(config.sgca_hidden_dim, config.sgca_num_layers)
        self.mca = nn.MultiheadAttention(config.graph_hidden_dim, num_heads=8, batch_first=True)
        self.atd_proj = nn.Linear(config.sgca_hidden_dim, config.graph_hidden_dim)
        self.graph_policy = GraphNavigationPolicy(
            hidden_dim=config.graph_hidden_dim,
            num_layers=config.graph_num_layers,
            instr_dim=config.llm_hidden_dim,
        )
        if config.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)
        for p in self.visual_encoder.parameters():
            p.requires_grad_(False)

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode images, or normalize already-encoded tensors to (B, N, D)."""
        if observations.dim() == 3:
            return observations
        if observations.dim() == 4:
            return observations.mean(dim=2)
        B, N, C, H, W = observations.shape
        with torch.no_grad():
            feats = self.visual_encoder(observations.view(B * N, C, H, W))
        return feats.view(B, N, -1)

    def forward(
        self,
        observations: torch.Tensor,
        instruction_ids: Optional[torch.Tensor],
        instruction_embeds: torch.Tensor,
        graph: Dict,
        target_actions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        vis_features = self.encode_observations(observations)
        state_logits, state_embed = self.left_brain(vis_features, instruction_ids, instruction_embeds, self.llm)
        imagine_logits, imagine_embed = self.right_brain(vis_features, instruction_embeds, self.llm)
        atd_embed, attn_maps = self.sgca(state_embed, imagine_embed, imagine_embed)
        atd_proj = self.atd_proj(atd_embed)
        vis_node_embed = graph["node_features"].to(atd_proj.device)
        graph_distances = graph["distances"].to(atd_proj.device)
        graph_edges = graph["edges"].to(atd_proj.device)
        fused_embed, _ = self.mca(query=vis_node_embed, key=atd_proj, value=atd_proj)
        action_logits = self.graph_policy(fused_embed, instruction_embeds, graph_edges, graph_distances)
        output = {
            "action_logits": action_logits,
            "state_logits": state_logits,
            "imagine_logits": imagine_logits,
            "atd_embed": atd_embed,
            "attention_maps": attn_maps,
        }
        if target_actions is not None:
            output["loss"] = self._compute_loss(action_logits, state_logits, imagine_logits, target_actions)
        return output

    def _compute_loss(
        self,
        action_logits: torch.Tensor,
        state_logits: torch.Tensor,
        imagine_logits: torch.Tensor,
        target_actions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        l_bc = F.cross_entropy(action_logits.view(-1, action_logits.size(-1)), target_actions["gt_actions"].view(-1))
        zero = torch.tensor(0.0, device=action_logits.device)
        l_pid = zero
        if "pseudo_actions" in target_actions:
            l_pid = F.cross_entropy(action_logits.view(-1, action_logits.size(-1)), target_actions["pseudo_actions"].view(-1))
        l_left = zero
        if "state_gt_ids" in target_actions:
            l_left = F.cross_entropy(
                state_logits.view(-1, state_logits.size(-1)),
                target_actions["state_gt_ids"].view(-1),
                ignore_index=-100,
            )
        l_right = zero
        if "imagine_gt_ids" in target_actions:
            l_right = F.cross_entropy(
                imagine_logits.view(-1, imagine_logits.size(-1)),
                target_actions["imagine_gt_ids"].view(-1),
                ignore_index=-100,
            )
        total = self.config.lambda_bc * l_bc + l_pid + l_left + l_right
        return {"total": total, "l_bc": l_bc, "l_pid": l_pid, "l_left": l_left, "l_right": l_right}

    @torch.no_grad()
    def generate_imagination(
        self,
        observations: torch.Tensor,
        instruction_embeds: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 100,
    ) -> list[str]:
        vis_features = self.encode_observations(observations)
        q = self.right_brain.qformer(vis_features, instruction_embeds)
        text_ids = self.llm.generate(inputs_embeds=self.right_brain.proj(q), max_new_tokens=max_new_tokens)
        return tokenizer.batch_decode(text_ids, skip_special_tokens=True)


def build_atd(config: Optional[ATDConfig] = None, pretrained: Optional[str] = None) -> AdaptiveTextDreamer:
    """Build ATD with heavy dependencies when available, otherwise lightweight mocks."""
    config = config or ATDConfig()
    try:
        from transformers import T5ForConditionalGeneration

        allow_download = os.getenv("DWF_ALLOW_MODEL_DOWNLOAD", "0") == "1"
        llm = T5ForConditionalGeneration.from_pretrained(config.llm_name, local_files_only=not allow_download)
    except Exception as exc:
        print(f"[ATD] Using lightweight local LLM fallback: {exc}")
        llm = _MockLLM(config.llm_hidden_dim)

    try:
        import timm

        visual_encoder = timm.create_model("vit_base_patch16_224", pretrained=os.getenv("DWF_ALLOW_MODEL_DOWNLOAD", "0") == "1")
        visual_encoder.head = nn.Identity()
    except Exception as exc:
        print(f"[ATD] Using lightweight local ViT fallback: {exc}")
        visual_encoder = _MockViT(768)

    model = AdaptiveTextDreamer(config, llm, visual_encoder)
    if pretrained is not None:
        checkpoint = torch.load(pretrained, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ATD] Trainable parameters: {n_params / 1e6:.1f}M")
    return model


class _MockLLM(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.vocab_size = 32128

        class _Encoder(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.layer = nn.Linear(d, d)

            def forward(self, inputs_embeds=None, **kwargs):
                del kwargs
                out = self.layer(inputs_embeds)

                class _Out:
                    pass

                result = _Out()
                result.last_hidden_state = out
                return result

        self.encoder = _Encoder(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, self.vocab_size)

    def generate(self, inputs_embeds=None, max_new_tokens: int = 100, **kwargs):
        del kwargs
        return torch.zeros(inputs_embeds.shape[0], max_new_tokens, dtype=torch.long, device=inputs_embeds.device)


class _MockViT(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(3 * 224 * 224, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))
