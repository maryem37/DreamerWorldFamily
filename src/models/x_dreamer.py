"""X-Dreamer and WorldDreamer modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class XDreamerConfig:
    cam_feature_dim: int = 1024
    text_feature_dim: int = 1024
    lora_rank: int = 4
    lora_rank_prime: int = 4
    ama_loss_weight: float = 0.1
    ama_start_iter: int = 1000
    geometry_iters: int = 2000
    appearance_iters: int = 1000
    batch_size: int = 4
    lr: float = 1e-3
    pitch_range: Tuple[float, float] = (-15.0, 45.0)
    yaw_range: Tuple[float, float] = (-180.0, 180.0)
    fov_range: Tuple[float, float] = (25.0, 45.0)
    sd_model_id: str = "stabilityai/stable-diffusion-2-1-base"


class CGLoRA(nn.Module):
    """Camera-guided LoRA residual block."""

    def __init__(self, d_model: int, config: XDreamerConfig):
        super().__init__()
        if config.lora_rank % 2 != 0:
            raise ValueError("lora_rank must be even")
        self.d = d_model
        self.r = config.lora_rank
        self.U_txt = nn.Linear(config.text_feature_dim, config.lora_rank_prime, bias=False)
        self.V_txt = nn.Linear(config.lora_rank_prime, d_model * (self.r // 2), bias=False)
        self.U_cam = nn.Linear(config.cam_feature_dim, config.lora_rank_prime, bias=False)
        self.V_cam = nn.Linear(config.lora_rank_prime, d_model * (self.r // 2), bias=False)
        self.B = nn.Parameter(torch.zeros(self.r, d_model))
        self.scale = 1.0 / self.r

    def forward(self, x: torch.Tensor, text_feat: torch.Tensor, cam_feat: torch.Tensor) -> torch.Tensor:
        B_size, seq, d = x.shape
        if d != self.d:
            raise ValueError(f"Expected input dim {self.d}, got {d}")
        A_txt = self.V_txt(self.U_txt(text_feat)).view(B_size, self.d, self.r // 2)
        A_cam = self.V_cam(self.U_cam(cam_feat)).view(B_size, self.d, self.r // 2)
        A = torch.cat([A_txt, A_cam], dim=-1)
        xA = torch.bmm(x, A)
        return self.scale * torch.mm(xA.view(-1, self.r), self.B).view(B_size, seq, d)


class CameraEncoder(nn.Module):
    """Encodes (x, y, z, yaw, pitch, fov)."""

    def __init__(self, out_dim: int = 1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, camera_params: torch.Tensor) -> torch.Tensor:
        return self.encoder(camera_params)


class AMALoss(nn.Module):
    """Attention-mask alignment loss."""

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def normalize_attention(self, attn_map: torch.Tensor) -> torch.Tensor:
        flat = attn_map.reshape(attn_map.shape[0], -1)
        shape = (attn_map.shape[0],) + (1,) * (attn_map.dim() - 1)
        min_val = flat.min(dim=-1, keepdim=True)[0].view(shape)
        max_val = flat.max(dim=-1, keepdim=True)[0].view(shape)
        span = max_val - min_val
        normed = (attn_map - min_val) / (span + self.epsilon)
        return torch.where(span < 1e-5, torch.ones_like(normed), normed)

    def forward(self, attention_maps: list[torch.Tensor], rendered_mask: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=rendered_mask.device)
        for attn in attention_maps:
            attn_2d = attn.mean(dim=1) if attn.dim() == 4 else attn
            h, w = attn_2d.shape[-2:]
            mask_resized = F.interpolate(rendered_mask.float(), size=(h, w), mode="bilinear", align_corners=False)
            total_loss = total_loss + F.l1_loss(self.normalize_attention(attn_2d), mask_resized.squeeze(1))
        return total_loss / max(len(attention_maps), 1)


class XDreamer(nn.Module):
    """Text-to-3D pipeline scaffold with CG-LoRA and AMA components."""

    def __init__(self, config: XDreamerConfig, stable_diffusion=None):
        super().__init__()
        self.config = config
        self.sd = stable_diffusion
        self.cam_encoder = CameraEncoder(config.cam_feature_dim)
        self._cg_loras = nn.ModuleDict()
        self.ama_loss = AMALoss()

    def inject_cg_loras(self, sd_unet: nn.Module) -> None:
        count = 0
        for name, module in sd_unet.named_modules():
            if isinstance(module, nn.Linear) and "attn" in name.lower():
                self._cg_loras[name.replace(".", "_")] = CGLoRA(module.out_features, self.config)
                count += 1
        for p in sd_unet.parameters():
            p.requires_grad_(False)
        print(f"[X-Dreamer] Injected CG-LoRA into {count} attention layers")

    def get_direction_text(self, yaw_angle: float) -> str:
        yaw = yaw_angle % 360
        if yaw < 45 or yaw >= 315:
            return "front view"
        if yaw < 135:
            return "right side view"
        if yaw < 225:
            return "back view"
        return "left side view"

    def sample_camera(self) -> Dict[str, float | str]:
        pitch = np.random.uniform(*self.config.pitch_range)
        yaw = np.random.uniform(*self.config.yaw_range)
        fov = np.random.uniform(*self.config.fov_range)
        r = 2.5
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        return {
            "x": float(r * np.cos(pitch_rad) * np.sin(yaw_rad)),
            "y": float(r * np.sin(pitch_rad)),
            "z": float(r * np.cos(pitch_rad) * np.cos(yaw_rad)),
            "yaw": float(yaw_rad),
            "pitch": float(pitch_rad),
            "fov": float(fov),
            "direction_text": self.get_direction_text(yaw),
        }

    def sds_loss(self, rendered: torch.Tensor, text_prompt: str, camera_params: Dict, text_encoder=None, noise_scheduler=None) -> torch.Tensor:
        del text_prompt, camera_params, text_encoder, noise_scheduler
        t = torch.rand(rendered.shape[0], device=rendered.device)
        noisy = rendered + t.view(-1, 1, 1, 1) * torch.randn_like(rendered)
        return F.mse_loss(noisy, rendered)

    def geometry_step(self, dmtet, text_prompt: str, text_encoder, iteration: int) -> Dict[str, torch.Tensor]:
        cam = self.sample_camera()
        normal_map, mask = dmtet.render(cam)
        l_sds = self.sds_loss(normal_map, text_prompt, cam, text_encoder)
        losses = {"sds": l_sds}
        if iteration >= self.config.ama_start_iter:
            attention_maps: list[torch.Tensor] = []
            if attention_maps:
                losses["ama"] = self.config.ama_loss_weight * self.ama_loss(attention_maps, mask)
        losses["total"] = sum(losses.values())
        return losses

    def appearance_step(self, mesh, material_encoder, text_prompt: str, text_encoder, iteration: int) -> Dict[str, torch.Tensor]:
        rendered_image, mask = self._render_pbr(mesh, material_encoder, self.sample_camera())
        losses = {"sds": self.sds_loss(rendered_image, text_prompt, {}, text_encoder)}
        if iteration >= self.config.ama_start_iter:
            attention_maps: list[torch.Tensor] = []
            if attention_maps:
                losses["ama"] = self.config.ama_loss_weight * self.ama_loss(attention_maps, mask)
        losses["total"] = sum(losses.values())
        return losses

    def _render_pbr(self, mesh, material_encoder, cam):
        del mesh, material_encoder, cam
        return torch.zeros(1, 3, 512, 512), torch.zeros(1, 1, 512, 512)


class WorldDreamer(nn.Module):
    """Unified DWF model joining ATD features with camera-conditioned world features."""

    def __init__(self, atd_config, xd_config: XDreamerConfig, world_dim: int = 1024):
        super().__init__()
        from .text_dreamer import build_atd

        self.text_dreamer = build_atd(atd_config)
        self.cam_encoder = CameraEncoder(xd_config.cam_feature_dim)
        self.ama_loss = AMALoss()
        self.world_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(world_dim, nhead=16, dim_feedforward=world_dim * 4, batch_first=True),
            num_layers=6,
        )
        self.nav_head = nn.Linear(world_dim, atd_config.graph_hidden_dim)
        self.gen_head = nn.Linear(world_dim, xd_config.cam_feature_dim)
        self.world_proj = nn.Linear(atd_config.sgca_hidden_dim, world_dim)

    def forward(
        self,
        observations: torch.Tensor,
        instruction_embeds: torch.Tensor,
        camera_params: Optional[torch.Tensor] = None,
        graph: Optional[Dict] = None,
        mode: str = "navigation",
    ) -> Dict[str, torch.Tensor | bool | Dict]:
        outputs: Dict[str, torch.Tensor | bool | Dict] = {}
        if mode in ("navigation", "unified"):
            if graph is None:
                B = observations.shape[0]
                graph = {
                    "node_features": torch.zeros(B, 1, self.text_dreamer.config.graph_hidden_dim, device=observations.device),
                    "edges": torch.zeros(B, 1, 2, dtype=torch.long, device=observations.device),
                    "distances": torch.zeros(B, 1, 1, device=observations.device),
                }
            nav_out = self.text_dreamer(observations, None, instruction_embeds, graph)
            outputs["navigation"] = nav_out
            outputs["world_features"] = self.world_encoder(self.world_proj(nav_out["atd_embed"]))
        if mode in ("generation", "unified") and camera_params is not None:
            cam_feat = self.cam_encoder(camera_params)
            if "world_features" in outputs:
                cam_feat = cam_feat + self.gen_head(outputs["world_features"].mean(1))
            outputs["camera_features"] = cam_feat
            outputs["generation_ready"] = True
        return outputs
