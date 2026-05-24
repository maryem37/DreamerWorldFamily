"""DWF training entry point."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.mesh_generator import generate_mesh

try:
    import torch
except ModuleNotFoundError:
    torch = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("DWF")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DREAMER WORLD FAMILY training")
    p.add_argument("--module", choices=["text_dreamer", "x_dreamer", "world_dreamer"], default="text_dreamer")
    p.add_argument("--config", default=None)
    p.add_argument("--output_dir", default="outputs/run")
    p.add_argument("--dataset", choices=["r2r", "reverie", "r4r"], default="r2r")
    p.add_argument("--data_dir", default="data/")
    p.add_argument("--use_prevalent", action="store_true", default=True)
    p.add_argument("--max_iterations", type=int, default=200000)
    p.add_argument("--prompt", default="A red apple, highly detailed")
    p.add_argument("--geo_iters", type=int, default=2000)
    p.add_argument("--app_iters", type=int, default=1000)
    p.add_argument("--llm", default="google/flan-t5-xl", choices=["google/flan-t5-xl", "google/flan-t5-xxl"])
    p.add_argument("--sgca_layers", type=int, default=4)
    p.add_argument("--num_queries", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke_steps", type=int, default=1, help="Run a small real local training/inference smoke loop")
    return p.parse_args()


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _deep_get(config: dict, path: str, default=None):
    current = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _infer_module(args: argparse.Namespace, config: dict) -> str:
    name = str(_deep_get(config, "model.name", "")).lower()
    if "world" in name:
        return "world_dreamer"
    if "xdreamer" in name or "x-dreamer" in name:
        return "x_dreamer"
    if "text" in name or "adaptive" in name:
        return "text_dreamer"
    return args.module


def _atd_config(args: argparse.Namespace, config: dict, prefix: str = "model") -> ATDConfig:
    from src.models.text_dreamer import ATDConfig

    return ATDConfig(
        num_query_tokens=int(_deep_get(config, f"{prefix}.num_query_tokens", args.num_queries)),
        qformer_hidden_dim=int(_deep_get(config, f"{prefix}.qformer_hidden_dim", 768)),
        qformer_num_layers=int(_deep_get(config, f"{prefix}.qformer_num_layers", 6)),
        qformer_num_heads=int(_deep_get(config, f"{prefix}.qformer_num_heads", 12)),
        llm_name=str(_deep_get(config, f"{prefix}.llm", args.llm)),
        freeze_llm=bool(_deep_get(config, f"{prefix}.freeze_llm", True)),
        sgca_num_layers=int(_deep_get(config, f"{prefix}.sgca_num_layers", args.sgca_layers)),
        sgca_hidden_dim=int(_deep_get(config, f"{prefix}.sgca_hidden_dim", 512)),
        graph_hidden_dim=int(_deep_get(config, f"{prefix}.graph_hidden_dim", 768)),
        graph_num_layers=int(_deep_get(config, f"{prefix}.graph_num_layers", 4)),
        batch_size=int(_deep_get(config, "training.batch_size", args.batch_size)),
        lr=float(_deep_get(config, "training.lr", args.lr)),
        weight_decay=float(_deep_get(config, "training.weight_decay", 0.05)),
        lambda_bc=float(_deep_get(config, "training.lambda_bc", 1.0)),
    )


def _xd_config(args: argparse.Namespace, config: dict, prefix: str = "model") -> XDreamerConfig:
    from src.models.x_dreamer import XDreamerConfig

    return XDreamerConfig(
        cam_feature_dim=int(_deep_get(config, f"{prefix}.cam_feature_dim", 1024)),
        text_feature_dim=int(_deep_get(config, f"{prefix}.text_feature_dim", 1024)),
        lora_rank=int(_deep_get(config, f"{prefix}.lora_rank", 4)),
        lora_rank_prime=int(_deep_get(config, f"{prefix}.lora_rank_prime", 4)),
        ama_loss_weight=float(_deep_get(config, f"{prefix}.ama_loss_weight", 0.1)),
        ama_start_iter=int(_deep_get(config, f"{prefix}.ama_start_iter", 1000)),
        geometry_iters=int(_deep_get(config, "training.geometry_iters", args.geo_iters)),
        appearance_iters=int(_deep_get(config, "training.appearance_iters", args.app_iters)),
        batch_size=int(_deep_get(config, "training.batch_size", args.batch_size)),
        lr=float(_deep_get(config, "training.lr", 1e-3)),
        pitch_range=tuple(_deep_get(config, "camera.pitch_range", [-15.0, 45.0])),
        yaw_range=tuple(_deep_get(config, "camera.yaw_range", [-180.0, 180.0])),
        fov_range=tuple(_deep_get(config, "camera.fov_range", [25.0, 45.0])),
        sd_model_id=str(_deep_get(config, f"{prefix}.sd_model_id", "stabilityai/stable-diffusion-2-1-base")),
    )


def _param_count(model, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def _run_text_smoke(model, args: argparse.Namespace, atd_config: ATDConfig, config: dict) -> dict:
    from src.data.r2r_dataset import build_r2r_loaders

    data_dir = str(_deep_get(config, "data.data_dir", _deep_get(config, "data.nav_data_dir", args.data_dir)))
    loaders = build_r2r_loaders(data_dir, batch_size=atd_config.batch_size, use_prevalent=args.use_prevalent)
    batch = next(iter(loaders["train"]))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=atd_config.lr, weight_decay=atd_config.weight_decay)
    losses = []
    model.train()
    for _ in range(max(args.smoke_steps, 0)):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["observations"], batch["instruction_ids"], batch["instruction_embeds"], batch["graph"], batch["target_actions"])
        loss = output["loss"]["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "data_dir": data_dir,
        "batch_size": atd_config.batch_size,
        "sample_sources": batch["source"],
        "sample_instruction": batch["instruction_text"][0],
        "smoke_steps": len(losses),
        "losses": losses,
    }


def _run_xdreamer_smoke(args: argparse.Namespace) -> dict:
    asset = generate_mesh(args.prompt, Path(args.output_dir) / "generated")
    return {
        "prompt": args.prompt,
        "job_id": asset.job_id,
        "kind": asset.kind,
        "material": asset.material,
        "vertices": asset.vertices,
        "faces": asset.faces,
        "quality": asset.quality,
        "files": {
            "obj": str(asset.obj_path),
            "stl": str(asset.stl_path),
            "preview": str(asset.preview_path),
            "metadata": str(asset.metadata_path),
        },
    }


def main() -> None:
    args = parse_args()
    loaded_config = _load_yaml(args.config)
    args.module = _infer_module(args, loaded_config)
    if torch is not None:
        torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary = {"args": vars(args), "config": loaded_config, "module": args.module, "seed": args.seed}
    log.info("DREAMER WORLD FAMILY - Training")
    device = "torch-not-installed" if torch is None else ("CUDA" if torch.cuda.is_available() else "CPU")
    log.info("Module: %s | Output: %s | Device: %s", args.module, args.output_dir, device)

    if args.module == "text_dreamer":
        if torch is None:
            raise SystemExit("text_dreamer training requires PyTorch. Install requirements.txt, then rerun this command.")
        from src.models.text_dreamer import build_atd

        config = _atd_config(args, loaded_config)
        model = build_atd(config)
        summary["model_config"] = asdict(config)
        summary["params"] = {"total": _param_count(model), "trainable": _param_count(model, True)}
        if args.smoke_steps > 0:
            summary["smoke"] = _run_text_smoke(model, args, config, loaded_config)
        log.info("[ATD] Trainable params: %.2fM", summary["params"]["trainable"] / 1e6)
    elif args.module == "x_dreamer":
        if torch is not None:
            from src.models.x_dreamer import XDreamer

            config = _xd_config(args, loaded_config)
            model = XDreamer(config)
            summary["model_config"] = asdict(config)
            summary["params"] = {"total": _param_count(model), "trainable": _param_count(model, True)}
        else:
            summary["model_config"] = _deep_get(loaded_config, "model", {})
            summary["params"] = {"total": 0, "trainable": 0, "note": "PyTorch not installed; generated procedural asset only"}
        if args.smoke_steps > 0:
            summary["smoke"] = _run_xdreamer_smoke(args)
        log.info("[X-Dreamer] Model built for prompt: %s", args.prompt)
        log.info("[X-Dreamer] Trainable params: %.2fM", summary["params"]["trainable"] / 1e6)
    else:
        if torch is None:
            raise SystemExit("world_dreamer training requires PyTorch. Install requirements.txt, then rerun this command.")
        from src.models.x_dreamer import WorldDreamer

        atd_config = _atd_config(args, loaded_config, "model.atd")
        xd_config = _xd_config(args, loaded_config, "model.xdreamer")
        model = WorldDreamer(atd_config, xd_config, world_dim=int(_deep_get(loaded_config, "model.world_dim", 1024)))
        summary["model_config"] = {"atd": asdict(atd_config), "xdreamer": asdict(xd_config)}
        summary["params"] = {"total": _param_count(model), "trainable": _param_count(model, True)}
        if args.smoke_steps > 0:
            summary["smoke"] = _run_xdreamer_smoke(args)
        log.info("[WorldDreamer] Model built: %.2fM params", summary["params"]["total"] / 1e6)
    out = Path(args.output_dir) / "run_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Run summary saved to %s", out)


if __name__ == "__main__":
    main()
