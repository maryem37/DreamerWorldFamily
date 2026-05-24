"""DWF evaluation script."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.metrics import NavigationResult, Text3DMetrics, VLNMetrics
from src.utils.r2r_eval import evaluate_r2r_file, write_example_r2r

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def demo_evaluate_atd(checkpoint: Optional[str] = None) -> Dict[str, float]:
    del checkpoint
    metrics = VLNMetrics()
    np.random.seed(42)
    for _ in range(100):
        gt_path = [f"vp_{j}" for j in range(np.random.randint(4, 10))]
        success = np.random.random() < 0.75
        pred_path = gt_path.copy() if success else gt_path[:-1] + ["vp_wrong_0"]
        all_vps = list(set(pred_path + gt_path))

        def dist(a: str, b: str) -> float:
            if a == b:
                return 0.0
            if "wrong" in a or "wrong" in b:
                return 8.0
            return abs(int(a.split("_")[1]) - int(b.split("_")[1])) * 1.5

        dists = {(a, b): dist(a, b) for a in all_vps for b in all_vps}
        metrics.update(NavigationResult(pred_path, gt_path, [gt_path[-1]], dists))
    scores = metrics.compute()
    scores["mode"] = "demo_synthetic"
    return scores


def evaluate_3d_generation(args) -> Dict[str, float]:
    metrics = Text3DMetrics()
    generated_root = Path(args.generated_dir)
    metadata_files = sorted(generated_root.glob("*/metadata.json")) if generated_root.exists() else []
    if metadata_files:
        for metadata_file in metadata_files:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
            metrics._results.append(metrics.compute_clip_score([], [payload.get("prompt", "")]))
        scores = metrics.compute()
        scores["mode"] = "generated_asset_metadata"
        scores["generated_dir"] = str(generated_root)
        scores["n_assets"] = len(metadata_files)
        return scores

    prompts = ["A marble bust of an angel", "A strawberry", "A hamburger"]
    for prompt in prompts:
        metrics._results.append(metrics.compute_clip_score([], [prompt]))
    scores = metrics.compute()
    scores["mode"] = "demo_synthetic"
    scores["message"] = f"No metadata.json files found under {generated_root}. Generate assets first or pass --generated_dir."
    return scores


def parse_args():
    p = argparse.ArgumentParser(description="DWF Evaluation")
    p.add_argument("--module", choices=["text_dreamer", "x_dreamer", "world_dreamer"], default="text_dreamer")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="val_unseen", choices=["val_seen", "val_unseen", "test"])
    p.add_argument("--dataset", default="r2r", choices=["r2r", "reverie", "r4r"])
    p.add_argument("--output_dir", default="outputs/eval")
    p.add_argument("--save_figures", action="store_true", default=True)
    p.add_argument("--r2r_json", default=None, help="Prediction/reference JSON file for real R2R-style evaluation")
    p.add_argument("--write_example_r2r", action="store_true", help="Write an example JSON upload file and exit")
    p.add_argument("--allow_demo_eval", action="store_true", help="Allow synthetic demo navigation scores when --r2r_json is not supplied")
    p.add_argument("--generated_dir", default="outputs/generated", help="Directory containing generated mesh metadata for x_dreamer evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.write_example_r2r:
        out = write_example_r2r(Path(args.output_dir) / "example_r2r_predictions.json")
        print(f"Wrote {out}")
        return
    if args.r2r_json:
        result = evaluate_r2r_file(args.r2r_json)
        scores = result["scores"]
        scores["mode"] = "r2r_json"
        scores["n_records"] = result["n_records"]
        out_file = Path(args.output_dir) / "r2r_json_results.json"
    elif args.module in ("text_dreamer", "world_dreamer"):
        if not args.allow_demo_eval:
            example = write_example_r2r(Path(args.output_dir) / "example_r2r_predictions.json")
            raise SystemExit(
                "Real navigation evaluation needs --r2r_json. "
                f"An example file was written to {example}. "
                "Use --allow_demo_eval only for synthetic smoke tests."
            )
        scores = demo_evaluate_atd(args.checkpoint)
        out_file = Path(args.output_dir) / f"nav_results_{args.split}.json"
        if args.save_figures:
            from src.utils.visualization import plot_metrics_comparison

            plot_metrics_comparison(
                ["NavGPT2", "DUET", "ATD"],
                [68, 72, scores["SR"]],
                [56, 60, scores["SPL"]],
                save_path=str(Path(args.output_dir) / "metrics_comparison.png"),
            )
    else:
        scores = evaluate_3d_generation(args)
        out_file = Path(args.output_dir) / "gen_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))
    log.info("Results saved to %s", out_file)


if __name__ == "__main__":
    main()
