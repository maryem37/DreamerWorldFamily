"""Caption collection scaffold for ATD left/right-brain supervision."""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def stitch_panoramic(images: List[np.ndarray], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    if not images:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    cols = min(len(images), 4)
    rows = math.ceil(len(images) / cols)
    th, tw = target_size
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = np.array(Image.fromarray(img).resize((tw, th)))
    return canvas


def image_to_base64(img: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class QwenVLCaptioner:
    PROMPT = "Describe the picture in detail, especially notice the furniture."

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        self.model_name = model_name
        self.device = device

    def caption(self, image: np.ndarray) -> str:
        del image
        objects = ["sofa", "table", "chair", "window", "door", "bookshelf", "lamp", "desk"]
        colors = ["red", "brown", "white", "wooden", "dark", "light blue"]
        return f"The image shows a room with a {random.choice(colors)} {random.choice(objects)} and nearby furniture."

    def caption_batch(self, images: List[np.ndarray]) -> List[str]:
        return [self.caption(img) for img in images]


class GPT4VStateEstimator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def estimate_state(self, image: np.ndarray, instruction: str) -> str:
        del image
        return f"Currently navigating according to: '{instruction[:80]}'. Continue toward the next landmark."

    def estimate_batch(self, images: List[np.ndarray], instructions: List[str], rate_limit_delay: float = 0.5) -> List[str]:
        out = []
        for img, instruction in zip(images, instructions):
            out.append(self.estimate_state(img, instruction))
            time.sleep(rate_limit_delay)
        return out


class GroundTruthCollector:
    def __init__(
        self,
        output_dir: str,
        qwen_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        use_gpt4v: bool = False,
        openai_api_key: Optional[str] = None,
        device: str = "cuda",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.captioner = QwenVLCaptioner(qwen_model, device)
        self.state_estimator = GPT4VStateEstimator(openai_api_key) if use_gpt4v else None

    def collect_episode(
        self,
        instr_id: str,
        scan: str,
        instruction: str,
        path: List[str],
        candidate_images: Dict[str, List[np.ndarray]],
        current_images: Dict[str, np.ndarray],
    ) -> List[Dict]:
        records = []
        for step, vp_id in enumerate(path[:-1]):
            candidates = candidate_images.get(vp_id, [])
            panorama = stitch_panoramic(candidates)
            imagination_gts = self.captioner.caption_batch(candidates) if candidates else ["A hallway.", "A room.", "An open space."]
            if self.state_estimator and vp_id in current_images:
                state_gt = self.state_estimator.estimate_state(current_images[vp_id], instruction)
            else:
                state_gt = f"Currently at step {step + 1} navigating towards goal."
            records.append(
                {
                    "instr_id": instr_id,
                    "scan": scan,
                    "viewpoint": vp_id,
                    "step": step,
                    "instruction": instruction,
                    "state_gt": state_gt,
                    "imagination_gt": imagination_gts,
                    "panorama_shape": list(panorama.shape),
                }
            )
        return records

    def save_records(self, records: List[Dict], split: str) -> None:
        with open(self.output_dir / f"{split}_imagination_gt.jsonl", "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def load_records(self, split: str) -> List[Dict]:
        path = self.output_dir / f"{split}_imagination_gt.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Collect imagination ground truth data")
    p.add_argument("--output_dir", default="data/imagination_gt")
    p.add_argument("--split", default="train", choices=["train", "val_seen", "val_unseen"])
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--use_gpt4v", action="store_true")
    p.add_argument("--openai_key", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    collector = GroundTruthCollector(args.output_dir, args.model, args.use_gpt4v, args.openai_key, args.device)
    records = collector.collect_episode(
        "mock_0",
        "scan_000",
        "Walk down the hallway and turn left into the room.",
        [f"vp_{i}" for i in range(5)],
        {},
        {},
    )
    collector.save_records(records, args.split)
    log.info("Saved %d records to %s", len(records), args.output_dir)
