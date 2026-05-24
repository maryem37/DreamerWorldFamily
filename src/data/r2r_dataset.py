"""R2R-style dataset loader with deterministic fallback samples."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


class R2RDataset(Dataset):
    """Load local R2R-style JSON records, or deterministic synthetic samples.

    Expected JSON formats are a plain list, or an object containing one of
    ``episodes``, ``predictions``, ``results``, or ``data``. The loader accepts
    common field names such as ``instruction``, ``instructions``, ``path`` and
    ``scan``. Visual features remain lightweight deterministic tensors unless a
    future feature store is connected.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        use_prevalent: bool = True,
        augment: bool = False,
        num_candidates: int = 4,
        length: int = 16,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_prevalent = use_prevalent
        self.augment = augment
        self.num_candidates = num_candidates
        self.records = self._load_records()
        self.length = len(self.records) if self.records else length

    def _candidate_files(self) -> List[Path]:
        names = [
            f"{self.split}.json",
            f"R2R_{self.split}.json",
            f"r2r_{self.split}.json",
            f"{self.split}_episodes.json",
        ]
        files = [self.data_dir / name for name in names]
        if self.data_dir.exists():
            files.extend(sorted(self.data_dir.glob(f"*{self.split}*.json")))
        return list(dict.fromkeys(files))

    def _records_from_payload(self, payload) -> List[dict]:
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            for key in ("episodes", "predictions", "results", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
        return []

    def _load_records(self) -> List[dict]:
        for path in self._candidate_files():
            if not path.exists():
                continue
            records = self._records_from_payload(json.loads(path.read_text(encoding="utf-8")))
            if records:
                return records
        return []

    def __len__(self) -> int:
        return self.length

    def _seed(self, *parts: object) -> int:
        text = "|".join(str(part) for part in parts)
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)

    def _feature(self, shape: tuple[int, ...], *parts: object) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self._seed(*parts))
        return torch.randn(*shape, generator=generator)

    def _instruction_ids(self, instruction: str, max_len: int = 80) -> torch.Tensor:
        values = [min(ord(ch), 255) for ch in instruction[:max_len]]
        values.extend([0] * (max_len - len(values)))
        return torch.tensor(values, dtype=torch.long)

    def _record_at(self, idx: int) -> Optional[dict]:
        if not self.records:
            return None
        return self.records[idx % len(self.records)]

    def __getitem__(self, idx: int) -> Dict:
        K = self.num_candidates
        record = self._record_at(idx)
        if record:
            raw_instruction = record.get("instruction") or record.get("instructions") or record.get("instr") or ""
            instruction = raw_instruction[0] if isinstance(raw_instruction, list) else str(raw_instruction)
            path = [str(x) for x in record.get("path", record.get("gt_path", [f"vp_{j}" for j in range(5)]))]
            instr_id = str(record.get("instr_id", record.get("id", f"{self.split}_{idx}")))
            scan = str(record.get("scan", record.get("scan_id", "unknown_scan")))
        else:
            instruction = "Walk down the hallway and turn left into the room."
            path = [f"vp_{j}" for j in range(5)]
            instr_id = f"{self.split}_{idx}"
            scan = f"scan_{idx % 3:03d}"
        return {
            "instr_id": instr_id,
            "scan": scan,
            "path": path,
            "instruction_text": instruction,
            "instruction_ids": self._instruction_ids(instruction),
            "instruction_embeds": self._feature((80, 2048), instr_id, "instruction"),
            "observations": self._feature((K, 768), instr_id, "observations"),
            "graph": {
                "node_features": self._feature((K, 768), instr_id, "nodes"),
                "edges": torch.zeros(K, 2, dtype=torch.long),
                "distances": torch.cdist(torch.arange(K, dtype=torch.float32).view(-1, 1), torch.arange(K, dtype=torch.float32).view(-1, 1)),
            },
            "target_actions": {"gt_actions": torch.tensor(0, dtype=torch.long)},
            "source": "json" if record else "deterministic_fallback",
        }


def collate_r2r(items: List[Dict]) -> Dict:
    graph = {
        "node_features": torch.stack([x["graph"]["node_features"] for x in items]),
        "edges": torch.stack([x["graph"]["edges"] for x in items]),
        "distances": torch.stack([x["graph"]["distances"] for x in items]),
    }
    targets = {"gt_actions": torch.stack([x["target_actions"]["gt_actions"] for x in items])}
    return {
        "instr_id": [x["instr_id"] for x in items],
        "scan": [x["scan"] for x in items],
        "path": [x["path"] for x in items],
        "source": [x["source"] for x in items],
        "instruction_text": [x["instruction_text"] for x in items],
        "instruction_ids": torch.stack([x["instruction_ids"] for x in items]),
        "instruction_embeds": torch.stack([x["instruction_embeds"] for x in items]),
        "observations": torch.stack([x["observations"] for x in items]),
        "graph": graph,
        "target_actions": targets,
    }


def build_r2r_loaders(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 0,
    use_prevalent: bool = True,
) -> Dict[str, DataLoader]:
    return {
        split: DataLoader(
            R2RDataset(data_dir, split, use_prevalent=use_prevalent),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_r2r,
        )
        for split in ("train", "val_seen", "val_unseen")
    }
