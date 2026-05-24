"""Real JSON-based R2R/VLN evaluation utilities.

Supported input formats:

1. A list of records:
   [
     {
       "instr_id": "0",
       "predicted_path": ["vp_0", "vp_1"],
       "gt_path": ["vp_0", "vp_1"],
       "goal_viewpoints": ["vp_1"],
       "distances": {"vp_0|vp_1": 1.5}
     }
   ]

2. A wrapped object:
   {"episodes": [...]} or {"predictions": [...]}.

Distances may be supplied as "a|b" keys, "a,b" keys, nested dictionaries,
or omitted for simple synthetic viewpoint ids like vp_0, vp_1.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .metrics import NavigationResult, VLNMetrics


def _records_from_payload(payload: Any) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("episodes", "predictions", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("R2R JSON must be a list or contain one of: episodes, predictions, results, data")


def _as_path(record: dict, *keys: str) -> List[str]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, list) and value:
            return [str(x) for x in value]
    raise ValueError(f"Missing path field. Expected one of: {', '.join(keys)}")


def _linear_distance(a: str, b: str) -> float:
    if a == b:
        return 0.0
    nums_a = re.findall(r"-?\d+", a)
    nums_b = re.findall(r"-?\d+", b)
    if nums_a and nums_b:
        return abs(int(nums_a[-1]) - int(nums_b[-1])) * 1.5
    return 1.5


def _parse_distances(raw: Any, nodes: Iterable[str]) -> Dict[Tuple[str, str], float]:
    if not raw:
        node_list = list(dict.fromkeys(nodes))
        return {(a, b): _linear_distance(a, b) for a in node_list for b in node_list}

    distances: Dict[Tuple[str, str], float] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    distances[(str(key), str(inner_key))] = float(inner_value)
                continue
            if isinstance(key, str):
                if "|" in key:
                    a, b = key.split("|", 1)
                elif "," in key:
                    a, b = key.split(",", 1)
                else:
                    continue
                distances[(a.strip(), b.strip())] = float(value)
    if not distances:
        raise ValueError("Distances were provided but no valid pairs were parsed")
    for (a, b), value in list(distances.items()):
        distances.setdefault((b, a), value)
    for node in nodes:
        distances.setdefault((node, node), 0.0)
    return distances


def evaluate_r2r_payload(payload: Any, success_threshold: float = 3.0) -> dict:
    records = _records_from_payload(payload)
    metrics = VLNMetrics(success_threshold=success_threshold)
    normalized = []
    for idx, record in enumerate(records):
        pred = _as_path(record, "predicted_path", "prediction", "path_pred", "trajectory")
        gt = _as_path(record, "gt_path", "ground_truth_path", "reference_path", "path")
        goals = record.get("goal_viewpoints") or record.get("goals") or [gt[-1]]
        goals = [str(x) for x in goals]
        nodes = pred + gt + goals
        distances = _parse_distances(record.get("distances"), nodes)
        result = NavigationResult(predicted_path=pred, gt_path=gt, goal_viewpoints=goals, distances=distances)
        metrics.update(result)
        normalized.append(
            {
                "instr_id": str(record.get("instr_id", record.get("id", idx))),
                "predicted_path": pred,
                "gt_path": gt,
                "goal_viewpoints": goals,
            }
        )
    scores = metrics.compute()
    return {"scores": scores, "n_records": len(records), "episodes": normalized}


def evaluate_r2r_file(path: str | Path, success_threshold: float = 3.0) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return evaluate_r2r_payload(payload, success_threshold=success_threshold)


def write_example_r2r(path: str | Path) -> Path:
    """Write a tiny valid example upload file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "instr_id": "demo_success",
            "predicted_path": ["vp_0", "vp_1", "vp_2", "vp_3"],
            "gt_path": ["vp_0", "vp_1", "vp_2", "vp_3"],
            "goal_viewpoints": ["vp_3"],
        },
        {
            "instr_id": "demo_detour",
            "predicted_path": ["vp_0", "vp_1", "vp_2", "vp_1", "vp_2", "vp_3"],
            "gt_path": ["vp_0", "vp_1", "vp_2", "vp_3"],
            "goal_viewpoints": ["vp_3"],
        },
    ]
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
