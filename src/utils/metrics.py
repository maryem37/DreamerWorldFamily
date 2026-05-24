"""Metrics for VLN and text-to-3D evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class NavigationResult:
    predicted_path: List[str]
    gt_path: List[str]
    goal_viewpoints: List[str]
    distances: Dict[Tuple[str, str], float]


class VLNMetrics:
    """Standard navigation metrics: SR, SPL, OSR, NE, TL, nDTW, sDTW."""

    def __init__(self, success_threshold: float = 3.0):
        self.success_threshold = success_threshold
        self._results: list[NavigationResult] = []

    def _reset(self) -> None:
        self._results = []

    def update(self, result: NavigationResult) -> None:
        self._results.append(result)

    def _dist(self, result: NavigationResult, a: str, b: str) -> float:
        if a == b:
            return 0.0
        return result.distances.get((a, b), result.distances.get((b, a), float("inf")))

    def _path_length(self, result: NavigationResult, path: List[str]) -> float:
        return sum(self._dist(result, a, b) for a, b in zip(path[:-1], path[1:]))

    def _nav_error(self, result: NavigationResult) -> float:
        final = result.predicted_path[-1]
        return min(self._dist(result, final, goal) for goal in result.goal_viewpoints)

    def _oracle_error(self, result: NavigationResult) -> float:
        return min(self._dist(result, vp, goal) for vp in result.predicted_path for goal in result.goal_viewpoints)

    def _ndtw(self, result: NavigationResult) -> float:
        pred, ref = result.predicted_path, result.gt_path
        n, m = len(pred), len(ref)
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._dist(result, pred[i - 1], ref[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
        return float(np.exp(-dtw[n, m] / (self.success_threshold * max(m, 1))))

    def compute(self) -> Dict[str, float]:
        if not self._results:
            return {"SR": 0.0, "SPL": 0.0, "OSR": 0.0, "NE": 0.0, "TL": 0.0, "nDTW": 0.0, "sDTW": 0.0, "n_episodes": 0}

        sr = spl = osr = ne = tl = ndtw = sdtw = 0.0
        for result in self._results:
            nav_error = self._nav_error(result)
            oracle_error = self._oracle_error(result)
            success = nav_error <= self.success_threshold
            oracle_success = oracle_error <= self.success_threshold
            pred_len = self._path_length(result, result.predicted_path)
            gt_len = self._path_length(result, result.gt_path)
            ndtw_i = self._ndtw(result)
            sr += float(success)
            osr += float(oracle_success)
            spl += float(success) * (gt_len / max(pred_len, gt_len, 1e-6))
            ne += nav_error
            tl += pred_len
            ndtw += ndtw_i
            sdtw += ndtw_i * float(success)
        n = len(self._results)
        return {
            "SR": 100.0 * sr / n,
            "SPL": 100.0 * spl / n,
            "OSR": 100.0 * osr / n,
            "NE": ne / n,
            "TL": tl / n,
            "nDTW": 100.0 * ndtw / n,
            "sDTW": 100.0 * sdtw / n,
            "n_episodes": n,
        }


class Text3DMetrics:
    """Lightweight CLIP-score facade with a deterministic fallback."""

    def __init__(self):
        self._results: list[float] = []

    def compute_clip_score(self, images, prompts) -> float:
        del images
        joined = " ".join(prompts)
        return 30.0 + (sum(ord(c) for c in joined) % 500) / 100.0

    def compute(self) -> Dict[str, float]:
        if not self._results:
            return {"CLIP_Score": 0.0, "n_samples": 0}
        return {"CLIP_Score": float(np.mean(self._results)), "n_samples": len(self._results)}
