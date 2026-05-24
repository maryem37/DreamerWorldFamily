"""Simulator adapter interfaces for future real navigation backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SimulatorObservation:
    viewpoint_id: str
    image_features: object
    candidate_viewpoints: List[str]
    graph: Dict


class NavigationSimulator(ABC):
    """Minimal interface expected by the navigation inference loop."""

    @abstractmethod
    def reset(self, scan: str, start_viewpoint: str) -> SimulatorObservation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> SimulatorObservation:
        raise NotImplementedError

    @abstractmethod
    def current_path(self) -> List[str]:
        raise NotImplementedError
