"""Matterport3D simulator adapter placeholder.

Install and wire Matterport3DSimulator here when dataset assets are available:
https://github.com/peteanderson80/Matterport3DSimulator
"""
from __future__ import annotations

from .base import NavigationSimulator, SimulatorObservation


class MatterportSimulatorAdapter(NavigationSimulator):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Matterport3D simulator is not installed/configured. "
            "Install Matterport3DSimulator and implement this adapter with your data paths."
        )

    def reset(self, scan: str, start_viewpoint: str) -> SimulatorObservation:
        raise NotImplementedError

    def step(self, action: int) -> SimulatorObservation:
        raise NotImplementedError

    def current_path(self):
        raise NotImplementedError
