"""Text-to-3D backend interface.

The current app uses the procedural backend. Real model backends such as
Shap-E, TripoSR, Stable Fast 3D, or InstantMesh can implement the same
generate(prompt, output_dir) contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.utils.mesh_generator import MeshAsset, generate_mesh


class TextTo3DBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, output_dir: str | Path) -> MeshAsset:
        raise NotImplementedError


class ProceduralMeshBackend(TextTo3DBackend):
    def generate(self, prompt: str, output_dir: str | Path) -> MeshAsset:
        return generate_mesh(prompt, output_dir)


class ExternalModelBackend(TextTo3DBackend):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, output_dir: str | Path) -> MeshAsset:
        raise RuntimeError(
            f"{self.model_name} backend is not installed. "
            "Add the model dependencies and implement this generate() method."
        )
