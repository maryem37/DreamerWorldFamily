"""DREAMER WORLD FAMILY package setup."""
from pathlib import Path

from setuptools import find_packages, setup


long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="dreamer-world-family",
    version="1.0.0",
    author="DWF Authors",
    description="Unified framework for textual imagination, VLN navigation, and text-to-3D generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "networkx>=3.1.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
        "omegaconf>=2.3.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.0.0", "isort>=5.12.0"],
        "3d": ["diffusers>=0.21.0", "accelerate>=0.24.0"],
        "eval": ["open_clip_torch>=2.20.0", "wandb>=0.15.0"],
        "notebook": ["jupyter>=1.0.0", "ipywidgets>=8.0.0", "plotly>=5.17.0"],
    },
    entry_points={
        "console_scripts": [
            "dwf-train=train:main",
            "dwf-eval=evaluate:main",
            "dwf-infer=inference:main",
        ],
    },
)
