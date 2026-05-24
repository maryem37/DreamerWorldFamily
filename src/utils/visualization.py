"""Visualization utilities for DWF."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

_DWF_CMAP = LinearSegmentedColormap.from_list("dwf", ["#0d1117", "#1a3a5c", "#1f6aa5", "#3fb950", "#f0883e", "#ffffff"])


def plot_sgca_attention(
    attention_maps: List[torch.Tensor],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
    title: str = "SGCA Attention Matrices",
) -> plt.Figure:
    n_layers = len(attention_maps)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    if n_layers == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0d1117")
    for i, (attn, ax) in enumerate(zip(attention_maps, axes)):
        if attn.dim() == 4:
            attn_2d = attn.mean(dim=(0, 1)).cpu().numpy()
        elif attn.dim() == 3:
            attn_2d = attn.mean(dim=0).cpu().numpy()
        else:
            attn_2d = attn.cpu().numpy()
        ax.imshow(attn_2d, cmap=_DWF_CMAP, aspect="auto")
        ax.set_title(f"Layer {i + 1}", color="#e6edf3")
        ax.tick_params(colors="#7d8590", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#21262d")
        rows, cols = attn_2d.shape
        ax.add_patch(mpatches.Rectangle((cols * 0.4, rows * 0.25), max(cols * 0.25, 1), max(rows * 0.3, 1), edgecolor="#f85149", facecolor="none"))
        ax.add_patch(mpatches.Rectangle((-0.5, -0.5), cols, max(rows * 0.2, 1), edgecolor="#f0883e", facecolor="none", linestyle="--"))
    fig.suptitle(title, color="#e6edf3", fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_navigation_graph(
    visited_nodes: List[str],
    candidate_nodes: List[str],
    current_node: str,
    edge_list: List[Tuple[str, str]],
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    imagination_texts: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    import networkx as nx

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    graph = nx.DiGraph()
    graph.add_nodes_from(set(visited_nodes + candidate_nodes + [current_node]))
    graph.add_edges_from(edge_list)
    pos = positions or nx.spring_layout(graph, seed=42, k=2.0)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#21262d", arrows=True, arrowsize=12)
    colors = ["#58a6ff" if n == current_node else "#3fb950" if n in visited_nodes else "#f0883e" for n in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=360)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_color="#e6edf3", font_size=7)
    if imagination_texts:
        for node_id, text in imagination_texts.items():
            if node_id in pos:
                ax.annotate(text[:60], xy=pos[node_id], fontsize=6, color="#f0883e")
    ax.axis("off")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_convergence(
    atd_sr: List[float],
    atd_spl: List[float],
    baseline_sr: List[float],
    baseline_spl: List[float],
    iterations: List[int],
    baseline_name: str = "NavGPT2",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    for ax, atd_vals, base_vals, metric in [
        (axes[0], atd_spl, baseline_spl, "SPL"),
        (axes[1], atd_sr, baseline_sr, "SR"),
    ]:
        ax.set_facecolor("#0d1117")
        iters = iterations[: len(atd_vals)]
        ax.plot(iters, atd_vals[: len(iters)], color="#58a6ff", label="ATD")
        ax.plot(iters, base_vals[: len(iters)], color="#f0883e", linestyle="--", label=baseline_name)
        ax.set_title(f"Val Unseen {metric}", color="#e6edf3")
        ax.tick_params(colors="#7d8590")
        ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_metrics_comparison(
    methods: List[str],
    sr_values: List[float],
    spl_values: List[float],
    highlight_method: str = "ATD",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    x = np.arange(len(methods))
    for ax, values, title in [(axes[0], sr_values, "Success Rate"), (axes[1], spl_values, "SPL")]:
        ax.set_facecolor("#0d1117")
        colors = ["#58a6ff" if highlight_method in method else "#21262d" for method in methods]
        ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right", color="#7d8590")
        ax.set_title(title, color="#e6edf3")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def generate_demo_figures(output_dir: str = "outputs/figures") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mock_attns = [torch.softmax(torch.randn(1, 8, 32, 32) * (0.5 + i * 0.3), dim=-1) for i in range(4)]
    plot_sgca_attention(mock_attns, save_path=f"{output_dir}/sgca_attention.png")
    iters = list(range(0, 200_001, 5000))
    vals = [20 + 45 * (1 - math.exp(-i / 60000)) for i in range(len(iters))]
    base = [18 + 42 * (1 - math.exp(-i / 80000)) for i in range(len(iters))]
    plot_convergence(vals, vals, base, base, iters, save_path=f"{output_dir}/convergence.png")
    plot_metrics_comparison(["NavGPT2", "DUET", "ATD"], [68, 72, 75], [56, 60, 63], save_path=f"{output_dir}/metrics.png")
