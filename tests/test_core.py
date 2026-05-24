import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_qformer_output_shape():
    from src.models.qformer import QFormer

    qf = QFormer(num_query_tokens=8, hidden_dim=64, num_layers=1, num_heads=4, vis_input_dim=128, text_input_dim=256)
    out = qf(torch.randn(2, 5, 128), torch.randn(2, 10, 256))
    assert out.shape == (2, 8, 64)


def test_sgca_attention_rows_sum():
    from src.models.sgca import SGCALayer

    layer = SGCALayer(hidden_dim=32, num_heads=4, dropout=0.0)
    _, attn = layer(torch.randn(1, 4, 32), torch.randn(1, 4, 32), torch.randn(1, 4, 32))
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)


def test_graph_policy_logits_shape():
    from src.models.graph_policy import GraphNavigationPolicy

    policy = GraphNavigationPolicy(hidden_dim=64, num_layers=1, num_heads=4)
    logits = policy(torch.randn(2, 5, 64), torch.randn(2, 10, 64), torch.zeros(2, 5, 2), torch.rand(2, 5, 5))
    assert logits.shape == (2, 5)


def test_cglora_zero_init():
    from src.models.x_dreamer import CGLoRA, XDreamerConfig

    cfg = XDreamerConfig(cam_feature_dim=16, text_feature_dim=16, lora_rank=2, lora_rank_prime=2)
    lora = CGLoRA(8, cfg)
    residual = lora(torch.randn(1, 3, 8), torch.randn(1, 16), torch.randn(1, 16))
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_ama_loss_nonnegative():
    from src.models.x_dreamer import AMALoss

    loss = AMALoss()([torch.rand(1, 2, 8, 8)], torch.rand(1, 1, 32, 32))
    assert loss.item() >= 0


def test_vln_metrics_perfect():
    from src.utils.metrics import NavigationResult, VLNMetrics

    path = [f"vp_{i}" for i in range(4)]
    dists = {(a, b): abs(int(a.split("_")[1]) - int(b.split("_")[1])) * 1.5 for a in path for b in path}
    metrics = VLNMetrics()
    metrics.update(NavigationResult(path, path, [path[-1]], dists))
    scores = metrics.compute()
    assert scores["SR"] == 100.0
    assert scores["SPL"] == 100.0


def test_r2r_loader_batch():
    from src.data.r2r_dataset import build_r2r_loaders

    batch = next(iter(build_r2r_loaders("missing", batch_size=2, num_workers=0)["train"]))
    assert batch["observations"].shape[0] == 2
