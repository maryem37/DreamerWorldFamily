from pathlib import Path


def test_project_structure_exists():
    root = Path(__file__).parent.parent
    for rel in [
        "README.md",
        "train.py",
        "evaluate.py",
        "inference.py",
        "app.py",
        "src/models/text_dreamer.py",
        "src/models/x_dreamer.py",
        "src/utils/mesh_generator.py",
        "demo/index.html",
        "configs/atd_r2r.yaml",
    ]:
        assert (root / rel).exists()


def test_mesh_generator_writes_assets(tmp_path):
    from src.utils.mesh_generator import generate_mesh

    asset = generate_mesh("A marble bust of an angel", tmp_path)
    assert asset.obj_path.exists()
    assert asset.mtl_path.exists()
    assert asset.stl_path.exists()
    assert asset.preview_path.exists()
    assert asset.metadata_path.exists()
    assert asset.vertices > 0
    assert asset.faces > 0
    assert asset.quality["watertight"] is True
    assert "v " in asset.obj_path.read_text(encoding="utf-8")


def test_r2r_json_evaluator(tmp_path):
    from src.utils.r2r_eval import evaluate_r2r_file, write_example_r2r

    example = write_example_r2r(tmp_path / "r2r.json")
    result = evaluate_r2r_file(example)
    scores = result["scores"]
    assert result["n_records"] == 2
    assert scores["SR"] == 100.0
    assert 0.0 < scores["SPL"] <= 100.0
