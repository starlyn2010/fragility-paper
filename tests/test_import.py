"""Smoke tests — verify core modules import and configs validate."""
import pathlib
import json
import ast

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"

def test_import_core_modules():
    import importlib.util
    import sys
    sys.path.insert(0, str(SRC))
    # Only exec lightweight env/agent modules — avoid generate_figures side effects
    for name in ["connect4_env", "human_like_agent", "minimax_mentor"]:
        spec = importlib.util.spec_from_file_location(name, SRC / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod is not None
    # For scripts with top-level side effects, just verify they parse
    for name in ["generate_figures", "liquid_rl_trainer", "ablation_ttt", "tournament"]:
        text = (SRC / f"{name}.py").read_text()
        ast.parse(text)
        assert len(text) > 100

def test_data_files_exist_and_valid_json():
    expected = [
        "ablation_results.json",
        "tournament_results.json",
        "cross_eval_results.json",
        "transfer_results.json",
        "benchmark_act_results.json",
    ]
    for fname in expected:
        p = DATA / fname
        assert p.exists(), f"missing {fname}"
        data = json.loads(p.read_text())
        assert isinstance(data, (dict, list))

def test_paper_assets():
    assert (ROOT / "paper" / "paper.tex").exists()
    assert (ROOT / "paper" / "fig1_nontransitivity.pdf").exists()
    assert (ROOT / "paper" / "fig2_transfer_curves.pdf").exists()
    tex = (ROOT / "paper" / "paper.tex").read_text()
    assert "Fragility of Optimal-Agent Training" in tex

def test_citation_and_license():
    assert (ROOT / "CITATION.cff").exists()
    assert (ROOT / "LICENSE").exists()
    assert (ROOT / "pyproject.toml").exists()
    assert (ROOT / ".github" / "workflows" / "ci.yml").exists()

def test_liquid_trainer_import():
    text = (SRC / "liquid_rl_trainer.py").read_text()
    ast.parse(text)
    assert "TrainingConfig" in text
    assert "Closed-form" in text or "CfC" in text
