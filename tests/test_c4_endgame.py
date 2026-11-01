"""Contract tests for the 7x6 endgame scaling harness.

The published scaling arms (8, 10, and the Colab 12/14-cell arms) all run
src/c4_endgame.py on a sampled root suite, so these tests pin the pieces the
result depends on: the root sampler keeps only positions with the requested
exact value, the certificate enumerates every teacher tie-break inside each
root, the matched control pools are learner-turn histories, the aggregate
covers every violation the base certificate reported, and the CLI writes the
JSON keys the paper and README quote. Everything runs on a 4x4 board with a
few roots, so the oracle stays lazy and the suite stays fast.

Run with pytest, or directly: python tests/test_c4_endgame.py
"""
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import c4_game as g
import c4_endgame as eg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 4x4 with six empty cells: draw roots are common and the oracle stays lazy.
ROWS, COLS, WIN, EMPTIES = 4, 4, 4, 6


def _setup(roots=3, seed=0):
    g.configure(ROWS, COLS, WIN)
    oracle = g.Oracle()
    suite, _games = eg.random_games(oracle, EMPTIES, roots, seed, 0)
    return oracle, suite


def _board_after(history):
    board = g.EMPTY
    for c in history:
        board = g.drop(board, c, g.turn(board))
    return board


class _StubPolicy:
    """Chooses a legal column by rule, ignoring the history."""

    def __init__(self, oracle, chooser):
        self.oracle = oracle
        self.chooser = chooser

    def act(self, history, board):
        return self.chooser(board, self.oracle.optimal(board), g.legal(board))


def _optimal(board, opts, legal):
    return opts[0]


def _avoid_optimal(board, opts, legal):
    bad = [c for c in legal if c not in opts]
    return bad[-1] if bad else opts[0]


def test_random_games_keeps_only_the_requested_value():
    oracle, suite = _setup(roots=4)
    assert len(suite) == 4
    seen = set()
    for root in suite:
        board = tuple(root["board"])
        assert len(root["prefix"]) == g.N - EMPTIES
        assert not g.terminal(board)
        assert root["value"] == 0 == oracle.value(board)
        assert _board_after(root["prefix"]) == board
        assert board not in seen
        seen.add(board)


def test_random_games_is_deterministic_in_the_seed():
    g.configure(ROWS, COLS, WIN)
    oracle = g.Oracle()
    a, _ = eg.random_games(oracle, EMPTIES, 3, 5, 0)
    b, _ = eg.random_games(oracle, EMPTIES, 3, 5, 0)
    assert [r["prefix"] for r in a] == [r["prefix"] for r in b]


def test_root_items_are_learner_turn_histories():
    oracle, suite = _setup()
    order = tuple(range(g.COLS))
    for root in suite:
        items = eg.root_items(root, oracle, order, random.Random(1), 2)
        assert items
        for h in items:
            board = _board_after(h)
            assert not g.terminal(board)
            assert g.turn(board) == 1
            assert len(h) >= len(root["prefix"])


def test_certify_accepts_a_policy_that_always_plays_optimally():
    oracle, suite = _setup()
    cert = eg.certify(_StubPolicy(oracle, _optimal), oracle, suite)
    assert cert["worst_draw_rate"] == 1.0
    assert cert["best_draw_rate"] == 1.0
    assert cert["random_draw_rate_exact"] == 1.0
    assert cert["n_roots"] == len(suite)
    assert cert["n_roots_drawn_worst"] == len(suite)
    assert cert["n_violations"] == 0


def test_certify_flags_a_suboptimal_policy_once_per_history():
    oracle, suite = _setup()
    cert = eg.certify(_StubPolicy(oracle, _avoid_optimal), oracle, suite)
    histories = [tuple(v["history"]) for v in cert["violations"]]
    assert histories, "a policy that avoids optimal columns must be flagged"
    assert len(histories) == len(set(histories))
    for v in cert["violations"]:
        assert v["action"] not in v["optimal"]
        assert 0 <= v["root"] < len(suite)
        assert g.turn(_board_after(v["history"])) == 1


def test_certificate_bounds_are_ordered_per_root():
    """The worst tie-break never beats the uniform average, root by root."""
    oracle, suite = _setup(roots=4)
    cert = eg.certify(_StubPolicy(oracle, _avoid_optimal), oracle, suite)
    assert 0.0 <= cert["worst_draw_rate"] <= cert["random_draw_rate_exact"] <= 1.0
    assert cert["worst_draw_rate"] <= cert["best_draw_rate"]
    assert len(cert["per_root"]) == len(suite)
    for r in cert["per_root"]:
        assert r["w"] <= r["d"] + 1e-12


def test_control_pools_hold_constrained_learner_histories():
    oracle, suite = _setup()
    for mode in ("all", "tree"):
        pool = eg.make_pool(suite, oracle, mode, seed=2026, games=2)
        assert pool
        for h in pool:
            board = _board_after(h)
            assert not g.terminal(board)
            assert g.turn(board) == 1
            # "constrained" means the teacher had no single forced reply
            assert len(oracle.optimal(board)) < len(g.legal(board))


def test_cli_writes_the_documented_result_contract(tmp_path=None):
    out_dir = str(tmp_path) if tmp_path is not None else os.path.join(
        REPO, "runs", "_test_endgame")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "c4_endgame_test.json")
    roots_out = os.path.join(REPO, "data", f"c4_endgame_roots_{EMPTIES}_2_s987.json")
    base_cmd = [sys.executable, os.path.join(REPO, "src", "c4_endgame.py"),
                "--rows", str(ROWS), "--cols", str(COLS), "--win", str(WIN),
                "--empties", str(EMPTIES), "--root-value", "0", "--branch", "2",
                "--hidden", "16", "--emb", "8", "--epochs", "25",
                "--agg-repeat", "2", "--stop", "both", "--max-iters", "2",
                "--seed", "987", "--threads", "1", "--no-controls",
                "--ckpt-out", out_dir]
    try:
        if os.path.exists(roots_out):
            os.remove(roots_out)
        subprocess.run(base_cmd + ["--roots", "2", "--out", out],
                       check=True, capture_output=True, text=True, cwd=REPO)
        data = json.loads(open(out, encoding="utf-8").read())
        for key in ("args", "environment", "game", "replay_histories", "base",
                    "repair", "controls", "elapsed_s"):
            assert key in data, key
        assert data["controls"] == []
        assert data["game"]["roots"] == 2 and data["game"]["root_value_0"] == 2
        assert data["replay_histories"] > 0
        assert data["base"]["certificate"]["n_violations"] == \
            len(data["base"]["certificate"]["violations"])
        assert len(data["repair"]["log"]) >= 2
        assert data["repair"]["final"]["worst_draw_rate"] in (0.0, 1.0)
        assert len(data["repair"]["per_root"]) == 2
        assert os.path.exists(os.path.join(out_dir, "c4_gru.pt"))

        # Core CGR contract: every base violation is a candidate for the aggregate.
        added = {tuple(h) for h in data["repair"]["aggregate"]}
        for v in data["base"]["certificate"]["violations"]:
            assert tuple(v["history"]) in added

        # The roots file is what makes a per-seed comparison matched.
        reloaded = json.loads(open(roots_out, encoding="utf-8").read())
        assert len(reloaded) == 2
        assert all(r["value"] == 0 and not g.terminal(tuple(r["board"]))
                   for r in reloaded)

        second = os.path.join(out_dir, "c4_endgame_test2.json")
        subprocess.run(base_cmd + ["--roots", "2", "--roots-file", roots_out,
                                   "--out", second],
                       check=True, capture_output=True, text=True, cwd=REPO)
        again = json.loads(open(second, encoding="utf-8").read())
        assert again["game"]["roots"] == 2
        assert again["replay_histories"] == data["replay_histories"]
    finally:
        if os.path.exists(roots_out):
            os.remove(roots_out)
        if tmp_path is None:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
