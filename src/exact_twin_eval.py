#!/usr/bin/env python3
"""exact_twin_eval.py — Twin control with exact (depth-9) optimal teachers.

The depth-6 teacher is a heuristic: on some teacher-turn boards its optimal set
differs from the exact depth-9 oracle. This script removes that confound. At
every teacher turn it computes the exact optimal action set with depth-9
minimax and then either

  * plays the first optimal action (the exact deterministic teacher), or
  * samples uniformly from the exact optimal set (an exact twin).

The exact deterministic teacher and the exact twins have the same value function
everywhere; they differ only in tie-breaking. If the twin gap survives this
control, it is not an artifact of depth-6 approximation error.

Uso:
  python src/exact_twin_eval.py --runs "v1_C=artifacts/v1_reentreno" \
      --config C --games 200 --twin-seeds 1234,5678,9012 \
      --out data/exact_twin_C.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, __import__("os").path.dirname(__file__))

from ablation_ttt import CONFIGS  # noqa: E402
from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv  # noqa: E402
from oracle import Oracle, env_from_board  # noqa: E402
from tbr_eval import load_agent, play  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


class ExactTeacher:
    """Depth-9 exact teacher at teacher turns, with a shared optimal-set cache."""

    _mentor = None
    _cache: dict[tuple[int, ...], tuple[int, ...]] = {}

    def __init__(self, seed: int, mode: str = "random"):
        if ExactTeacher._mentor is None:
            ExactTeacher._mentor = StochasticMinimax(
                mentor_player=-1, depth=9, noise_prob=0.0, tie_break="first"
            )
        self.mentor = ExactTeacher._mentor
        self.rng = random.Random(seed)
        self.mode = mode

    def optimal(self, board: tuple[int, ...]) -> tuple[int, ...]:
        hit = ExactTeacher._cache.get(board)
        if hit is not None:
            return hit
        env = env_from_board(board)
        values = {}
        for a in env.legal_actions():
            c = env.clone()
            c.step(a, player=env.HUMANO)
            values[a] = float(self.mentor._minimax_exact(c, 9, False))
        best = max(values.values())
        opt = tuple(a for a, v in values.items() if abs(v - best) < 1e-9)
        ExactTeacher._cache[board] = opt
        return opt

    def get_action(self, env, legal):
        board = tuple(int(v) for v in env._obs())
        opt = [a for a in self.optimal(board) if a in legal]
        if not opt:
            opt = legal
        if self.mode == "first":
            return opt[0]
        return self.rng.choice(opt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="'label=ckpt_dir;...'")
    ap.add_argument("--config", default="C", choices=["A", "B", "C", "D"])
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--twin-seeds", default="1234,5678,9012")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default=str(ROOT / "data" / "exact_twin_eval.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    oracle = Oracle()
    cfg = next(c for c in CONFIGS if c["id"] == args.config)
    env = TicTacToeEnv()
    twin_seeds = [int(s) for s in args.twin_seeds.split(",") if s.strip()]

    runs = []
    for item in args.runs.split(";"):
        item = item.strip()
        if not item:
            continue
        label, path = item.split("=", 1)
        runs.append((label.strip(), pathlib.Path(path.strip())))

    out = {
        "args": vars(args),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "when_utc": datetime.now(timezone.utc).isoformat(),
            "threads": torch.get_num_threads(),
        },
        "config": cfg,
        "twin_seeds": twin_seeds,
        "results": {},
    }

    for label, ckpt_dir in runs:
        path = ckpt_dir / f"ablation_{cfg['id']}.pth"
        if not path.exists():
            print(f"[skip] {label}: falta {path}")
            continue
        agent = load_agent(cfg, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"\n=== {label} ({cfg['id']}) sha256:{sha} ===")
        row = {"checkpoint": str(path), "sha256_16": sha, "opponents": {}}
        t0 = time.time()
        opp_first = ExactTeacher(0, mode="first")
        row["opponents"]["exact_first"] = play(agent, env, opp_first, args.games, oracle)
        for seed in twin_seeds:
            opp = ExactTeacher(seed, mode="random")
            row["opponents"][f"exact_twin_{seed}"] = play(agent, env, opp, args.games, oracle)

        first = row["opponents"]["exact_first"]
        twin_drs = [row["opponents"][f"exact_twin_{s}"]["dr"] for s in twin_seeds]
        twin_reg = [row["opponents"][f"exact_twin_{s}"]["mean_regret"] for s in twin_seeds]
        twin_opt = [row["opponents"][f"exact_twin_{s}"]["optimal_action_rate"] for s in twin_seeds]
        row["summary"] = {
            "dr_exact_first": first["dr"],
            "regret_exact_first": first["mean_regret"],
            "opt_exact_first": first["optimal_action_rate"],
            "dr_exact_twin_mean": float(np.mean(twin_drs)),
            "dr_exact_twin_min": float(np.min(twin_drs)),
            "dr_exact_twin_max": float(np.max(twin_drs)),
            "twin_gap_mean": float(np.mean(twin_drs) - first["dr"]),
            "regret_exact_twin_mean": float(np.mean(twin_reg)),
            "opt_exact_twin_mean": float(np.mean(twin_opt)),
            "elapsed_s": round(time.time() - t0, 1),
        }
        out["results"][label] = row
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        s = row["summary"]
        print(
            f"  exact first: DR {s['dr_exact_first']*100:.1f}% regret {s['regret_exact_first']:.4f} | "
            f"exact twins: DR {s['dr_exact_twin_mean']*100:.1f}% "
            f"[{s['dr_exact_twin_min']*100:.1f},{s['dr_exact_twin_max']*100:.1f}] "
            f"regret {s['regret_exact_twin_mean']:.4f} | gap {s['twin_gap_mean']*100:+.1f}",
            flush=True,
        )

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nCrudo: {p}")


if __name__ == "__main__":
    main()
