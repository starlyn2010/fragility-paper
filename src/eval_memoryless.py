#!/usr/bin/env python3
"""eval_memoryless.py — Evaluate a memoryless oracle-distilled policy in games.

Uso:
  python src/eval_memoryless.py --ckpt runs/memoryless_s0/mlp.pt --games 400 \
      --twins 30 --coherent 30 --out data/eval_memoryless_s0.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from liquid_rl_trainer import RandomAgent, StochasticMinimax, TicTacToeEnv  # noqa: E402
from human_like_agent import BoundedRationalMinimax  # noqa: E402
from memoryless import MLPPolicy  # noqa: E402
from tbr_eval import play  # noqa: E402
from oracle import Oracle  # noqa: E402
from twin_ensemble import play as play_fast  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--twins", type=int, default=30)
    ap.add_argument("--coherent", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "data" / "eval_memoryless.json"))
    args = ap.parse_args()

    agent = MLPPolicy.load(args.ckpt)
    env = TicTacToeEnv()
    oracle = Oracle()
    out = {"checkpoint": args.ckpt, "games": args.games, "results": {}}

    m6 = StochasticMinimax(depth=6, noise_prob=0.0, tie_break="first")
    out["results"]["m6_fixed"] = play(agent, env, m6, args.games, oracle)
    out["results"]["random_uniform"] = play(agent, env, RandomAgent(), args.games, oracle)
    out["results"]["weak_minimax"] = play(
        agent, env, BoundedRationalMinimax(depth_normal=2, depth_tension=4, noise_std=0.5),
        args.games, oracle)

    for name, tb, n in (("twin_random", "random", args.twins),
                        ("twin_coherent", "episode", args.coherent)):
        drs = []
        for i in range(n):
            opp = StochasticMinimax(depth=6, noise_prob=0.0, tie_break=tb, seed=1000 + i)
            r = play_fast(agent, env, opp, 100)
            drs.append(r["dr"])
        out["results"][name] = {
            "n": n, "games_each": 100,
            "dr_mean": float(np.mean(drs)), "dr_min": float(np.min(drs)),
            "dr_max": float(np.max(drs)), "n_at_100": int(np.sum(np.array(drs) >= 1.0)),
        }
        print(f"{name}: mean {np.mean(drs)*100:.1f} min {np.min(drs)*100:.1f} "
              f"max {np.max(drs)*100:.1f} at100 {out['results'][name]['n_at_100']}/{n}")

    s = out["results"]
    print(f"M6: D {s['m6_fixed']['dr']*100:.1f}% regret {s['m6_fixed']['mean_regret']:.4f} | "
          f"random W {s['random_uniform']['wr']*100:.1f}% | weak W {s['weak_minimax']['wr']*100:.1f}%")
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Crudo: {p}")


if __name__ == "__main__":
    main()
