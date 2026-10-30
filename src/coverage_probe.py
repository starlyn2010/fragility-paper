#!/usr/bin/env python3
"""coverage_probe.py — Which agent-turn states does play against a teacher visit?

A deterministic optimal teacher induces a narrow state distribution; a twin
teacher induces a different one. This script measures the unique agent-turn
boards visited against M6 and against held-out twins, their overlap, and their
share of the 2,423 reachable agent-turn states.

Uso:
  python src/coverage_probe.py --config C --ckpt runs/fixed_C_s42 --games 200 \
      --out data/coverage_fixed_C_s42.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_ttt import CONFIGS  # noqa: E402
from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv  # noqa: E402
from oracle import Oracle  # noqa: E402
from space_probe import enumerate_agent_states  # noqa: E402
from tbr_eval import load_agent  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def visited(agent, env, opponent, n_games):
    boards = set()
    histories = set()
    for _ in range(n_games):
        hx = torch.zeros(1, agent.hidden_dim)
        obs = env.reset()
        done = False
        turn = 0
        hist = []
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            if turn % 2 == 0:
                board = tuple(int(v) for v in env._obs())
                boards.add(board)
                hist.append(board)
                histories.add(tuple(hist))
                legal_mask = torch.zeros(agent.action_dim, dtype=torch.bool)
                for a in legal:
                    legal_mask[a] = True
                with torch.no_grad():
                    action, hx = agent.get_action(obs, hx, legal_mask=legal_mask, deterministic=True)
                obs, _, done, _ = env.step(action, player=env.AGENTE)
            else:
                action = opponent.get_action(env, legal)
                obs, _, done, _ = env.step(action, player=env.HUMANO)
            turn += 1
    return boards, histories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C", choices=["B", "C"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--twin-seeds", default="1234,5678,9012")
    ap.add_argument("--out", default=str(ROOT / "data" / "coverage_probe.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    random.seed(20261027)
    np.random.seed(20261027)
    cfg = next(c for c in CONFIGS if c["id"] == args.config)
    path = pathlib.Path(args.ckpt)
    if path.is_dir():
        path = path / f"ablation_{args.config}.pth"
    agent = load_agent(cfg, path)
    env = TicTacToeEnv()
    oracle = Oracle()
    all_states = {tuple(s["board"]) for s in enumerate_agent_states()}

    m6 = StochasticMinimax(depth=6, noise_prob=0.0, tie_break="first")
    m6_set, m6_hist = visited(agent, env, m6, args.games)

    twin_set = set()
    twin_hist = set()
    per_twin = {}
    per_twin_hist = {}
    for seed in [int(x) for x in args.twin_seeds.split(",")]:
        opp = StochasticMinimax(depth=6, noise_prob=0.0, tie_break="random", seed=seed)
        s, h = visited(agent, env, opp, args.games)
        per_twin[str(seed)] = len(s)
        per_twin_hist[str(seed)] = len(h)
        twin_set |= s
        twin_hist |= h

    tie_in_m6 = sum(1 for b in m6_set if len(oracle.lookup(b)["optimal"]) > 1)
    tie_in_twins = sum(1 for b in twin_set if len(oracle.lookup(b)["optimal"]) > 1)

    res = {
        "checkpoint": str(path),
        "games_per_opponent": args.games,
        "n_reachable": len(all_states),
        "m6_unique": len(m6_set),
        "m6_unique_histories": len(m6_hist),
        "m6_share": len(m6_set) / len(all_states),
        "m6_tie_states": tie_in_m6,
        "twins_union_unique": len(twin_set),
        "twins_union_unique_histories": len(twin_hist),
        "twins_union_share": len(twin_set) / len(all_states),
        "twins_union_tie_states": tie_in_twins,
        "per_twin_unique": per_twin,
        "per_twin_unique_histories": per_twin_hist,
        "overlap": len(m6_set & twin_set),
        "twin_only": len(twin_set - m6_set),
        "jaccard": len(m6_set & twin_set) / max(1, len(m6_set | twin_set)),
    }
    print(json.dumps(res, indent=2))
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
