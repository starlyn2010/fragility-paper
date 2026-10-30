#!/usr/bin/env python3
"""c4_verify.py — independent re-certification of a repaired Connect Three policy.

Re-derives the worst case with a separate, memo-free recursion and also plays
all deterministic column-order teachers plus randomly adaptive optimal
teachers. Usage:
  python src/c4_verify.py --ckpt runs/c4_canon_s0/c4_gru.pt \
      --rows 4 --cols 3 --win 3 --hidden 64 --out data/c4_verify_s0.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c4_game as g
from c4_repair import GRUPolicy

ROOT = pathlib.Path(__file__).resolve().parent.parent


def worst_case(policy, oracle):
    """Memo-free backward induction (independent of c4_game.certify)."""
    def rec(board, history):
        w = g.winner(board)
        if w != 0 or not g.legal(board):
            return 1.0 if w != 2 else 0.0
        if g.turn(board) == 1:
            a = policy.act(history, board)
            if a not in g.legal(board):
                return 0.0
            return rec(g.drop(board, a, 1), history + (a,))
        vals = [rec(g.drop(board, c, 2), history + (c,))
                for c in oracle.optimal(board)]
        return min(vals)

    return rec(g.EMPTY, ())


def random_adaptive_teacher(policy, oracle, games: int, seed: int) -> float:
    rng = random.Random(seed)
    draws = 0
    for _ in range(games):
        b, h = g.EMPTY, ()
        while not g.terminal(b):
            p = g.turn(b)
            a = policy.act(h, b) if p == 1 else rng.choice(oracle.optimal(b))
            b = g.drop(b, a, p)
            h = h + (a,)
        draws += 1 if g.winner(b) != 2 else 0
    return draws / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--win", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "c4_verify.json"))
    args = ap.parse_args()

    g.configure(args.rows, args.cols, args.win)
    oracle = g.Oracle()
    oracle.all_boards()
    policy = GRUPolicy(args.hidden)
    policy.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    policy.eval()

    t0 = time.time()
    w = worst_case(policy, oracle)
    orders = g.teacher_orders()
    order_draws = [g.evaluate_teacher(lambda h: policy.act(h), oracle, o) for o in orders]
    adaptive = random_adaptive_teacher(policy, oracle, args.games, args.seed)
    out = {
        "ckpt": str(args.ckpt),
        "game": {"rows": args.rows, "cols": args.cols, "win": args.win},
        "exhaustive_worst_case": w,
        "order_draw_rates": order_draws,
        "order_mean": sum(order_draws) / len(order_draws),
        "adaptive_teacher_draw_rate": adaptive,
        "adaptive_games": args.games,
        "elapsed_s": time.time() - t0,
    }
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
