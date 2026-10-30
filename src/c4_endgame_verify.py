#!/usr/bin/env python3
"""c4_endgame_verify.py — independent check of a 7x6 endgame-repaired policy.

Reloads the sampled endgame suite, recomputes the worst/best/uniform draw rate
with a memo-free recursion, and simulates deterministic and adaptive optimal
teachers. Usage:
  python src/c4_endgame_verify.py --ckpt runs/c4_end_ctrl_s0/c4_gru.pt \
      --roots data/c4_endgame_roots_8_40_s0.json --hidden 128 --emb 16 \
      --orders 120 --games 40 --out data/c4_end_verify_s0.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c4_game as g
from c4_repair import GRUPolicy

ROOT = pathlib.Path(__file__).resolve().parent.parent


def certify_root(policy, oracle, root):
    b0, h0 = tuple(root["board"]), tuple(root["prefix"])

    def worst(b, h):
        if g.terminal(b):
            return 1.0 if g.winner(b) != 2 else 0.0
        if g.turn(b) == 1:
            a = policy.act(h, b)
            return worst(g.drop(b, a, 1), h + (a,))
        return min(worst(g.drop(b, c, 2), h + (c,)) for c in oracle.optimal(b))

    def mean(b, h):
        if g.terminal(b):
            return 1.0 if g.winner(b) != 2 else 0.0
        if g.turn(b) == 1:
            a = policy.act(h, b)
            return mean(g.drop(b, a, 1), h + (a,))
        opts = oracle.optimal(b)
        return sum(mean(g.drop(b, c, 2), h + (c,)) for c in opts) / len(opts)

    return worst(b0, h0), mean(b0, h0)


def play(policy, oracle, root, order=None, rng=None):
    b, h = tuple(root["board"]), tuple(root["prefix"])
    while not g.terminal(b):
        if g.turn(b) == 1:
            c = policy.act(h, b)
        else:
            opts = oracle.optimal(b)
            c = (next(x for x in order if x in opts)
                 if order is not None else rng.choice(opts))
        b, h = g.drop(b, c, g.turn(b)), h + (c,)
    return 1 if g.winner(b) != 2 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--roots", required=True)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--win", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--orders", type=int, default=120)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "data" / "c4_endgame_verify.json"))
    args = ap.parse_args()

    g.configure(args.rows, args.cols, args.win)
    oracle = g.Oracle()
    roots = json.loads(pathlib.Path(args.roots).read_text(encoding="utf-8"))
    policy = GRUPolicy(args.hidden, args.emb)
    policy.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    policy.eval()

    per = [certify_root(policy, oracle, r) for r in roots]
    ws = [w for w, _ in per]
    ds = [d for _, d in per]
    rng = random.Random(args.seed)
    orders = [tuple(rng.sample(range(g.COLS), g.COLS)) for _ in range(args.orders)]
    order_rates = [sum(play(policy, oracle, r, order=o) for r in roots) / len(roots)
                   for o in orders]
    adaptive = [sum(play(policy, oracle, r, rng=rng) for _ in range(args.games)) / args.games
                for r in roots]
    out = {
        "ckpt": str(args.ckpt),
        "roots": len(roots),
        "worst_draw_rate": min(ws),
        "best_draw_rate": max(ws),
        "random_draw_rate_exact": sum(ds) / len(ds),
        "n_roots_drawn_worst": sum(1 for w in ws if w == 1.0),
        "deterministic_orders": len(orders),
        "order_mean": sum(order_rates) / len(order_rates),
        "order_min": min(order_rates),
        "order_max": max(order_rates),
        "adaptive_games_per_root": args.games,
        "adaptive_mean": sum(adaptive) / len(adaptive),
        "adaptive_min": min(adaptive),
    }
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
