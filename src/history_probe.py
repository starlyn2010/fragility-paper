#!/usr/bin/env python3
"""history_probe.py — Exact regret over agent-turn *histories*, not boards.

A recurrent policy's state is the ordered sequence of boards it has seen, not
the current board. Two move orders can reach the same board with different
hidden states. `space_probe.py` deduplicates by board, so it can certify a policy
that fails on the alternative histories induced by a different tie-break.

This script enumerates unique agent-turn histories and computes exact minimax
regret for each, using the same recurrence as evaluation.

Uso:
  python src/history_probe.py --config C --runs "label=runs/dir;..." \
      --out data/history_probe.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_ttt import CONFIGS, AblationLiquidAgent  # noqa: E402
from oracle import Oracle, env_from_board  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def enumerate_histories():
    """All unique agent-turn histories (ordered tuples of boards)."""
    start = ((0,) * 9, 0, ())
    seen = {(start[0], start[1], start[2])}
    stack = [start]
    out = []
    while stack:
        board, turn, hist = stack.pop()
        env = env_from_board(board)
        if env.done:
            continue
        legal = env.legal_actions()
        is_agent = turn % 2 == 0
        hist_now = hist + (board,) if is_agent else hist
        if is_agent:
            out.append({"board": board, "history": hist_now, "moves": turn})
        for a in legal:
            nxt = env.clone()
            nxt.step(a, player=env.AGENTE if is_agent else env.HUMANO)
            nb = tuple(int(v) for v in nxt._obs())
            key = (nb, turn + 1, hist_now)
            if key not in seen:
                seen.add(key)
                stack.append((nb, turn + 1, hist_now))
    return out


def summary(rows):
    if not rows:
        return {}
    reg = np.array([r["regret"] for r in rows], dtype=float)
    tie = [r for r in rows if r["tie"]]
    return {
        "n": len(rows),
        "mean_regret": float(reg.mean()),
        "max_regret": float(reg.max()),
        "optimal_action_rate": float(np.mean([r["optimal"] for r in rows])),
        "tie_n": len(tie),
        "tie_mean_regret": float(np.mean([r["regret"] for r in tie])) if tie else float("nan"),
        "tie_optimal_action_rate": float(np.mean([r["optimal"] for r in tie])) if tie else float("nan"),
    }


def batched_step(agent, boards, hx, chunk=4096):
    """Run one recurrent step for a batch of boards; returns masked probs and hx."""
    probs_out = []
    hx_out = []
    for i in range(0, len(boards), chunk):
        b = boards[i:i + chunk]
        obs_t = torch.tensor(b, dtype=torch.float32)
        legal = torch.tensor([[v == 0 for v in board] for board in b], dtype=torch.bool)
        with torch.no_grad():
            if getattr(agent, "s12_enabled", False):
                logits, _, hx_new, _, _ = agent.forward_adaptive(obs_t, hx[i:i + chunk])
            else:
                logits, _, hx_new = agent.forward_step(obs_t, hx[i:i + chunk])
            logits = logits.masked_fill(~legal, -float("inf"))
            probs_out.append(torch.softmax(logits, dim=-1))
            hx_out.append(hx_new)
    return torch.cat(probs_out, dim=0), torch.cat(hx_out, dim=0)


def probe_agent(agent, histories):
    """Exact regret over all histories, batching steps by history length.

    The hidden state after k agent turns depends only on the ordered sequence of
    k boards. Grouping by k lets us advance every history one step in a single
    batched GRU call, instead of replaying each prefix from scratch.
    """
    by_len: dict[int, list[dict]] = {}
    for h in histories:
        by_len.setdefault(len(h["history"]), []).append(h)

    rows = []
    prev_index: dict[tuple, int] = {}
    prev_hx = None
    for k in sorted(by_len):
        group = by_len[k]
        boards = [h["history"][-1] for h in group]
        if k == 1:
            hx_in = torch.zeros(len(group), agent.hidden_dim)
        else:
            hx_in = torch.stack([prev_hx[prev_index[h["history"][:-1]]] for h in group])
        probs, hx_out = batched_step(agent, boards, hx_in)
        chosen = probs.argmax(dim=-1).tolist()
        for h, action in zip(group, chosen):
            info = h["oracle"]
            rows.append({
                "regret": max(0.0, info["best"] - info["values"][action]),
                "optimal": int(action in info["optimal"]),
                "tie": h["tie"],
            })
        prev_hx = hx_out
        prev_index = {h["history"]: j for j, h in enumerate(group)}
    return summary(rows)


def history_actions(agent, histories):
    """Greedy action for every history, keyed by the ordered board sequence."""
    by_len: dict[int, list[dict]] = {}
    for h in histories:
        by_len.setdefault(len(h["history"]), []).append(h)

    table: dict[tuple, int] = {}
    prev_index: dict[tuple, int] = {}
    prev_hx = None
    for k in sorted(by_len):
        group = by_len[k]
        boards = [h["history"][-1] for h in group]
        if k == 1:
            hx_in = torch.zeros(len(group), agent.hidden_dim)
        else:
            hx_in = torch.stack([prev_hx[prev_index[h["history"][:-1]]] for h in group])
        probs, hx_out = batched_step(agent, boards, hx_in)
        chosen = probs.argmax(dim=-1).tolist()
        for h, action in zip(group, chosen):
            table[h["history"]] = int(action)
        prev_hx = hx_out
        prev_index = {h["history"]: j for j, h in enumerate(group)}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C", choices=["B", "C"])
    ap.add_argument("--runs", required=True, help="'label=ckpt_dir;...'")
    ap.add_argument("--out", default=str(ROOT / "data" / "history_probe.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = next(c for c in CONFIGS if c["id"] == args.config)
    histories = enumerate_histories()
    boards = {h["board"] for h in histories}
    oracle = Oracle()
    for h in histories:
        info = oracle.lookup(h["board"])
        h["oracle"] = info
        h["tie"] = len(info["optimal"]) >= 2
    print(f"historias={len(histories)} tableros={len(boards)} "
          f"empates={sum(h['tie'] for h in histories)}", flush=True)

    out = {
        "config": args.config,
        "n_histories": len(histories),
        "n_boards": len(boards),
        "n_tie_histories": int(sum(h["tie"] for h in histories)),
        "results": {},
    }

    for item in args.runs.split(";"):
        item = item.strip()
        if not item:
            continue
        label, ckpt_dir = item.split("=", 1)
        path = pathlib.Path(ckpt_dir.strip()) / f"ablation_{args.config}.pth"
        if not path.exists():
            print(f"[skip] {label}: falta {path}")
            continue
        agent = AblationLiquidAgent(args.config, 9, 9, cfg["hidden_dim"])
        agent.s12_enabled = cfg["act"]
        agent.s12_max_iters = 3
        agent.load_state_dict(torch.load(str(path), map_location="cpu"))
        agent.eval()
        res = probe_agent(agent, histories)
        res["sha256_16"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        out["results"][label] = res
        print(f"{label:>18} n={res['n']} regret {res['mean_regret']:.4f} "
              f"opt {res['optimal_action_rate']:.3f} | tie {res['tie_mean_regret']:.4f} "
              f"opt {res['tie_optimal_action_rate']:.3f}", flush=True)
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nCrudo: {args.out}")


if __name__ == "__main__":
    main()
