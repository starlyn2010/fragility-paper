#!/usr/bin/env python3
"""distill_oracle.py — Controlled coverage experiment.

The twin-mentor gap is hypothesised to be a coverage failure: a deterministic
optimal teacher exposes the learner to a tiny subset of the optimal state space.
To test this causally, we train the same network by supervised distillation from
the exact minimax oracle on nested subsets of the 2,423 reachable agent-turn
states. The teacher's tie-break plays no role here; only coverage varies.

If coverage is the mechanism, twin robustness should rise with the number of
unique states, reaching 100% at full coverage.

Uso:
  python src/distill_oracle.py --config C --coverage 25,100,400,1200,2423 \
      --seeds 0,1 --epochs 400 --out data/distill_C.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_ttt import CONFIGS, AblationLiquidAgent  # noqa: E402
from oracle import Oracle  # noqa: E402
from space_probe import enumerate_agent_states, policy_probs  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def prepare_states():
    states = enumerate_agent_states()
    oracle = Oracle()
    for s in states:
        s["oracle"] = oracle.lookup(s["board"])
        s["tie"] = len(s["oracle"]["optimal"]) >= 2
        s["k_optimal"] = len(s["oracle"]["optimal"])
    return states


def group_by_length(states):
    groups: dict[int, list[dict]] = {}
    for s in states:
        groups.setdefault(len(s["history"]), []).append(s)
    return groups


def batch_tensors(items):
    """Return list of (obs_t, legal_mask_t, optimal_idx, best_value)."""
    L = len(items[0]["history"])
    steps = []
    for t in range(L):
        boards = np.stack([it["history"][t] for it in items]).astype(np.float32)
        obs = torch.from_numpy(boards)
        mask = obs.eq(0.0)
        steps.append((obs, mask))
    optimal = [torch.tensor(it["oracle"]["optimal"], dtype=torch.long) for it in items]
    best = torch.tensor([it["oracle"]["best"] for it in items], dtype=torch.float32)
    return steps, optimal, best


def forward_sequence(agent, steps, hidden_dim):
    hx = torch.zeros(len(steps[0][0]), hidden_dim)
    logits = value = None
    for obs, _ in steps:
        logits, value, hx = agent.forward_step(obs, hx)
    return logits, value


def train(agent, states, epochs, batch_size, lr, seed, max_minutes=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    groups = group_by_length(states)
    order_rng = np.random.default_rng(seed + 1000)
    t0 = time.time()
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for L, items in groups.items():
            idx = order_rng.permutation(len(items))
            for start in range(0, len(items), batch_size):
                batch = [items[i] for i in idx[start:start + batch_size]]
                steps, optimal, best = batch_tensors(batch)
                logits, value = forward_sequence(agent, steps, agent.hidden_dim)
                legal = steps[-1][1]
                logits = logits.masked_fill(~legal, -1e9)
                logp = torch.log_softmax(logits, dim=-1)
                pi_loss = torch.stack([
                    -logp[i, a].mean() for i, a in enumerate(optimal)
                ]).mean()
                v_loss = torch.nn.functional.mse_loss(value, best)
                loss = pi_loss + 0.25 * v_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()
                total += float(loss.detach()) * len(batch)
                n += len(batch)
        if (epoch + 1) % 100 == 0:
            print(f"    epoch {epoch+1:4d} loss {total / max(n, 1):.4f}", flush=True)
        if max_minutes > 0 and (time.time() - t0) / 60.0 > max_minutes:
            print(f"    time limit {max_minutes} min reached at epoch {epoch+1}", flush=True)
            break


@torch.no_grad()
def full_space_metrics(agent, states):
    rows = []
    for s in states:
        hx = torch.zeros(1, agent.hidden_dim)
        probs = None
        for board in s["history"]:
            legal = [i for i, v in enumerate(board) if v == 0]
            probs, hx = policy_probs(agent, board, hx, legal)
        chosen = int(np.argmax(probs))
        regret = max(0.0, s["oracle"]["best"] - s["oracle"]["values"][chosen])
        rows.append({
            "regret": regret,
            "optimal": int(chosen in s["oracle"]["optimal"]),
            "tie": s["tie"],
            "k": s["k_optimal"],
        })
    reg = np.array([r["regret"] for r in rows], dtype=float)
    tie = [r for r in rows if r["tie"]]
    return {
        "n": len(rows),
        "mean_regret": float(reg.mean()),
        "median_regret": float(np.median(reg)),
        "max_regret": float(reg.max()),
        "optimal_action_rate": float(np.mean([r["optimal"] for r in rows])),
        "tie_mean_regret": float(np.mean([r["regret"] for r in tie])),
        "tie_optimal_action_rate": float(np.mean([r["optimal"] for r in tie])),
        "notie_mean_regret": float(np.mean([r["regret"] for r in rows if not r["tie"]])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C", choices=["B", "C"])
    ap.add_argument("--coverage", default="25,100,400,1200,2423")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=0,
                    help="override hidden size (0 = config default)")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop after this many minutes per checkpoint (0 = off)")
    ap.add_argument("--out", default=str(ROOT / "data" / "distill.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = next(c for c in CONFIGS if c["id"] == args.config)
    hidden = args.hidden or cfg["hidden_dim"]
    states = prepare_states()
    # Nested subsets: the same permutation for every coverage level.
    perm = np.random.default_rng(7).permutation(len(states))
    levels = [int(x) for x in args.coverage.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    out = {
        "args": vars(args),
        "hidden": hidden,
        "n_states": len(states),
        "n_tie_states": int(sum(s["tie"] for s in states)),
        "results": {},
    }

    for n_cov in levels:
        subset = [states[i] for i in perm[:min(n_cov, len(states))]]
        for seed in seeds:
            suffix = "" if hidden == cfg["hidden_dim"] else f"_h{hidden}"
            label = f"{args.config}_cov{n_cov}_s{seed}{suffix}"
            ckpt_dir = ROOT / "runs" / f"distill_{label}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"ablation_{args.config}.pth"
            agent = AblationLiquidAgent(args.config, 9, 9, hidden)
            agent.s12_enabled = cfg["act"]
            agent.s12_max_iters = 3
            print(f"\n=== {label}: {len(subset)} states, {args.epochs} epochs ===", flush=True)
            train(agent, subset, args.epochs, args.batch_size, args.lr, seed,
                  max_minutes=args.max_minutes)
            agent.eval()
            torch.save(agent.state_dict(), str(path))
            m = full_space_metrics(agent, states)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            out["results"][label] = {
                "coverage": len(subset),
                "seed": seed,
                "params": int(sum(p.numel() for p in agent.parameters())),
                "checkpoint": str(path),
                "sha256_16": sha,
                **m,
            }
            print(f"  {label:>16} regret {m['mean_regret']:.4f} "
                  f"opt {m['optimal_action_rate']:.3f} "
                  f"tie_regret {m['tie_mean_regret']:.4f} "
                  f"tie_opt {m['tie_optimal_action_rate']:.3f}", flush=True)
            p = pathlib.Path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nCrudo: {args.out}")


if __name__ == "__main__":
    main()
