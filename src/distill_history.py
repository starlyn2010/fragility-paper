#!/usr/bin/env python3
"""distill_history.py — Is the failure capacity or history coverage?

The board-level distillation sees 2,423 canonical histories and still fails on
the other 177,938. This script trains the same recurrent architecture on the
full history space with exact oracle targets, so capacity and data coverage are
separated:

  * `--coverage canonical` reproduces the 2,423-history control;
  * `--coverage all` trains on all 180,361 histories;
  * `--coverage N` trains on a random nested subset of N histories.

If the full-history model reaches zero regret on every history, capacity was
never the binding constraint: the earlier failure was coverage induced by the
recurrent state. If it cannot, the result is reported as a capacity or
optimization limit.

Uso:
  python src/distill_history.py --coverage all --epochs 20 \
      --init runs/distill_C_cov2423_s0/ablation_C.pth \
      --out data/distill_history_C_full.json
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
from history_probe import enumerate_histories  # noqa: E402
from oracle import Oracle  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def prepare(histories, oracle, coverage: str):
    if coverage == "all":
        selected = histories
    elif coverage == "canonical":
        seen, selected = set(), []
        for h in histories:
            board = h["history"][-1]
            if board not in seen:
                seen.add(board)
                selected.append(h)
    else:
        n = int(coverage)
        perm = np.random.default_rng(7).permutation(len(histories))
        selected = [histories[i] for i in perm[:min(n, len(histories))]]

    cache: dict[tuple, dict] = {}
    items = []
    for h in selected:
        board = h["history"][-1]
        info = cache.get(board)
        if info is None:
            info = oracle.lookup(board)
            vals = [0.0] * 9
            for a, v in info["values"].items():
                vals[a] = float(v)
            info = {"optimal": info["optimal"], "best": info["best"], "values": vals}
            cache[board] = info
        items.append({
            "history": h["history"],
            "optimal": info["optimal"],
            "best": info["best"],
            "values": info["values"],
            "tie": len(info["optimal"]) >= 2,
        })
    return items


def group_by_length(items):
    groups: dict[int, dict] = {}
    for it in items:
        groups.setdefault(len(it["history"]), []).append(it)
    packed = {}
    for length, group in groups.items():
        hist = torch.tensor([it["history"] for it in group], dtype=torch.float32)
        target = torch.zeros(len(group), 9)
        best = torch.tensor([it["best"] for it in group], dtype=torch.float32)
        values = torch.tensor([it["values"] for it in group], dtype=torch.float32)
        tie = torch.tensor([it["tie"] for it in group], dtype=torch.bool)
        for i, it in enumerate(group):
            target[i, it["optimal"]] = 1.0 / len(it["optimal"])
        packed[length] = {
            "hist": hist, "target": target, "best": best,
            "values": values, "tie": tie,
        }
    return packed


def forward_sequence(agent, hist):
    hx = torch.zeros(hist.shape[0], agent.hidden_dim)
    logits = value = None
    for t in range(hist.shape[1]):
        logits, value, hx = agent.forward_step(hist[:, t], hx)
    return logits, value


def train(agent, groups, epochs, batch_size, lr, seed, max_minutes, eval_groups):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 1000)
    t0 = time.time()
    best_metric = None
    history = []

    for epoch in range(epochs):
        agent.train()
        total, n = 0.0, 0
        for length in sorted(groups, reverse=True):
            g = groups[length]
            order = rng.permutation(g["hist"].shape[0])
            for start in range(0, len(order), batch_size):
                idx = torch.from_numpy(order[start:start + batch_size])
                hist = g["hist"][idx]
                target = g["target"][idx]
                best = g["best"][idx]
                logits, value = forward_sequence(agent, hist)
                legal = hist[:, -1].eq(0.0)
                logp = torch.log_softmax(logits.masked_fill(~legal, -1e9), dim=-1)
                pi_loss = -(target * logp).sum(dim=-1).mean()
                v_loss = torch.nn.functional.mse_loss(value, best)
                loss = pi_loss + 0.25 * v_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()
                total += float(loss.detach()) * len(idx)
                n += len(idx)
        agent.eval()
        metrics = evaluate(agent, eval_groups)
        history.append({"epoch": epoch + 1, **metrics})
        print(f"  epoch {epoch + 1:3d} loss {total / max(n, 1):.4f} "
              f"regret {metrics['mean_regret']:.5f} "
              f"opt {metrics['optimal_action_rate']:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)

        key = (metrics["mean_regret"], -metrics["optimal_action_rate"])
        if best_metric is None or key < best_metric[0]:
            best_metric = (key, epoch + 1, metrics)
        if metrics["mean_regret"] < 1e-6 and metrics["optimal_action_rate"] > 1 - 1e-9:
            print("  converged: every history optimal", flush=True)
            break
        if (time.time() - t0) / 60.0 > max_minutes:
            print(f"  time limit {max_minutes} min reached", flush=True)
            break

    return history, best_metric


@torch.no_grad()
def evaluate(agent, groups, batch_size=16384):
    regrets, optimals, ties = [], [], []
    agent.eval()
    for length, g in groups.items():
        n = g["hist"].shape[0]
        chosen_all = []
        for start in range(0, n, batch_size):
            hist = g["hist"][start:start + batch_size]
            logits, _ = forward_sequence(agent, hist)
            legal = hist[:, -1].eq(0.0)
            chosen_all.append(logits.masked_fill(~legal, -1e9).argmax(dim=-1))
        chosen = torch.cat(chosen_all)
        values = g["values"].gather(1, chosen.unsqueeze(1)).squeeze(1)
        regret = torch.clamp(g["best"] - values, min=0.0)
        optimal = g["target"].gather(1, chosen.unsqueeze(1)).squeeze(1) > 0
        regrets.append(regret)
        optimals.append(optimal)
        ties.append(g["tie"])
    reg = torch.cat(regrets)
    opt = torch.cat(optimals)
    tie = torch.cat(ties)
    return {
        "n": int(reg.numel()),
        "mean_regret": float(reg.mean()),
        "max_regret": float(reg.max()),
        "optimal_action_rate": float(opt.float().mean()),
        "tie_n": int(tie.sum()),
        "tie_mean_regret": float(reg[tie].mean()) if bool(tie.any()) else None,
        "tie_optimal_action_rate": float(opt[tie].float().mean()) if bool(tie.any()) else None,
        "notie_mean_regret": float(reg[~tie].mean()) if bool((~tie).any()) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C", choices=["B", "C"])
    ap.add_argument("--coverage", default="all",
                    help="all | canonical | an integer number of histories")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--init", default="", help="checkpoint to fine-tune from")
    ap.add_argument("--hidden", type=int, default=0,
                    help="override hidden size (0 = config default)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; evaluate --ckpt on all histories")
    ap.add_argument("--ckpt", default="", help="checkpoint for --eval-only")
    ap.add_argument("--out", default=str(ROOT / "data" / "distill_history_C.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = next(c for c in CONFIGS if c["id"] == args.config)
    histories = enumerate_histories()
    oracle = Oracle()
    train_items = prepare(histories, oracle, args.coverage)
    all_items = prepare(histories, oracle, "all")
    train_groups = group_by_length(train_items)
    eval_groups = group_by_length(all_items)
    print(f"histories={len(histories)} train={len(train_items)} "
          f"groups={ {k: v['hist'].shape[0] for k, v in train_groups.items()} }",
          flush=True)

    label = f"{args.config}_hist{args.coverage}_s{args.seed}"
    hidden = args.hidden or cfg["hidden_dim"]

    if args.eval_only:
        if not args.ckpt:
            raise SystemExit("--eval-only requires --ckpt")
        agent = AblationLiquidAgent(args.config, 9, 9, hidden)
        agent.s12_enabled = cfg["act"]
        agent.s12_max_iters = 3
        agent.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        agent.eval()
        final = evaluate(agent, eval_groups)
        out = {
            "args": vars(args),
            "checkpoint": args.ckpt,
            "params": int(sum(p.numel() for p in agent.parameters())),
            "metrics": final,
        }
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"regret {final['mean_regret']:.6f} opt "
              f"{final['optimal_action_rate']:.6f} "
              f"tie_regret {final['tie_mean_regret']:.6f}", flush=True)
        print(f"Crudo: {args.out}", flush=True)
        return

    ckpt_dir = ROOT / "runs" / f"distill_hist_{label}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ablation_{args.config}.pth"

    agent = AblationLiquidAgent(args.config, 9, 9, hidden)
    agent.s12_enabled = cfg["act"]
    agent.s12_max_iters = 3
    if args.init:
        agent.load_state_dict(torch.load(args.init, map_location="cpu"))
        print(f"init from {args.init}", flush=True)

    t0 = time.time()
    history, best = train(agent, train_groups, args.epochs, args.batch_size,
                          args.lr, args.seed, args.max_minutes, eval_groups)
    elapsed = time.time() - t0
    agent.eval()
    torch.save(agent.state_dict(), str(path))
    final = evaluate(agent, eval_groups)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    out = {
        "args": vars(args),
        "n_histories": len(histories),
        "n_boards": len({h["history"][-1] for h in histories}),
        "train": final,
        "checkpoint": str(path),
        "sha256_16": sha,
        "epochs_run": len(history),
        "elapsed_s": elapsed,
        "best_epoch": best[1] if best else None,
        "best": best[2] if best else None,
        "history": history,
    }
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nfinal regret {final['mean_regret']:.6f} "
          f"opt {final['optimal_action_rate']:.6f} "
          f"tie_regret {final['tie_mean_regret']:.6f} "
          f"max {final['max_regret']:.1f}  [{elapsed:.0f}s]", flush=True)
    print(f"Crudo: {args.out}", flush=True)


if __name__ == "__main__":
    main()
