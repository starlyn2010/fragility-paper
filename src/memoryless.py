#!/usr/bin/env python3
"""memoryless.py — A Markov policy trained on the full board space.

The recurrent agents can fail against a twin even when they are optimal on every
reachable *board*, because their hidden state depends on move order. Tic-Tac-Toe
is Markov, so a memoryless policy removes that dependence. If the twin gap is a
history-coverage failure, this policy should be robust after seeing every board.

Uso:
  python src/memoryless.py --epochs 1500 --seeds 0,1 --out data/memoryless.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle import Oracle  # noqa: E402
from space_probe import enumerate_agent_states  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


class MLP(nn.Module):
    def __init__(self, obs_dim=9, action_dim=9, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class MLPPolicy:
    """Minimal interface compatible with tbr_eval.play."""

    def __init__(self, net: MLP):
        self.net = net
        self.hidden_dim = 1
        self.action_dim = 9

    @torch.no_grad()
    def get_action(self, obs, hx, legal_mask=None, deterministic=True):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.net(x)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask.unsqueeze(0), -1e9)
        return int(logits.argmax(dim=1).item()), hx

    def save(self, path):
        torch.save(self.net.state_dict(), str(path))

    @classmethod
    def load(cls, path):
        net = MLP()
        net.load_state_dict(torch.load(str(path), map_location="cpu"))
        net.eval()
        return cls(net)


def build_data():
    states = enumerate_agent_states()
    oracle = Oracle()
    boards, optimals = [], []
    for s in states:
        info = oracle.lookup(s["board"])
        boards.append(s["board"])
        optimals.append(info["optimal"])
    x = torch.tensor(np.asarray(boards), dtype=torch.float32)
    target = torch.zeros(len(optimals), 9)
    for i, opt in enumerate(optimals):
        target[i, opt] = 1.0 / len(opt)
    return x, optimals, target


def train(x, target, epochs, lr, seed):
    torch.manual_seed(seed)
    net = MLP()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = x.size(0)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for start in range(0, n, 256):
            idx = torch.from_numpy(perm[start:start + 256].astype(np.int64))
            logp = torch.log_softmax(net(x[idx]), dim=-1)
            loss = -(target[idx] * logp).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        if (epoch + 1) % 250 == 0:
            print(f"  epoch {epoch+1:4d} loss {total / n:.4f}", flush=True)
    net.eval()
    return net


@torch.no_grad()
def board_metrics(net, x, optimals):
    logits = net(x)
    chosen = logits.argmax(dim=1).numpy()
    ok = np.array([chosen[i] in optimals[i] for i in range(len(optimals))])
    return {
        "n": int(len(ok)),
        "optimal_action_rate": float(ok.mean()),
        "n_suboptimal": int((~ok).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(ROOT / "data" / "memoryless.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    x, optimals, target = build_data()
    print(f"boards={x.size(0)}", flush=True)
    out = {"n_boards": int(x.size(0)), "results": {}}
    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"\n=== MLP seed {seed} ===", flush=True)
        net = train(x, target, args.epochs, args.lr, seed)
        m = board_metrics(net, x, optimals)
        ckpt_dir = ROOT / "runs" / f"memoryless_s{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "mlp.pt"
        MLPPolicy(net).save(path)
        out["results"][f"MLP_s{seed}"] = {"checkpoint": str(path), **m}
        print(f"  {m}", flush=True)
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
