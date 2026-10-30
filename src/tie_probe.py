#!/usr/bin/env python3
"""tie_probe.py — Mide la geometria de la politica sobre clases de jugadas optimas.

En estados donde el minimax tiene k>=2 jugadas igual de optimas, un docente con
desempate fijo expone solo una de ellas; TBR expone las k uniformemente. Este
sonda mide, sobre un conjunto FIJO de historias aleatorias (identico para todos
los checkpoints), cuanta masa de politica queda en la clase optima y como se
reparte dentro de ella.

Salida: data/tie_probe.json con medias por checkpoint, IC bootstrap agrupado por
historia unica (n_eff), y sha256 del checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import sys
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_ttt import CONFIGS, AblationLiquidAgent  # noqa: E402
from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def mirrored(env: TicTacToeEnv) -> TicTacToeEnv:
    m = env.clone()
    flip = {0: 0, 1: 2, 2: 1}
    m.grid = [[flip[v] for v in row] for row in env.grid]
    return m


def agent_turn_histories(n_games: int, seed: int = 0) -> list[tuple[tuple[int, ...], ...]]:
    """Historias de observaciones en turnos del agente, con ambos jugadores al azar."""
    rng = random.Random(seed)
    env = TicTacToeEnv()
    histories: list[tuple[tuple[int, ...], ...]] = []
    for _ in range(n_games):
        env.reset()
        obs_agent: list[tuple[int, ...]] = []
        turn = 0
        while not env.done:
            legal = env.legal_actions()
            if not legal:
                break
            if turn % 2 == 0:
                obs_agent.append(tuple(int(v) for v in env._obs()))
            a = rng.choice(legal)
            env.step(a, player=env.AGENTE if turn % 2 == 0 else env.HUMANO)
            turn += 1
        histories.append(tuple(obs_agent))
    return histories


class OptimalOracle:
    """Valor exacto y conjunto de acciones optimas desde la perspectiva del agente."""

    def __init__(self) -> None:
        self.mentor = StochasticMinimax(mentor_player=-1, depth=9, noise_prob=0.0, tie_break="first")
        self.cache: dict[tuple[int, ...], tuple[float, tuple[int, ...]]] = {}

    def lookup(self, board: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
        hit = self.cache.get(board)
        if hit is not None:
            return hit
        env = TicTacToeEnv()
        env.grid = [list(board[0:3]), list(board[3:6]), list(board[6:9])]
        value, actions = self.mentor.optimal_actions(mirrored(env), depth=9)
        out = (float(value), tuple(actions))
        self.cache[board] = out
        return out


def policy_probs(agent, obs: tuple[int, ...], hx: torch.Tensor, legal: list[int]):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        if getattr(agent, "s12_enabled", False):
            logits, _, hx_new, _, _ = agent.forward_adaptive(obs_t, hx)
        else:
            logits, _, hx_new = agent.forward_step(obs_t, hx)
        mask = torch.full((agent.action_dim,), False)
        for a in legal:
            mask[a] = True
        logits = logits.masked_fill(~mask.unsqueeze(0), -float("inf"))
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probs, hx_new


def bootstrap_mean(values: list[float], groups: list[int], n_boot: int = 2000, seed: int = 0):
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    by_group: dict[int, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    keys = list(by_group)
    arr = np.asarray(values, dtype=float)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = [i for g in pick for i in by_group[g]]
        means.append(float(arr[idx].mean()))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="'label=ckpt_dir;...'")
    ap.add_argument("--config", default="B", choices=["A", "B", "C", "D"])
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "tie_probe.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = next(c for c in CONFIGS if c["id"] == args.config)
    histories = agent_turn_histories(args.games, args.seed)
    oracle = OptimalOracle()

    decisions = []
    for h_idx, hist in enumerate(histories):
        for step, board in enumerate(hist):
            legal = [i for i, v in enumerate(board) if v == 0]
            if not legal:
                continue
            value, optimal = oracle.lookup(board)
            decisions.append({
                "history": h_idx,
                "step": step,
                "board": board,
                "legal": legal,
                "optimal": optimal,
                "tie": len(optimal) >= 2,
            })
    decision_index = {(d["history"], d["step"]): d for d in decisions}
    n_unique_hist = len({d["history"] for d in decisions if d["tie"]})
    print(f"decisiones={len(decisions)} empates={sum(d['tie'] for d in decisions)} "
          f"historias_unicas_con_empate={n_unique_hist}", flush=True)

    runs = []
    for item in args.runs.split(";"):
        item = item.strip()
        if not item:
            continue
        label, path = item.split("=", 1)
        runs.append((label.strip(), pathlib.Path(path.strip())))

    out = {
        "args": vars(args),
        "config": cfg,
        "environment": {
            "when_utc": datetime.now(timezone.utc).isoformat(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "n_decisions": len(decisions),
        "n_tie_decisions": sum(d["tie"] for d in decisions),
        "n_unique_tie_histories": n_unique_hist,
        "results": {},
    }

    for label, ckpt_dir in runs:
        path = ckpt_dir / f"ablation_{cfg['id']}.pth"
        if not path.exists():
            print(f"[skip] {label}: falta {path}")
            continue
        agent = AblationLiquidAgent(cfg["id"], 9, 9, cfg["hidden_dim"])
        agent.s12_enabled = cfg["act"]
        agent.s12_max_iters = 3
        agent.load_state_dict(torch.load(str(path), map_location="cpu"))
        agent.eval()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]

        rows = []
        hx = torch.zeros(1, agent.hidden_dim)
        hist_cache: dict[int, list[tuple[dict, np.ndarray, list[int]]]] = {}
        for h_idx, hist in enumerate(histories):
            hx = torch.zeros(1, agent.hidden_dim)
            cache = []
            for step, board in enumerate(hist):
                legal = [i for i, v in enumerate(board) if v == 0]
                probs, hx = policy_probs(agent, board, hx, legal)
                d = decision_index.get((h_idx, step))
                if d is not None:
                    cache.append((d, probs, legal))
            hist_cache[h_idx] = cache

        for h_idx, cache in hist_cache.items():
            for d, probs, legal in cache:
                if not d["tie"]:
                    continue
                optimal = d["optimal"]
                p_opt = [float(probs[a]) for a in optimal]
                mass = float(sum(p_opt))
                k = len(optimal)
                ent = -sum(p * math.log(p + 1e-12) for p in p_opt)
                ent_norm = ent / math.log(k)
                rows.append({
                    "history": h_idx,
                    "mass_optimal": mass,
                    "entropy_within_optimal": ent_norm,
                    "peak_within_optimal": max(p_opt),
                    "mass_first_optimal": float(probs[optimal[0]]),
                    "argmax_is_optimal": int(int(np.argmax(probs)) in optimal),
                    "k_optimal": k,
                    "n_legal": len(legal),
                })

        def agg(key):
            vals = [r[key] for r in rows]
            return float(np.mean(vals)) if vals else float("nan")

        groups = [r["history"] for r in rows]
        ci_mass = bootstrap_mean([r["mass_optimal"] for r in rows], groups, seed=args.seed + 1)
        ci_ent = bootstrap_mean([r["entropy_within_optimal"] for r in rows], groups, seed=args.seed + 2)

        summary = {
            "checkpoint": str(path),
            "sha256_16": sha,
            "n_tie_rows": len(rows),
            "mean_mass_optimal": agg("mass_optimal"),
            "ci_mass_optimal": ci_mass,
            "mean_entropy_within_optimal": agg("entropy_within_optimal"),
            "ci_entropy_within_optimal": ci_ent,
            "mean_peak_within_optimal": agg("peak_within_optimal"),
            "mean_mass_first_optimal": agg("mass_first_optimal"),
            "argmax_is_optimal_rate": agg("argmax_is_optimal"),
            "mean_k_optimal": agg("k_optimal"),
        }
        out["results"][label] = summary
        print(f"{label:>14} sha:{sha} | mass_opt {summary['mean_mass_optimal']:.3f} "
              f"[{ci_mass[0]:.3f},{ci_mass[1]:.3f}] | H_opt {summary['mean_entropy_within_optimal']:.3f} "
              f"[{ci_ent[0]:.3f},{ci_ent[1]:.3f}] | mass_first {summary['mean_mass_first_optimal']:.3f} "
              f"| argmax_opt {summary['argmax_is_optimal_rate']:.3f}", flush=True)

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nCrudo: {p}")


if __name__ == "__main__":
    main()
