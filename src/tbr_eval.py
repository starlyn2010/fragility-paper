#!/usr/bin/env python3
"""
tbr_eval.py — Evaluación de la intervención Tie-Break Randomization (TBR).

Mide, para cada checkpoint:
  - M6 fijo (el "examen" del paper, desempate historico)
  - K mentores gemelos (misma fuerza, desempate aleatorio con semillas no vistas)
  - Random uniforme real (RandomAgent; el baseline historico usaba
    StochasticMinimax(depth=0), que NO es aleatorio)
  - BoundedRationalMinimax (rival debil pero sensato; mide "agencia")

Métrica primaria: twin_gap = DR(media de gemelos) - DR(M6 fijo).
Un agente que memorizó al maestro tiene twin_gap muy negativo.

Uso:
  python src/tbr_eval.py --runs "b_s42=ablation_logs;tbr_s42=runs/tbr_B_s42" \
      --config B --games 200 --out data/tbr_eval.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_ttt import CONFIGS, AblationLiquidAgent  # noqa: E402
from human_like_agent import BoundedRationalMinimax  # noqa: E402
from liquid_rl_trainer import RandomAgent, StochasticMinimax, TicTacToeEnv  # noqa: E402
from oracle import Oracle  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_agent(cfg: dict, path: pathlib.Path) -> AblationLiquidAgent:
    agent = AblationLiquidAgent(cfg["id"], 9, 9, cfg["hidden_dim"])
    agent.s12_enabled = cfg["act"]
    agent.s12_max_iters = 3
    agent.load_state_dict(torch.load(str(path), map_location="cpu"))
    agent.eval()
    return agent


def play(agent, env, opponent, n_games: int, oracle: Oracle) -> dict:
    wins = draws = 0
    is_shadow = type(opponent).__name__ == "ShadowAgent"
    regret_sum = 0.0
    n_decisions = 0
    n_optimal = 0
    tie_regret = 0.0
    tie_decisions = 0
    notie_regret = 0.0
    notie_decisions = 0
    per_game = {"regret": [], "decisions": [], "optimal": [], "outcome": []}
    for _ in range(n_games):
        hx = torch.zeros(1, agent.hidden_dim)
        obs = env.reset()
        done = False
        turn = 0
        g_regret = g_decisions = g_optimal = 0
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            legal_mask = torch.zeros(agent.action_dim, dtype=torch.bool)
            for a in legal:
                legal_mask[a] = True
            if turn % 2 == 0:
                board = tuple(int(v) for v in env._obs())
                info = oracle.lookup(board)
                with torch.no_grad():
                    action, hx = agent.get_action(obs, hx, legal_mask=legal_mask, deterministic=True)
                dec_regret = max(0.0, info["best"] - info["values"][action])
                g_regret += dec_regret
                g_optimal += int(action in info["optimal"])
                g_decisions += 1
                if len(info["optimal"]) > 1:
                    tie_regret += dec_regret
                    tie_decisions += 1
                else:
                    notie_regret += dec_regret
                    notie_decisions += 1
                obs, _, done, _ = env.step(action, player=env.AGENTE)
            else:
                if is_shadow:
                    opp_action, hx = opponent.get_action(obs, hx, legal_mask=legal_mask)
                elif isinstance(opponent, RandomAgent):
                    opp_action = opponent.get_action(legal)
                else:
                    opp_action = opponent.get_action(env, legal)
                obs, _, done, _ = env.step(opp_action, player=env.HUMANO)
            turn += 1
            if done:
                break
        r = env.resultado()
        if r == 1:
            wins += 1
        elif r == 0:
            draws += 1
        regret_sum += g_regret
        n_decisions += g_decisions
        n_optimal += g_optimal
        per_game["regret"].append(g_regret)
        per_game["decisions"].append(g_decisions)
        per_game["optimal"].append(g_optimal)
        per_game["outcome"].append(r)
    losses = n_games - wins - draws
    return {
        "games": n_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "wr": wins / n_games,
        "dr": draws / n_games,
        "lr": losses / n_games,
        "wr_ci": wilson(wins, n_games),
        "dr_ci": wilson(draws, n_games),
        "lr_ci": wilson(losses, n_games),
        "n_decisions": n_decisions,
        "mean_regret": regret_sum / n_decisions if n_decisions else float("nan"),
        "optimal_action_rate": n_optimal / n_decisions if n_decisions else float("nan"),
        "tie_decisions": tie_decisions,
        "tie_decision_share": tie_decisions / n_decisions if n_decisions else float("nan"),
        "regret_tie": tie_regret / tie_decisions if tie_decisions else float("nan"),
        "regret_notie": notie_regret / notie_decisions if notie_decisions else float("nan"),
        "per_game": per_game,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="lista 'label=ckpt_dir' separada por ';'")
    ap.add_argument("--config", default="B", choices=["A", "B", "C", "D"])
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--twin-seeds", default="1234,5678,9012")
    ap.add_argument("--seed", type=int, default=20261027,
                    help="semilla global para RandomAgent y BoundedRationalMinimax")
    ap.add_argument("--out", default=str(ROOT / "data" / "tbr_eval.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    oracle = Oracle()
    cfg = next(c for c in CONFIGS if c["id"] == args.config)
    env = TicTacToeEnv()
    twin_seeds = [int(s) for s in args.twin_seeds.split(",") if s.strip()]

    opponents: dict[str, object] = {
        "m6_fixed": StochasticMinimax(depth=6, noise_prob=0.0, tie_break="first"),
        "random_uniform": RandomAgent(),
        "weak_minimax": BoundedRationalMinimax(depth_normal=2, depth_tension=4, noise_std=0.5),
    }
    for seed in twin_seeds:
        opponents[f"twin_{seed}"] = StochasticMinimax(
            depth=6, noise_prob=0.0, tie_break="random", seed=seed
        )

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
        for name, opp in opponents.items():
            res = play(agent, env, opp, args.games, oracle)
            row["opponents"][name] = res
            print(f"  {name:>14}: W {res['wr']*100:5.1f}%  D {res['dr']*100:5.1f}%  "
                  f"L {res['lr']*100:5.1f}%  [D 95% CI {res['dr_ci'][0]*100:.1f}-{res['dr_ci'][1]*100:.1f}]"
                  f"  regret {res['mean_regret']:.4f}  opt {res['optimal_action_rate']:.3f}",
                  flush=True)
        twin_drs = [row["opponents"][f"twin_{s}"]["dr"] for s in twin_seeds]
        twin_reg = [row["opponents"][f"twin_{s}"]["mean_regret"] for s in twin_seeds]
        twin_opt = [row["opponents"][f"twin_{s}"]["optimal_action_rate"] for s in twin_seeds]
        row["summary"] = {
            "dr_m6_fixed": row["opponents"]["m6_fixed"]["dr"],
            "dr_twin_mean": float(np.mean(twin_drs)),
            "dr_twin_min": float(np.min(twin_drs)),
            "dr_twin_max": float(np.max(twin_drs)),
            "twin_gap_mean": float(np.mean(twin_drs) - row["opponents"]["m6_fixed"]["dr"]),
            "twin_gap_worst": float(np.min(twin_drs) - row["opponents"]["m6_fixed"]["dr"]),
            "wr_random_uniform": row["opponents"]["random_uniform"]["wr"],
            "wr_weak_minimax": row["opponents"]["weak_minimax"]["wr"],
            "draws_vs_weak": row["opponents"]["weak_minimax"]["dr"],
            "regret_m6_fixed": row["opponents"]["m6_fixed"]["mean_regret"],
            "regret_twin_mean": float(np.mean(twin_reg)),
            "regret_gap_twin": float(np.mean(twin_reg) - row["opponents"]["m6_fixed"]["mean_regret"]),
            "opt_rate_m6_fixed": row["opponents"]["m6_fixed"]["optimal_action_rate"],
            "opt_rate_twin_mean": float(np.mean(twin_opt)),
            "opt_rate_gap_twin": float(np.mean(twin_opt) - row["opponents"]["m6_fixed"]["optimal_action_rate"]),
            "regret_tie_twin_mean": float(np.mean([row["opponents"][f"twin_{s}"]["regret_tie"] for s in twin_seeds])),
            "regret_notie_twin_mean": float(np.mean([row["opponents"][f"twin_{s}"]["regret_notie"] for s in twin_seeds])),
            "tie_share_twin_mean": float(np.mean([row["opponents"][f"twin_{s}"]["tie_decision_share"] for s in twin_seeds])),
            "elapsed_s": round(time.time() - t0, 1),
        }
        out["results"][label] = row
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        s = row["summary"]
        print(f"  -> DR(M6) {s['dr_m6_fixed']*100:.1f}% | DR(gemelos) {s['dr_twin_mean']*100:.1f}% "
              f"| twin_gap {s['twin_gap_mean']*100:+.1f} pts | regret_gap {s['regret_gap_twin']:+.4f} "
              f"| WR(random) {s['wr_random_uniform']*100:.1f}% "
              f"| WR(weak) {s['wr_weak_minimax']*100:.1f}%", flush=True)

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nCrudo: {p}")


if __name__ == "__main__":
    main()
