#!/usr/bin/env python3
"""twin_ensemble.py — How many equally strong teachers are needed to reject a policy?

Evaluates one checkpoint against many held-out tie-break seeds. Each twin uses
the same depth-6 minimax value function and differs only in its random tie-break.
No regret oracle: this is a pure game-outcome scaling test.

Uso:
  python src/twin_ensemble.py --config C --ckpt runs/tbr_C_s43 --seeds 30 --games 100
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
from liquid_rl_trainer import RandomAgent, StochasticMinimax, TicTacToeEnv  # noqa: E402
from tbr_eval import load_agent  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def play(agent, env, opponent, n_games: int) -> dict:
    wins = draws = 0
    for _ in range(n_games):
        hx = torch.zeros(1, agent.hidden_dim)
        obs = env.reset()
        done = False
        turn = 0
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            legal_mask = torch.zeros(agent.action_dim, dtype=torch.bool)
            for a in legal:
                legal_mask[a] = True
            if turn % 2 == 0:
                with torch.no_grad():
                    action, hx = agent.get_action(obs, hx, legal_mask=legal_mask, deterministic=True)
                obs, _, done, _ = env.step(action, player=env.AGENTE)
            else:
                if isinstance(opponent, RandomAgent):
                    opp_action = opponent.get_action(legal)
                else:
                    opp_action = opponent.get_action(env, legal)
                obs, _, done, _ = env.step(opp_action, player=env.HUMANO)
            turn += 1
        r = env.resultado()
        wins += int(r == 1)
        draws += int(r == 0)
    return {"games": n_games, "wins": wins, "draws": draws,
            "losses": n_games - wins - draws,
            "wr": wins / n_games, "dr": draws / n_games,
            "lr": (n_games - wins - draws) / n_games}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C", choices=["B", "C"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", default="", help="optional label override")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--tie-break", default="random", choices=["random", "episode"],
                    help="random: independent tie per decision; episode: one coherent rule per game")
    ap.add_argument("--out", default=str(ROOT / "data" / "twin_ensemble.json"))
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
    label = args.labels or path.parent.name

    out = {"args": vars(args), "label": label, "results": {}}
    fixed = StochasticMinimax(depth=6, noise_prob=0.0, tie_break="first")
    out["results"]["m6_fixed"] = play(agent, env, fixed, args.games)
    for i in range(args.seeds):
        seed = 1000 + i
        twin = StochasticMinimax(depth=6, noise_prob=0.0, tie_break=args.tie_break, seed=seed)
        res = play(agent, env, twin, args.games)
        out["results"][f"twin_{seed}"] = res
        print(f"  {args.tie_break} {seed:>4}: D {res['dr']*100:5.1f}%  W {res['wr']*100:5.1f}%", flush=True)

    dr = np.array([v["dr"] for k, v in out["results"].items() if k.startswith("twin_")])
    out["summary"] = {
        "n_twins": len(dr), "games_per_twin": args.games,
        "dr_mean": float(dr.mean()), "dr_min": float(dr.min()), "dr_max": float(dr.max()),
        "n_at_100": int((dr >= 1.0).sum()), "n_below_90": int((dr < 0.9).sum()),
    }
    print(f"\n{label}: fixed DR {out['results']['m6_fixed']['dr']*100:.1f}% | "
          f"twins mean {dr.mean()*100:.1f}% min {dr.min()*100:.1f}% max {dr.max()*100:.1f}% "
          f"| at100 {int((dr >= 1.0).sum())}/{len(dr)} | below90 {int((dr < 0.9).sum())}/{len(dr)}")
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Crudo: {p}")


if __name__ == "__main__":
    main()
