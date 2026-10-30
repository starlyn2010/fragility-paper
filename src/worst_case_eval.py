#!/usr/bin/env python3
"""worst_case_eval.py — Exact certification against every optimal teacher.

The fixed mentor and the random twins are two points inside the set of all
optimal teachers. This script computes the two endpoints exactly:

  * worst case: at every teacher turn, choose an optimal move that minimizes the
    learner's outcome (the teacher may adapt its tie-breaks to the history);
  * best case: the most favorable optimal teacher;
  * exact random-twin draw rate: the average over uniform tie-breaking, which is
    the expectation the 30-twin ensembles estimate by sampling.

The learner may be recurrent (state = ordered history) or memoryless (state =
board). Teacher moves are restricted to the exact depth-9 optimal set, so every
opponent in the search is exactly as strong as the fixed mentor.

Uso:
  python src/worst_case_eval.py --preset paper --out data/worst_case.json
  python src/worst_case_eval.py --runs "C42=C:runs/fixed_C_s42" \
      --mlp "MLP0=runs/memoryless_s0/mlp.pt" --out data/worst_case.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import random
import sys
from datetime import datetime, timezone
from functools import lru_cache

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_ttt import CONFIGS, AblationLiquidAgent  # noqa: E402
from history_probe import enumerate_histories, history_actions  # noqa: E402
from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv  # noqa: E402
from memoryless import MLPPolicy  # noqa: E402
from oracle import Oracle, env_from_board  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMPTY = (0,) * 9

PAPER_RUNS = [
    ("A-v1", "A", "artifacts/v1_reentreno"),
    ("B-v1", "B", "artifacts/v1_reentreno"),
    ("C-v1", "C", "artifacts/v1_reentreno"),
    ("B-42", "B", "runs/fixed_B_s42"),
    ("B-43", "B", "runs/fixed_B_s43"),
    ("B-44", "B", "runs/fixed_B_s44"),
    ("C-42", "C", "runs/fixed_C_s42"),
    ("C-43", "C", "runs/fixed_C_s43"),
    ("C-44", "C", "runs/fixed_C_s44"),
]

TBR_RUNS = [
    ("B-TBR-42", "B", "runs/tbr_B_s42"),
    ("B-TBR-43", "B", "runs/tbr_B_s43"),
    ("B-TBR-44", "B", "runs/tbr_B_s44"),
    ("C-TBR-42", "C", "runs/tbr_C_s42"),
    ("C-TBR-43", "C", "runs/tbr_C_s43"),
    ("C-TBR-44", "C", "runs/tbr_C_s44"),
]

EXTRA_RUNS = [
    ("B-episode-42", "B", "runs/episode_B_s42"),
    ("C-episode-42", "C", "runs/episode_C_s42"),
    ("C-episode-43", "C", "runs/episode_C_s43"),
    ("C-mixed-42", "C", "runs/mixed_C_s42"),
    ("C-noise-42", "C", "runs/noise_C_s42"),
    ("C-distill-full-s0", "C", "runs/distill_C_cov2423_s0"),
]

PRESETS = {
    "paper": PAPER_RUNS,
    "tbr": TBR_RUNS,
    "extra": EXTRA_RUNS,
    "all": PAPER_RUNS + TBR_RUNS + EXTRA_RUNS,
}


def resolve_checkpoint(config: str, spec: str) -> pathlib.Path:
    p = pathlib.Path(spec)
    if p.is_dir():
        p = p / f"ablation_{config}.pth"
    return p


def board_state(board: tuple[int, ...]):
    env = env_from_board(board)
    return env.done, env.resultado(), sum(1 for v in board if v != 0)


def apply_move(board: tuple[int, ...], action: int, player: int) -> tuple[int, ...]:
    env = env_from_board(board)
    env.step(action, player=player)
    return tuple(int(v) for v in env._obs())


def make_teacher():
    mentor = StochasticMinimax(mentor_player=-1, depth=9, noise_prob=0.0,
                              tie_break="first")

    @lru_cache(maxsize=None)
    def optimal(board: tuple[int, ...]) -> tuple[int, ...]:
        env = env_from_board(board)
        _, actions = mentor.optimal_actions(env, depth=9)
        return tuple(int(a) for a in actions)

    return optimal


def analyze(policy_action, teacher_optimal, oracle: Oracle, collect_witness=True):
    """Return worst/best/exact-random outcome over the set of optimal teachers."""

    @lru_cache(maxsize=None)
    def rec(board: tuple[int, ...], hist: tuple):
        done, result, pieces = board_state(board)
        if done:
            return result, result, (1.0 if result == 0 else 0.0), (1.0 if result == -1 else 0.0)

        if pieces % 2 == 0:  # learner
            action = policy_action(board, hist)
            nxt = apply_move(board, action, TicTacToeEnv.AGENTE)
            return rec(nxt, hist + (board,))

        actions = teacher_optimal(board)
        worst, best, p_draw, p_loss = 2, -2, 0.0, 0.0
        for action in actions:
            nxt = apply_move(board, action, TicTacToeEnv.HUMANO)
            w, b, pd, pl = rec(nxt, hist)
            worst = min(worst, w)
            best = max(best, b)
            p_draw += pd
            p_loss += pl
        n = len(actions)
        return worst, best, p_draw / n, p_loss / n

    worst, best, p_draw, p_loss = rec(EMPTY, ())

    witness = None
    if collect_witness:
        board, hist, steps = EMPTY, (), []
        while True:
            done, result, pieces = board_state(board)
            if done:
                witness = {"result": int(result), "steps": steps}
                break
            if pieces % 2 == 0:
                action = policy_action(board, hist)
                info = oracle.lookup(board)
                steps.append({
                    "player": "agent",
                    "board": list(board),
                    "action": int(action),
                    "regret": float(max(0.0, info["best"] - info["values"][action])),
                    "optimal": bool(action in info["optimal"]),
                })
                nxt = apply_move(board, action, TicTacToeEnv.AGENTE)
                hist = hist + (board,)
            else:
                actions = teacher_optimal(board)
                scored = []
                for action in actions:
                    nxt = apply_move(board, action, TicTacToeEnv.HUMANO)
                    w, _, _, _ = rec(nxt, hist)
                    scored.append((w, action))
                _, action = min(scored, key=lambda x: x[0])
                steps.append({
                    "player": "teacher",
                    "board": list(board),
                    "action": int(action),
                })
                nxt = apply_move(board, action, TicTacToeEnv.HUMANO)
            board = nxt

    return {
        "worst_result": int(worst),
        "best_result": int(best),
        "worst_draw_rate": 1.0 if worst == 0 else 0.0,
        "best_draw_rate": 1.0 if best == 0 else 0.0,
        "random_draw_rate_exact": float(p_draw),
        "random_loss_rate_exact": float(p_loss),
        "n_states": int(rec.cache_info().currsize),
        "witness": witness,
    }


def monte_carlo(policy_action, teacher_optimal, games: int, seed: int) -> float:
    """Independent check: sample uniform optimal tie-breaks and count draws."""
    rng = random.Random(seed)
    draws = 0
    for _ in range(games):
        board, hist = EMPTY, ()
        while True:
            done, result, pieces = board_state(board)
            if done:
                draws += int(result == 0)
                break
            if pieces % 2 == 0:
                old = board
                action = policy_action(board, hist)
                board = apply_move(board, action, TicTacToeEnv.AGENTE)
                hist = hist + (old,)
            else:
                action = rng.choice(teacher_optimal(board))
                board = apply_move(board, action, TicTacToeEnv.HUMANO)
    return draws / games


def load_recurrent(config: str, path: pathlib.Path, hidden: int | None = None):
    cfg = next(c for c in CONFIGS if c["id"] == config)
    agent = AblationLiquidAgent(config, 9, 9, hidden or cfg["hidden_dim"])
    agent.s12_enabled = cfg["act"]
    agent.s12_max_iters = 3
    agent.load_state_dict(torch.load(str(path), map_location="cpu"))
    agent.eval()
    return agent


def recurrent_policy(agent, histories):
    table = history_actions(agent, histories)

    def policy(board: tuple[int, ...], hist: tuple) -> int:
        key = hist + (board,)
        if key not in table:
            raise KeyError(f"history not enumerated: {key}")
        return table[key]

    return policy


def memoryless_policy(mlp: MLPPolicy, histories):
    boards = sorted({h["history"][-1] for h in histories})
    x = torch.tensor(boards, dtype=torch.float32)
    legal = torch.tensor([[v == 0 for v in b] for b in boards], dtype=torch.bool)
    with torch.no_grad():
        logits = mlp.net(x).masked_fill(~legal, -1e9)
        actions = logits.argmax(dim=1).tolist()
    table = {b: int(a) for b, a in zip(boards, actions)}

    def policy(board: tuple[int, ...], hist: tuple) -> int:
        return table[board]

    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="label=config:path;...")
    ap.add_argument("--mlp", default="", help="label=path;...")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "worst_case.json"))
    ap.add_argument("--no-witness", action="store_true")
    ap.add_argument("--mc-games", type=int, default=0,
                    help="independent uniform-tie-break games per run (0 = off)")
    ap.add_argument("--mc-seed", type=int, default=2026)
    args = ap.parse_args()

    torch.set_num_threads(1)
    histories = enumerate_histories()
    oracle = Oracle()
    teacher_optimal = make_teacher()
    print(f"histories={len(histories)} boards={len({h['board'] for h in histories})}",
          flush=True)

    runs = [(label, config, path, None) for label, config, path in PRESETS[args.preset]] if args.preset else []
    for item in args.runs.split(";"):
        item = item.strip()
        if not item:
            continue
        label, spec = item.split("=", 1)
        config, path = spec.split(":", 1)
        config = config.strip()
        hidden = None
        if "@" in config:
            config, h = config.split("@", 1)
            hidden = int(h)
        runs.append((label.strip(), config, path.strip(), hidden))

    out = {
        "args": vars(args),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "when_utc": datetime.now(timezone.utc).isoformat(),
        },
        "n_histories": len(histories),
        "n_boards": len({h["board"] for h in histories}),
        "results": {},
    }

    for label, config, spec, hidden in runs:
        path = resolve_checkpoint(config, spec)
        if not path.exists():
            print(f"[skip] {label}: missing {path}", flush=True)
            continue
        agent = load_recurrent(config, path, hidden)
        policy = recurrent_policy(agent, histories)
        res = analyze(policy, teacher_optimal, oracle, not args.no_witness)
        res.update({
            "kind": "recurrent",
            "config": config,
            "checkpoint": str(path),
            "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        })
        if args.mc_games > 0:
            res["mc_draw_rate"] = monte_carlo(policy, teacher_optimal,
                                              args.mc_games, args.mc_seed)
            res["mc_games"] = args.mc_games
            res["mc_seed"] = args.mc_seed
        out["results"][label] = res
        print(f"{label:>16} worst {res['worst_draw_rate']:.3f} "
              f"best {res['best_draw_rate']:.3f} "
              f"random {res['random_draw_rate_exact']:.4f} "
              f"states {res['n_states']}", flush=True)
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for item in args.mlp.split(";"):
        item = item.strip()
        if not item:
            continue
        label, spec = item.split("=", 1)
        path = pathlib.Path(spec.strip())
        if not path.exists():
            print(f"[skip] {label}: missing {path}", flush=True)
            continue
        mlp = MLPPolicy.load(path)
        policy = memoryless_policy(mlp, histories)
        res = analyze(policy, teacher_optimal, oracle, not args.no_witness)
        res.update({
            "kind": "memoryless",
            "config": "MLP",
            "checkpoint": str(path),
            "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        })
        if args.mc_games > 0:
            res["mc_draw_rate"] = monte_carlo(policy, teacher_optimal,
                                              args.mc_games, args.mc_seed)
            res["mc_games"] = args.mc_games
            res["mc_seed"] = args.mc_seed
        out["results"][label] = res
        print(f"{label:>16} worst {res['worst_draw_rate']:.3f} "
              f"best {res['best_draw_rate']:.3f} "
              f"random {res['random_draw_rate_exact']:.4f} "
              f"states {res['n_states']}", flush=True)
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nCrudo: {args.out}")


if __name__ == "__main__":
    main()
