#!/usr/bin/env python3
"""space_probe.py — Regret exhaustivo de la politica sobre TODO el espacio de estados.

Enumera todos los tableros alcanzables en los que le toca jugar al agente, calcula
el valor minimax exacto de cada accion (perspectiva del agente) y mide el regret de
la accion greedy de cada checkpoint. No hay muestreo: n_eff = numero de tableros.

Tambien reporta el regret separado en estados con empate (varias jugadas optimas)
y sin empate, que es donde TBR deberia actuar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from collections import deque
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_ttt import CONFIGS, AblationLiquidAgent  # noqa: E402
from liquid_rl_trainer import TicTacToeEnv  # noqa: E402
from oracle import Oracle, env_from_board  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def enumerate_agent_states():
    start = ((0,) * 9, 0, ())
    seen = {(start[0], start[1])}
    q = deque([start])
    states = []
    while q:
        board, turn, hist = q.popleft()
        env = env_from_board(board)
        if env.done:
            continue
        legal = env.legal_actions()
        is_agent = turn % 2 == 0
        hist_now = hist + (board,) if is_agent else hist
        if is_agent:
            states.append({"board": board, "history": hist_now, "moves": turn})
        for a in legal:
            nxt = env.clone()
            nxt.step(a, player=env.AGENTE if is_agent else env.HUMANO)
            nb = tuple(int(v) for v in nxt._obs())
            key = (nb, turn + 1)
            if key not in seen:
                seen.add(key)
                q.append((nb, turn + 1, hist_now))
    return states


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="'label=ckpt_dir;...'")
    ap.add_argument("--config", default="B", choices=["A", "B", "C", "D"])
    ap.add_argument("--out", default=str(ROOT / "data" / "space_probe.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = next(c for c in CONFIGS if c["id"] == args.config)
    states = enumerate_agent_states()
    oracle = Oracle()
    for s in states:
        s["oracle"] = oracle.lookup(s["board"])
        s["tie"] = len(s["oracle"]["optimal"]) >= 2
        s["k_optimal"] = len(s["oracle"]["optimal"])
    n_tie = sum(s["tie"] for s in states)
    print(f"estados_agente={len(states)} empates={n_tie} "
          f"k_medio={np.mean([s['k_optimal'] for s in states]):.2f}", flush=True)

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
        "n_states": len(states),
        "n_tie_states": n_tie,
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
        root_probs = None
        for s in states:
            hx = torch.zeros(1, agent.hidden_dim)
            probs = None
            for board in s["history"]:
                legal = [i for i, v in enumerate(board) if v == 0]
                probs, hx = policy_probs(agent, board, hx, legal)
            chosen = int(np.argmax(probs))
            best = s["oracle"]["best"]
            regret = float(max(0.0, best - s["oracle"]["values"][chosen]))
            p_opt = [float(probs[a]) for a in s["oracle"]["optimal"]]
            canon = min(s["oracle"]["optimal"])
            mass_opt = float(sum(p_opt))
            rows.append({
                "moves": s["moves"],
                "tie": s["tie"],
                "k": s["k_optimal"],
                "regret": regret,
                "optimal": int(chosen in s["oracle"]["optimal"]),
                "chosen": chosen,
                "mass_optimal": mass_opt,
                "mass_canonical": float(probs[canon]),
                "mass_other_optimal": mass_opt - float(probs[canon]),
                "peak_optimal": float(max(p_opt)),
            })
            if s["moves"] == 0:
                root_probs = [float(p) for p in probs]

        def summary(sel):
            if not sel:
                return {}
            reg = np.array([r["regret"] for r in sel], dtype=float)
            return {
                "n": len(sel),
                "mean_regret": float(reg.mean()),
                "median_regret": float(np.median(reg)),
                "max_regret": float(reg.max()),
                "optimal_action_rate": float(np.mean([r["optimal"] for r in sel])),
                "mean_mass_optimal": float(np.mean([r["mass_optimal"] for r in sel])),
                "mean_mass_canonical": float(np.mean([r["mass_canonical"] for r in sel])),
                "mean_mass_other_optimal": float(np.mean([r["mass_other_optimal"] for r in sel])),
                "mean_peak_optimal": float(np.mean([r["peak_optimal"] for r in sel])),
            }

        res = {
            "checkpoint": str(path),
            "sha256_16": sha,
            "all": summary(rows),
            "tie": summary([r for r in rows if r["tie"]]),
            "no_tie": summary([r for r in rows if not r["tie"]]),
            "by_phase": {
                f"moves_{m}": summary([r for r in rows if r["moves"] == m])
                for m in sorted({r["moves"] for r in rows})
            },
            "root_probs": root_probs,
        }
        out["results"][label] = res
        a, t = res["all"], res["tie"]
        print(f"{label:>14} sha:{sha} | regret {a['mean_regret']:.4f} "
              f"(med {a['median_regret']:.3f}) | opt {a['optimal_action_rate']:.3f} "
              f"| empatados: regret {t['mean_regret']:.4f} opt {t['optimal_action_rate']:.3f} "
              f"canon {t['mean_mass_canonical']:.3f} other {t['mean_mass_other_optimal']:.3f} "
              f"| mass_opt {a['mean_mass_optimal']:.3f}", flush=True)

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nCrudo: {p}")


if __name__ == "__main__":
    main()
