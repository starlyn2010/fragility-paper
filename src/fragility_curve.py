"""
fragility_curve.py — la fragilidad como CURVA, no como un punto.

El paper mide el colapso contra un solo rival estocastico (Random). Lo que
falta es la curva: ¿de cuanta desviacion del adversario se rompe cada
config? Si el agente aprendio un "mapa hash" del mentor determinista, deberia
ser solido mientras el mentor juega como el mentor y romperse de golpe en
cuanto el mentor se desvía (ruido) o es mas debil (menor profundidad).

Dos barridos por config:
  A) RUIDO: minimax de profundidad 6 con noise_prob p (p=0 es el mentor
     determinista; p=1 es juego casi aleatorio).
  B) PROFUNDIDAD: mentor de profundidad d (0..6), sin ruido: la escalera de
     habilidad del adversario.

Salida: data/fragility_curve.json (+ figura si hay matplotlib).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_ttt import CONFIGS, AblationLiquidAgent, evaluate_agent  # noqa: E402
from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "ablation_logs"


def cargar(cfg, path):
    agent = AblationLiquidAgent(cfg["id"], 9, 9, cfg["hidden_dim"])
    agent.s12_enabled = cfg["act"]
    agent.s12_max_iters = 3
    agent.load_state_dict(torch.load(str(path), map_location="cpu"))
    agent.eval()
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100, help="partidas por punto de la curva")
    ap.add_argument("--ruido", default="0,0.02,0.05,0.1,0.2,0.4,0.7,1.0")
    ap.add_argument("--profundidad", default="0,1,2,3,4,5,6")
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--configs", default="A,B,C,D")
    ap.add_argument("--out", default=str(ROOT / "data" / "fragility_curve.json"))
    ap.add_argument("--fig", default=str(ROOT / "paper" / "fig3_fragility_curve.pdf"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = {"args": vars(args), "curvas": {}}
    ruidos = [float(x) for x in args.ruido.split(",")]
    profs = [int(x) for x in args.profundidad.split(",")]
    for cfg in CONFIGS:
        if cfg["id"] not in args.configs.split(","):
            continue
        path = pathlib.Path(args.ckpt_dir) / f"ablation_{cfg['id']}.pth"
        if not path.exists():
            print(f"  {cfg['id']}: falta checkpoint")
            continue
        agent = cargar(cfg, path)
        env = TicTacToeEnv()
        curva_ruido, curva_prof = [], []
        for p in ruidos:
            opp = StochasticMinimax(depth=6, noise_prob=p)
            w, d = evaluate_agent(agent, env, opp, n=args.games)
            curva_ruido.append({"ruido": p, "wr": w, "dr": d})
            print(f"  {cfg['id']} ruido {p:<5} WR {w*100:5.1f}%  DR {d*100:5.1f}%", flush=True)
        for dep in profs:
            opp = StochasticMinimax(depth=dep, noise_prob=0.0)
            w, d = evaluate_agent(agent, env, opp, n=args.games)
            curva_prof.append({"prof": dep, "wr": w, "dr": d})
            print(f"  {cfg['id']} prof {dep:<5} WR {w*100:5.1f}%  DR {d*100:5.1f}%", flush=True)
        # punto de quiebre: el menor ruido con WR < 0.5
        quiebre = None
        for c in curva_ruido:
            if c["wr"] < 0.5:
                quiebre = c["ruido"]
                break
        out["curvas"][cfg["id"]] = {"label": cfg["label"], "ruido": curva_ruido,
                                    "profundidad": curva_prof, "quiebre_ruido": quiebre}
        print(f"  -> {cfg['id']}: se rompe con ruido >= {quiebre}\n", flush=True)

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Crudo: {p}")

    # figura (opcional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for cid, c in out["curvas"].items():
            x = [r["ruido"] for r in c["ruido"]]
            y = [r["wr"] for r in c["ruido"]]
            ax[0].plot(x, y, marker="o", label=cid)
            x2 = [r["prof"] for r in c["profundidad"]]
            y2 = [r["wr"] for r in c["profundidad"]]
            ax[1].plot(x2, y2, marker="s", label=cid)
        ax[0].set_xlabel("ruido del mentor (noise\_prob)")
        ax[0].set_ylabel("win rate (%)")
        ax[0].set_title("Fragilidad vs ruido del adversario")
        ax[1].set_xlabel("profundidad del adversario")
        ax[1].set_title("Escalera de habilidad")
        for a in ax:
            a.grid(alpha=.3)
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.fig)
        print(f"Figura: {args.fig}")
    except Exception as exc:  # noqa: BLE001
        print(f"(sin figura: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
