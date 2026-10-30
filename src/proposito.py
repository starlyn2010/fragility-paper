"""
proposito.py — ¿el agente tiene un PROPÓSITO o solo una CONDUCTA?

La pregunta filosofica: si un agente persigue un objetivo (ganar), su
conducta debe seguir persiguiendo ese objetivo aunque el mundo cambie y nadie
lo premie. Si en realidad solo memo (una conducta aprendida), su juego se
desvanece en cuanto el adversario deja de ser el que memorizo.

CONTRASTE (mismo mundo, sin recompensa en el juego, sin aprender mientras juega):
  - AGENTE: el del paper (entrenado CON recompensa, REINFORCE+GAE).
  - REGLA:  el minimax optimo (la imitacion ideal: copia la jugada optima
            siempre, dentro y fuera de distribucion). Es el "agente que hace
            exactamente lo que hay que hacer" y por lo tanto el mejor ejemplo
            de un agente que persigue el objetivo.

Si el AGENTE se parece a la REGLA en un mundo donde el mentor falla, entonces
persigue el objetivo (tiene un proposito). Si se aleja de la REGLA (mientras la
REGLA sigue jugando perfecto), entonces su conducta era una conducta, no un
objetivo.

Condiciones (el rival del lado derecho):
  1. mentor M6 determinista (el "examen": los dos deberian empatar)
  2. mentor que falla 15% / 30% (el mundo cambia: el proposito se aprovecha,
     la conducta memorizada se desvaste)
  3. minimax de profundidad 2 (rival debil pero sensato)
  4. rival aleatorio

Se mide el win/draw rate y el % de jugadas OPTIMAS de cada politica.

Uso:
  python src/proposito.py --games 80
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

from ablation_ttt import CONFIGS, AblationLiquidAgent  # noqa: E402
from distilar_mentor import jugada_optima  # noqa: E402
from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "ablation_logs"
VACIO, HUMANO, AGENTE = 0, 1, 2


def cargar(arch_id, path, hidden_dim):
    m = AblationLiquidAgent(arch_id, 9, 9, hidden_dim)
    m.s12_enabled = arch_id in ("A", "D")
    m.s12_max_iters = 3
    m.load_state_dict(torch.load(str(path), map_location="cpu"))
    m.eval()
    return m


def jugar(politica, rival, n_games, mentor=None, mentor2=None, aleatorio=False,
          seed=0, regla=False):
    """Juega n_games sin recompensa. Devuelve (wr, dr, % de jugadas optimas)."""
    rng = random.Random(seed)
    env = TicTacToeEnv()
    wins = draws = losses = 0
    opt_total = opt_ok = 0
    for _g in range(n_games):
        env.reset()
        # el estado interno se reinicia en cada partida
        h = torch.zeros(1, politica.hidden_dim) if (politica is not None and not regla) else None
        done = False
        turn = 0
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            if turn % 2 == 0:
                tab = tuple(env.grid[r][c] for r in range(3) for c in range(3))
                mejores, _v = jugada_optima(tab)
                if regla:
                    a = mejores[0] if mejores else legal[0]
                else:
                    obs = torch.from_numpy(np.array(tab, dtype=np.float32)).unsqueeze(0)
                    with torch.no_grad():
                        if getattr(politica, "s12_enabled", False):
                            logits, _v, h2, _p, _i = politica.forward_adaptive(obs, h)
                        else:
                            logits, _v, h2 = politica.forward_step(obs, h)
                    h = h2
                    mask = torch.full_like(logits, -1e9)
                    mask[0, legal] = 0.0
                    a = int((logits + mask).argmax(-1).item())
                if mejores:
                    opt_total += 1
                    opt_ok += int(a in mejores)
                if a not in legal:
                    a = legal[0]
                env.step(a, player=env.AGENTE)
            else:
                if aleatorio:
                    a = rng.choice(legal)
                elif mentor2 is not None:
                    a = mentor2.get_action(env, legal)
                else:
                    a = mentor.get_action(env, legal)
                env.step(a, player=env.HUMANO)
            turn += 1
            if done:
                break
        r = env.resultado()
        if r == 1:
            wins += 1
        elif r == 0:
            draws += 1
        else:
            losses += 1
    valor = (wins - losses) / n_games
    return (wins / n_games, draws / n_games,
            (opt_ok / opt_total) if opt_total else float("nan"), valor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--configs", default="A,B,C,D")
    ap.add_argument("--out", default=str(ROOT / "data" / "proposito.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = {"args": vars(args), "resultados": {}}

    condiciones = [
        ("M6 (examen)", dict(mentor=StochasticMinimax(depth=6, noise_prob=0.0))),
        ("M6 falla 15%", dict(mentor=StochasticMinimax(depth=6, noise_prob=0.15))),
        ("M6 falla 30%", dict(mentor=StochasticMinimax(depth=6, noise_prob=0.30))),
        ("minimax prof 2", dict(mentor2=StochasticMinimax(depth=2, noise_prob=0.0))),
        ("azar", dict(aleatorio=True)),
    ]

    print(f"{'config':<6}{'condicion':<16}{'agente wr':>11}{'regla wr':>10}"
          f"{'agente %opt':>13}{'regla %opt':>11}")
    for cfg in CONFIGS:
        if cfg["id"] not in args.configs.split(","):
            continue
        p_r = pathlib.Path(args.ckpt_dir) / f"ablation_{cfg['id']}.pth"
        if not p_r.exists():
            print(f"  {cfg['id']}: falta checkpoint")
            continue
        agente = cargar(cfg["id"], p_r, cfg["hidden_dim"])
        filas = {}
        for cond, kwargs in condiciones:
            kw = dict(kwargs)
            wr_a, dr_a, op_a, val_a = jugar(agente, None, args.games, seed=17, regla=False, **kw)
            wr_r, dr_r, op_r, val_r = jugar(None, None, args.games, seed=17, regla=True, **kw)
            filas[cond] = {"agente": {"wr": wr_a, "dr": dr_a, "frac_optimas": op_a, "valor": val_a},
                           "regla": {"wr": wr_r, "dr": dr_r, "frac_optimas": op_r, "valor": val_r}}
            print(f"{cfg['id']:<6}{cond:<16}{wr_a:>10.0%}{wr_r:>10.0%}"
                  f"{op_a:>12.0%}{op_r:>11.0%}  | valor {val_a:+.2f} vs {val_r:+.2f}")
        out["resultados"][cfg["id"]] = {"label": cfg["label"], **filas}

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nCrudo: {p}")


if __name__ == "__main__":
    main()
