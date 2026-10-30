"""
twin_mentor.py — ¿el agente aprendió las reglas o memorizó al maestro?

La pregunta del paper ("latent hash map") es, en su versión dura: ¿el agente
aprendió a JUGAR o aprendió a RESPONDER a un adversario concreto? Son dos
cosas distintas y se pueden separar con un experimento limpio.

Diseño: DOS MENTORES GEMELOS.
  - Mentor A: el minimax determinista de siempre (el que se usa en el
    entrenamiento: desempata SIEMPRE con la primera jugada óptima).
  - Mentor B: EXACTAMENTE la misma fuerza (juega óptimamente), pero desempata
    al azar entre las jugadas igualmente óptimas.

Los dos son indistinguibles en fuerza: contra un jugador perfecto ambos
empatizan. Lo único que cambia es el ESTILO. Por lo tanto:
  - si el agente aprendió las REGLAS, le da igual: juega igual contra A que
    contra B;
  - si memorizó al MAESTRO (huella digital de A), se rompe contra B aunque B
    juegue igual de óptimamente.

Se mide:
  1. WR/DR contra A y contra B (N partidas cada uno, mismo protocolo).
  2. La diferencia: si B rompe al agente, la fragilidad es de estilo, no de
     fuerza.
  3. Como control, un mentor C con ruido (fuerza menor) para verificar que el
     detector de degradaciones responde.

Es la version computable de "saber una regla es lo mismo que conocer a quien
la aplica".

Uso:
  python src/twin_mentor.py --games 200
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

from ablation_ttt import CONFIGS, AblationLiquidAgent, evaluate_agent  # noqa: E402
from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "ablation_logs"


class TieBreakMinimax(StochasticMinimax):
    """Gemelo del mentor: MISMA fuerza, distinto estilo.

    Juega la jugada minimax, pero cuando hay varias jugadas igualmente
    óptimas elige una al azar (el minimax original elige siempre la primera).
    No toca la profundidad ni la evaluacion: solo el desempate.
    """

    def __init__(self, seed: int = 0, **kw):
        super().__init__(**kw)
        self.rng = random.Random(seed)

    def _best_action(self, env, legal, depth):
        vals = []
        for a in legal:
            c = env.clone()
            c.step(a, player=env.HUMANO)
            vals.append((self._minimax(c, depth, -float("inf"), float("inf"), False), a))
        best = max(v for v, _a in vals)
        winners = [a for v, a in vals if v == best]
        return self.rng.choice(winners)


def cargar(cfg, path):
    agent = AblationLiquidAgent(cfg["id"], 9, 9, cfg["hidden_dim"])
    agent.s12_enabled = cfg["act"]
    agent.s12_max_iters = 3
    agent.load_state_dict(torch.load(str(path), map_location="cpu"))
    agent.eval()
    return agent


def valor_de(mentor, env, accion):
    """Valor (segun el evaluador del mentor) de jugar `accion`."""
    c = env.clone()
    c.step(accion, player=env.HUMANO)
    return mentor._minimax(c, mentor.depth, -float("inf"), float("inf"), False)


def control_equivalencia(m_a, m_b, n_games=30):
    """Control: el gemelo debe ser IGUAL de fuerte.

    En cada estado donde A y B eligen jugadas distintas, se comprueba que las
    dos jugadas tienen EXACTAMENTE el mismo valor: o sea, las dos son
    igualmente óptimas y lo único que cambia es el estilo.
    """
    env = TicTacToeEnv()
    n_est = n_difieren = n_mismo_valor = 0
    rng = random.Random(0)
    for _ in range(n_games):
        env.reset()
        done = False
        turn = 0
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            if turn % 2 == 0:
                env.step(rng.choice(legal), player=env.AGENTE)
            else:
                a = m_a.get_action(env, legal)
                b = m_b.get_action(env, legal)
                n_est += 1
                if a != b:
                    n_difieren += 1
                    if abs(valor_de(m_a, env, a) - valor_de(m_a, env, b)) < 1e-9:
                        n_mismo_valor += 1
                env.step(a, player=env.HUMANO)
            turn += 1
            if done:
                break
    return {"estados_mentor": n_est, "difieren": n_difieren,
            "mismo_valor": n_mismo_valor,
            "frac_difieren": n_difieren / max(1, n_est),
            "frac_difieren_iguales": n_mismo_valor / max(1, n_difieren)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--configs", default="A,B,C,D")
    ap.add_argument("--out", default=str(ROOT / "data" / "twin_mentor.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    env = TicTacToeEnv()

    mentores = {
        "A_determinista": StochasticMinimax(depth=6, noise_prob=0.0),
        "B_gemelo_tiebreak": TieBreakMinimax(depth=6, noise_prob=0.0, seed=1234),
        "C_con_ruido": StochasticMinimax(depth=6, noise_prob=0.10),
    }
    out = {"args": vars(args), "resultados": {}}
    # CONTROL: el gemelo debe ser igual de fuerte que el mentor
    ctrl = control_equivalencia(StochasticMinimax(depth=6, noise_prob=0.0),
                                TieBreakMinimax(depth=6, noise_prob=0.0, seed=1234),
                                n_games=30)
    out["control_equivalencia"] = ctrl
    print(f"CONTROL: en {ctrl['difieren']} estados el mentor y el gemelo eligen distinto; "
          f"de esos, {ctrl['frac_difieren_iguales']:.0%} tienen EXACTAMENTE el mismo valor "
          f"(-> misma fuerza, distinto estilo). El mentor difiere en el "
          f"{ctrl['frac_difieren']:.0%} de los estados.\n")
    print(f"{'config':<6}" + "".join(f"{k:>24}" for k in mentores) + f"{'B-A':>10}")
    for cfg in CONFIGS:
        if cfg["id"] not in args.configs.split(","):
            continue
        path = pathlib.Path(args.ckpt_dir) / f"ablation_{cfg['id']}.pth"
        if not path.exists():
            print(f"  {cfg['id']}: falta checkpoint")
            continue
        agent = cargar(cfg, path)
        fila = {}
        for nombre, opp in mentores.items():
            w, d = evaluate_agent(agent, env, opp, n=args.games)
            fila[nombre] = {"wr": w, "dr": d, "losses": 1.0 - w - d}
            print(f"  {cfg['id']} vs {nombre:<20} WR {w*100:5.1f}%  DR {d*100:5.1f}%  "
                  f"derrotas {(1-w-d)*100:5.1f}%", flush=True)
        delta = fila["B_gemelo_tiebreak"]["dr"] - fila["A_determinista"]["dr"]
        fila["delta_dr_B_menos_A"] = delta
        out["resultados"][cfg["id"]] = {"label": cfg["label"], **fila}
        veredicto = ("SE ROMPE con el gemelo" if delta < -0.25 else "aguanta al gemelo")
        print(f"  -> {cfg['id']}: los empates caen de {fila['A_determinista']['dr']*100:.0f}% "
              f"a {fila['B_gemelo_tiebreak']['dr']*100:.0f}% con un rival IGUAL de fuerte "
              f"({delta*100:+.0f} pts) -> {veredicto}\n", flush=True)

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Crudo: {p}")


if __name__ == "__main__":
    main()
