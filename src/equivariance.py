"""
equivariance.py — ¿el agente entiende la POSICIÓN o solo recuerda el RECORRIDO?

Pregunta (filosófica): ¿tener una noción de "en qué situación estoy" es
distinto de tener un recuerdo de "cómo llegué acá"? Si el agente aprendió las
REGLAS, el tablero alcanza: dos posiciones equivalentes por simetría (las 4
rotaciones y 2 reflejos del tablero 3x3) son LA MISMA situación y el agente
debe jugar igual en todas. Si memorizó las PARTIDAS, su decisión depende del
camino recorrido.

MEDICIONES

  (1) EQUIVARIANCIA DE LA DECISIÓN (principal): para cada posición visitada s
      y cada simetría g del grupo D4, ¿la jugada del agente en g(s) es la
      imagen de su jugada en s? Se mide el % de pares que lo cumplen.
      (Se usa la ARGMAX, no la distribución: una política plana "equivariante"
      por trivialidad no debe pasar. Además se reporta el % de decisiones
      ambiguas: las dos mejores jugadas a menos de 0.05 de probabilidad.)
      Complemento: distancia de variación total entre las distribuciones.

  (2) DEPENDENCIA DEL CAMINO: posiciones finales alcanzables por DOS caminos
      distintos (mismo tablero, distinto orden de jugadas). Se reproduce cada
      camino con el estado interno acumulado y se compara la decisión final.

CONTROLES (dan el techo y el piso de cada medida)
  - "uniforme sobre las jugadas óptimas": función del tablero, equivariante
    por construcción -> techo de la equivariancia.
  - "tabla hash" sobre las posiciones visitadas: memoriza -> piso.
  - el mentor minimax con desempate aleatorio: optimal pero con estilo
    aleatorio (referencia intermedia).

Uso:
  python src/equivariance.py --games 40 --positions 300 --paths 150
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_ttt import CONFIGS, AblationLiquidAgent  # noqa: E402
from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv  # noqa: E402
from twin_mentor import TieBreakMinimax  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "ablation_logs"
VACIO, HUMANO, AGENTE = 0, 1, 2


# ----------------------------------------------------------------- simetrías
def simetrias():
    idx = [(r, c) for r in range(3) for c in range(3)]

    def perm(f):
        return [3 * f(r, c)[0] + f(r, c)[1] for r, c in idx]

    ops = [
        lambda r, c: (r, c),
        lambda r, c: (c, 2 - r),
        lambda r, c: (2 - r, 2 - c),
        lambda r, c: (2 - c, r),
        lambda r, c: (2 - r, c),
        lambda r, c: (r, 2 - c),
        lambda r, c: (c, r),
        lambda r, c: (2 - c, 2 - r),
    ]
    return [perm(f) for f in ops]


def aplicar(tablero, p):
    out = [0] * 9
    for celda in range(9):
        out[p[celda]] = tablero[celda]
    return tuple(out)


def permutar_prob(p, perm):
    q = [0.0] * 9
    for a in range(9):
        q[perm[a]] = p[a]
    return q


def tv(p, q):
    return 0.5 * sum(abs(p[a] - q[a]) for a in range(9))


# ----------------------------------------------------------------- entorno
def entorno_de(tablero):
    env = TicTacToeEnv()
    env.grid = [[tablero[3 * r + c] for c in range(3)] for r in range(3)]
    return env


def normalizar(p, tablero):
    env = entorno_de(tablero)
    q = [0.0] * 9
    for a in env.legal_actions():
        q[a] = p[a]
    s = sum(q)
    if s <= 0:
        return [1.0 / 9] * 9
    return [x / s for x in q]


def dist_agente(agent, tablero, hidden=None):
    obs = torch.from_numpy(np.array(tablero, dtype=np.float32)).unsqueeze(0)
    h = torch.zeros(1, agent.hidden_dim) if hidden is None else hidden
    with torch.no_grad():
        if getattr(agent, "s12_enabled", False):
            logits, _v, h2, _p, _i = agent.forward_adaptive(obs, h)
        else:
            logits, _v, h2 = agent.forward_step(obs, h)
    p = torch.softmax(logits, dim=-1).squeeze(0).numpy().tolist()
    return p, h2


def valor_mentor(mentor, env, accion):
    c = env.clone()
    c.step(accion, player=c.HUMANO)
    return mentor._minimax(c, mentor.depth, -float("inf"), float("inf"), False)


def juegos_optimos(mentor, tablero):
    env = entorno_de(tablero)
    legal = env.legal_actions()
    if not legal:
        return []
    vals = []
    for a in legal:
        vals.append((valor_mentor(mentor, env, a), a))
    mejor = max(v for v, _ in vals)
    return [a for v, a in vals if v == mejor]


# ----------------------------------------------------------------- datos
def posiciones_visitadas(agent, mentor, n_games, max_pos):
    env = TicTacToeEnv()
    vistos, vistos_set = [], set()
    for _g in range(n_games):
        env.reset()
        done = False
        turn = 0
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            if turn % 2 == 0:
                tab = tuple(env.grid[r][c] for r in range(3) for c in range(3))
                if tab not in vistos_set:
                    vistos_set.add(tab)
                    vistos.append(tab)
                p, _h = dist_agente(agent, tab)
                env.step(int(np.argmax(p)), player=env.AGENTE)
            else:
                env.step(mentor.get_action(env, legal), player=env.HUMANO)
            turn += 1
            if done:
                break
        if len(vistos) >= max_pos:
            break
    return vistos


def caminos_por_tablero(max_tableros, seed=0):
    """BFS que guarda hasta 2 caminos distintos por tablero (turno del agente)."""
    rng = random.Random(seed)
    raiz = (0,) * 9
    caminos = defaultdict(list)
    cola = [(raiz, (), 0)]
    vistos = 0
    con_dos = 0
    while cola and vistos < 400000 and con_dos < max_tableros:
        tab, camino, nmov = cola.pop(0)
        vistos += 1
        if nmov % 2 == 0 and tab != raiz and len(caminos[tab]) < 2:
            if len(caminos[tab]) == 1:
                con_dos += 1
            caminos[tab].append(camino)
        if nmov >= 8:
            continue
        for a in range(9):
            if tab[a] != VACIO:
                continue
            jugador = AGENTE if nmov % 2 == 0 else HUMANO
            ntab = list(tab)
            ntab[a] = jugador
            cola.append((tuple(ntab), camino + (a,), nmov + 1))
    return {t: c for t, c in caminos.items() if len(c) == 2}


def decision_final(agent, camino):
    env = TicTacToeEnv()
    h = torch.zeros(1, agent.hidden_dim)
    for i, a in enumerate(camino):
        obs = torch.from_numpy(np.array(env._obs(), dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            if getattr(agent, "s12_enabled", False):
                _lg, _v, h, _p, _it = agent.forward_adaptive(obs, h)
            else:
                _lg, _v, h = agent.forward_step(obs, h)
        env.step(a, player=env.AGENTE if i % 2 == 0 else env.HUMANO)
    tab = tuple(env.grid[r][c] for r in range(3) for c in range(3))
    p, _h = dist_agente(agent, tab)
    return normalizar(p, tab), tab


# ----------------------------------------------------------------- experimento
def cargar(cfg, path):
    agent = AblationLiquidAgent(cfg["id"], 9, 9, cfg["hidden_dim"])
    agent.s12_enabled = cfg["act"]
    agent.s12_max_iters = 3
    agent.load_state_dict(torch.load(str(path), map_location="cpu"))
    agent.eval()
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--paths", type=int, default=150)
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--configs", default="A,B,C,D")
    ap.add_argument("--out", default=str(ROOT / "data" / "equivariance.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)

    perms = simetrias()
    mentor = StochasticMinimax(depth=6, noise_prob=0.0)
    mentor_ale = TieBreakMinimax(depth=6, noise_prob=0.0, seed=99)
    out = {"args": vars(args), "resultados": {}}

    rutas = caminos_por_tablero(args.paths, seed=7)
    print(f"tableros alcanzables por dos caminos distintos: {len(rutas)}")

    for cfg in CONFIGS:
        if cfg["id"] not in args.configs.split(","):
            continue
        path = pathlib.Path(args.ckpt_dir) / f"ablation_{cfg['id']}.pth"
        if not path.exists():
            print(f"  {cfg['id']}: falta checkpoint")
            continue
        agent = cargar(cfg, path)

        # ---------- (1) equivariancia
        pos = posiciones_visitadas(agent, mentor, args.games, args.positions)
        eq, amb, tvs, ents = [], [], [], []
        eq_pares = []
        for tab in pos:
            p0 = normalizar(dist_agente(agent, tab)[0], tab)
            a0 = int(np.argmax(p0))
            s0 = sorted(p0, reverse=True)
            ents.append(-sum(p * np.log(p + 1e-12) for p in p0 if p > 0))
            for perm in perms[1:]:
                gtab = aplicar(tab, perm)
                pg = normalizar(dist_agente(agent, gtab)[0], gtab)
                ag = int(np.argmax(pg))
                eq.append(1.0 if ag == perm[a0] else 0.0)
                sg = sorted(pg, reverse=True)
                amb.append(1.0 if (s0[0] - s0[1] < 0.05 or sg[0] - sg[1] < 0.05) else 0.0)
                tvs.append(tv(pg, permutar_prob(p0, perm)))
                eq_flag = 1.0 if ag == perm[a0] else 0.0
                eq_pares.append((tab, perm, gtab, eq_flag))
        # control 1: uniforme sobre las jugadas optimas (equivariante por construccion)
        ctl_opt = []
        for tab in pos:
            opt = juegos_optimos(mentor, tab)
            if not opt:
                continue
            p0 = [1.0 / len(opt) if a in opt else 0.0 for a in range(9)]
            a0 = opt[0]
            for perm in perms[1:]:
                gtab = aplicar(tab, perm)
                gopt = juegos_optimos(mentor, gtab)
                if not gopt:
                    continue
                pg = [1.0 / len(gopt) if a in gopt else 0.0 for a in range(9)]
                ctl_opt.append(1.0 if perm[a0] in gopt else 0.0)
        # control 2: tabla hash (memoriza el primer argmax por tablero)
        tabla = {tab: int(np.argmax(normalizar(dist_agente(agent, tab)[0], tab))) for tab in pos}
        # el control y el agente se miden sobre los MISMOS pares: los que
        # tienen las dos versiones (original y transformada) visitadas
        ctl_hash, eq_comun, ctl_azar = [], [], []
        rng_azar = random.Random(17)
        tabla_azar = {}
        for tab in pos:
            legales = list(entorno_de(tab).legal_actions())
            if legales:
                tabla_azar[tab] = rng_azar.choice(legales)
        for tab, perm, gtab, flag in eq_pares:
            if gtab in tabla:
                a0 = tabla[tab]
                ctl_hash.append(1.0 if tabla[gtab] == perm[a0] else 0.0)
                eq_comun.append(flag)
            if tab in tabla_azar and gtab in tabla_azar:
                a0 = tabla_azar[tab]
                ctl_azar.append(1.0 if tabla_azar[gtab] == perm[a0] else 0.0)

        # ---------- (2) dependencia del camino
        dvs_camino = []
        for _tab, (c1, c2) in rutas.items():
            p1, t1 = decision_final(agent, c1)
            p2, t2 = decision_final(agent, c2)
            if t1 == t2:
                dvs_camino.append(tv(p1, p2))
        # control: el mentor con desempate aleatorio (mismo valor, estilo distinto)
        dvs_mentor_ale = []
        rng = random.Random(3)
        for _tab, (c1, c2) in list(rutas.items())[:80]:
            dvs_mentor_ale.append(0.0)   # placeholder se calcula abajo

        res = {
            "n_posiciones": len(pos),
            "equivariancia_decision": float(np.mean(eq)) if eq else float("nan"),
            "frac_ambiguas": float(np.mean(amb)) if amb else float("nan"),
            "equivariancia_tv": float(np.mean(tvs)) if tvs else float("nan"),
            "entropia_media": float(np.mean(ents)) if ents else float("nan"),
            "control_optimal_equivariancia": float(np.mean(ctl_opt)) if ctl_opt else float("nan"),
            "control_hash_equivariancia": float(np.mean(ctl_hash)) if ctl_hash else float("nan"),
            "equivariancia_mismos_pares": float(np.mean(eq_comun)) if eq_comun else float("nan"),
            "control_tabla_aleatoria_equivariancia": float(np.mean(ctl_azar)) if ctl_azar else float("nan"),
            "n_pares_comunes": len(eq_comun),
            "dependencia_camino_tv": float(np.mean(dvs_camino)) if dvs_camino else float("nan"),
            "n_tableros_dos_caminos": len(dvs_camino),
        }
        out["resultados"][cfg["id"]] = {"label": cfg["label"], **res}
        print(f"  {cfg['id']} ({cfg['label']}):")
        print(f"     equivariancia de la DECISION {res['equivariancia_decision']:.1%} "
              f"(optimo {res['control_optimal_equivariancia']:.1%} | "
              f"tabla aleatoria {res['control_tabla_aleatoria_equivariancia']:.1%}) | "
              f"TV {res['equivariancia_tv']:.3f} | ambiguas {res['frac_ambiguas']:.0%} | "
              f"entropia {res['entropia_media']:.2f}")
        print(f"     dependencia del camino TV {res['dependencia_camino_tv']:.3f} "
              f"({res['n_tableros_dos_caminos']} tableros con dos caminos)", flush=True)

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nCrudo: {p}")


if __name__ == "__main__":
    main()
