"""
distilar_mentor.py — un agente que NUNCA recibe recompensa.

Para el experimento "proposito sin premio" hace falta un segundo tipo de
agente: uno que sabe jugar pero que fue entrenado SIN recompensa, solo imitando
al mentor en las posiciones que realmente aparecen en el juego.

Se destila la jugada OPTIMA del primer jugador (minimax correcto, escrito
a mano) sobre las posiciones que el agente^{*}(mentor) ve en sus partidas reales
-> el resultado es un agente "imitador": copia la conducta del mentor SOLO en
distribución; fuera de ella no tiene a quién imitar.

Guarda: ablation_logs/bc_<cfg>.pth + el accuracy de train/val.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_ttt import CONFIGS, AblationLiquidAgent  # noqa: E402
from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "ablation_logs"
VACIO, HUMANO, AGENTE = 0, 1, 2


# ----------------------------------------------------------------- minimax
def valor(env, turno):
    """Valor desde la perspectiva del primer jugador (+1 gana AGENTE)."""
    if env.done:
        return env.resultado()
    legal = env.legal_actions()
    if not legal:
        return 0
    vals = []
    for a in legal:
        c = env.clone()
        c.step(a, player=turno)
        vals.append(valor(c, HUMANO if turno == AGENTE else AGENTE))
    return max(vals) if turno == AGENTE else min(vals)


@lru_cache(maxsize=None)
def jugada_optima(tablero):
    env = TicTacToeEnv()
    env.grid = [[tablero[3 * r + c] for c in range(3)] for r in range(3)]
    legal = env.legal_actions()
    if not legal:
        return None, []
    mejores = []
    mejor = -99
    for a in legal:
        c = env.clone()
        c.step(a, player=AGENTE)
        v = valor(c, HUMANO)
        if v > mejor:
            mejor, mejores = v, [a]
        elif v == mejor:
            mejores.append(a)
    return mejores, mejor


# ----------------------------------------------------------------- datos
def posiciones_del_agente(agent, mentor, n_games, max_pos, explorar=True, rng=None):
    """Tableros que el agente ve en sus turnos, jugando de verdad vs M6.

    Con `explorar=True` se MUESTREA la politica (en vez de argmax): si no, el
    agente juega siempre igual y apenas ve unas pocas posiciones distintas
    (con 6 partidas: 7 posiciones -> la destilacion no tiene nada que aprender).
    """
    rng = rng or np.random.default_rng(0)
    env = TicTacToeEnv()
    vistos, vistos_set = [], set()
    for _g in range(n_games):
        env.reset()
        h = torch.zeros(1, agent.hidden_dim)
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
                obs = torch.from_numpy(np.array(tab, dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    if getattr(agent, "s12_enabled", False):
                        logits, _v, h2, _p, _i = agent.forward_adaptive(obs, h)
                    else:
                        logits, _v, h2 = agent.forward_step(obs, h)
                if explorar:
                    probs = torch.softmax(logits, -1).squeeze(0).numpy()
                    probs = np.where(np.isin(np.arange(9), legal), probs, 0.0)
                    probs = probs / probs.sum()
                    a = int(rng.choice(9, p=probs))
                else:
                    a = int(logits.argmax(-1).item())
                env.step(a, player=env.AGENTE)
                h = h2
            else:
                env.step(mentor.get_action(env, legal), player=env.HUMANO)
            turn += 1
            if done:
                break
        if len(vistos) >= max_pos:
            break
    return vistos


# ----------------------------------------------------------------- main
def cargar(cfg, path):
    agent = AblationLiquidAgent(cfg["id"], 9, 9, cfg["hidden_dim"])
    agent.s12_enabled = cfg["act"]
    agent.s12_max_iters = 3
    agent.load_state_dict(torch.load(str(path), map_location="cpu"))
    agent.eval()
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--positions", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--configs", default="A,B,C,D")
    ap.add_argument("--out", default=str(ROOT / "data" / "distilacion_mentor.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    mentor = StochasticMinimax(depth=6, noise_prob=0.0)
    out = {"args": vars(args), "resultados": {}}

    for cfg in CONFIGS:
        if cfg["id"] not in args.configs.split(","):
            continue
        path = pathlib.Path(args.ckpt_dir) / f"ablation_{cfg['id']}.pth"
        if not path.exists():
            print(f"  {cfg['id']}: falta checkpoint")
            continue
        agente = cargar(cfg, path)
        pos = posiciones_del_agente(agente, mentor, args.games, args.positions)
        # objetivos: la jugada optima del primer jugador
        X, Y, opt_sets = [], [], []
        for tab in pos:
            mejores, _v = jugada_optima(tab)
            if not mejores:
                continue
            X.append(tab)
            Y.append(mejores[0])
            opt_sets.append(mejores)
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.int64)
        n = len(X)
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        nval = max(1, n // 5)
        va, tr = idx[:nval], idx[nval:]

        # modelo NUEVO (misma arquitectura), entrenado solo con imitacion
        torch.manual_seed(0)
        bc = AblationLiquidAgent(cfg["id"], 9, 9, cfg["hidden_dim"])
        bc.s12_enabled = cfg["act"]
        bc.s12_max_iters = 3
        opt = torch.optim.Adam(bc.parameters(), lr=args.lr)

        def acc(idx_set):
            """(argmax exacto, dentro del conjunto optimo)"""
            with torch.no_grad():
                h = torch.zeros(len(idx_set), bc.hidden_dim)
                o = torch.from_numpy(X[idx_set])
                if getattr(bc, "s12_enabled", False):
                    lg, _v, h, _p, _i = bc.forward_adaptive(o, h)
                else:
                    lg, _v, h = bc.forward_step(o, h)
                pred = lg.argmax(-1).numpy()
            exacto = float(np.mean(pred == Y[idx_set]))
            dentro = float(np.mean([pred[i] in opt_sets[j]
                                     for i, j in enumerate(idx_set)]))
            return exacto, dentro

        _losses = []
        for ep in range(1, args.epochs + 1):
            bc.train()
            perm = rng.permutation(len(tr))
            for s in range(0, len(perm), 32):
                b = tr[perm[s:s + 32]]
                if len(b) < 2:
                    continue
                h = torch.zeros(len(b), bc.hidden_dim)
                o = torch.from_numpy(X[b])
                if getattr(bc, "s12_enabled", False):
                    lg, _v, h, _p, _i = bc.forward_adaptive(o, h)
                else:
                    lg, _v, h = bc.forward_step(o, h)
                mask = torch.full_like(lg, -1e4)
                for i, bi in enumerate(b):
                    mask[i, opt_sets[bi]] = 0.0
                lg = lg + mask
                if not torch.isfinite(lg).all():
                    lg = torch.nan_to_num(lg, nan=0.0, posinf=0.0, neginf=-1e4)
                loss = F.cross_entropy(lg, torch.from_numpy(Y[b]))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bc.parameters(), 1.0)
                opt.step()
            if ep % 50 == 0 or ep == args.epochs:
                a_tr, o_tr = acc(tr)
                a_va, o_va = acc(va)
                print(f"  {cfg['id']}: ep {ep:4d} | loss {np.mean(_losses[-40:]):.3f} | argmax train {a_tr:.1%} "
                      f"val {a_va:.1%} | dentro del optimo train {o_tr:.1%} val {o_va:.1%}",
                      flush=True)
        torch.save(bc.state_dict(), CKPT_DIR / f"bc_{cfg['id']}.pth")
        a_tr, o_tr = acc(tr)
        a_va, o_va = acc(va)
        out["resultados"][cfg["id"]] = {"label": cfg["label"], "n_posiciones": n,
                                        "acc_argmax_train": a_tr, "acc_argmax_val": a_va,
                                        "acc_optimo_train": o_tr, "acc_optimo_val": o_va}
        print(f"  -> {cfg['id']}: guardado bc_{cfg['id']}.pth "
              f"(dentro del optimo: train {o_tr:.1%} val {o_va:.1%})", flush=True)

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nCrudo: {p}")


if __name__ == "__main__":
    main()
