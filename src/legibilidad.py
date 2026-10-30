"""
legibilidad.py — ¿se puede LEER la mente del agente?

Pregunta (filosófica): una máquina cuyas decisiones no se pueden explicar
¿puede tener responsabilidad? Antes de eso hay que ver si sus decisiones se
pueden explicar en ALGUN sentido. Cinco mediciones:

  (1) MAPA DE ATENCIÓN AL TABLERO: para cada casilla, cuanto cambia la
      decision si se la mueve (intercambio con una casilla vacia). Da un mapa
      3x3 de "qué mira el agente" -> es la figura mas legible del experimento.
  (2) IMPORTANCIA DE LAS DIMENSIONES LATENTES: se anula cada una de las
      300 dims del estado interno y se mide cuanto cambia la decision. De ahi
      sale: (a) el perfil de importancia, (b) cuantos numeros bastan para
      reconstruir la decision (fidelidad vs k).
  (3) INTERCAMBIO DE ESTADO: se le da el estado interno de OTRA posicion y se
      ve que decision toma. Si la decision sigue al estado prestado, el estado
      interno es una "orden"; si es una mezcla, el estado es un estado.
  (4) CONTRAFACTUAL MINIMO: el cambio MAS PEQUEÑO del tablero que le hace
      cambiar la jugada (mover una sola pieza). Si existe para casi todas las
      decisiones, ladecision es "delicada" y por lo tanto explicable.
  (5) LEGIBILIDAD: cuantos numeros internos bastan para predecir la jugada
      (sonda lineal sobre las k dims mas importantes, accuracy sobre decisiones
      no vistas), comparado con la sonda que solo ve el tablero.

Uso:
  python src/legibilidad.py --games 25 --decisions 400
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

from ablation_ttt import CONFIGS, AblationLiquidAgent  # noqa: E402
from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "ablation_logs"
VACIO, HUMANO, AGENTE = 0, 1, 2


# ----------------------------------------------------------------- utilidades
def dist(agent, obs, h):
    with torch.no_grad():
        if getattr(agent, "s12_enabled", False):
            logits, _v, h2, _p, _i = agent.forward_adaptive(obs, h)
        else:
            logits, _v, h2 = agent.forward_step(obs, h)
    return torch.softmax(logits, dim=-1).squeeze(0).numpy(), h2


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def entorno_de(tablero):
    env = TicTacToeEnv()
    env.grid = [[tablero[3 * r + c] for c in range(3)] for r in range(3)]
    return env


def obs_de(tablero):
    return torch.from_numpy(np.array(tablero, dtype=np.float32)).unsqueeze(0)


def distribucion(agent, tablero, h=None):
    p, _ = dist(agent, obs_de(tablero), h if h is not None else torch.zeros(1, agent.hidden_dim))
    legales = entorno_de(tablero).legal_actions()
    q = np.zeros(9)
    for a in legales:
        q[a] = p[a]
    s = q.sum()
    if s <= 0:
        return np.ones(9) / 9
    return q / s


# ----------------------------------------------------------------- datos
def recolectar(agent, mentor, n_games, max_dec):
    """Decisiones reales: (tablero, hidden, distribucion, jugada)."""
    env = TicTacToeEnv()
    dec = []
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
                p, h2 = dist(agent, obs_de(tab), h)
                q = np.zeros(9)
                for a in legal:
                    q[a] = p[a]
                s = q.sum()
                q = q / s if s > 0 else np.ones(9) / 9
                dec.append({"tab": tab, "h": h2.clone(), "p": q, "a": int(np.argmax(q)),
                            "legales": list(legal), "partida": _g})
                if len(dec) >= max_dec:
                    return dec
                env.step(int(np.argmax(q)), player=env.AGENTE)
                h = h2
            else:
                env.step(mentor.get_action(env, legal), player=env.HUMANO)
            turn += 1
            if done:
                break
    return dec


# ----------------------------------------------------------------- mediciones
def mapa_tablero(agent, dec, n=120):
    """(1) importancia de cada casilla: cuanto cambia la decision al moverla."""
    imp = np.zeros(9)
    cuentas = np.zeros(9)
    for d in dec[:n]:
        tab, h, p0 = d["tab"], d["h"], d["p"]
        vacias = [i for i in range(9) if tab[i] == VACIO]
        for c in range(9):
            if tab[c] == VACIO:
                continue
            for e in vacias:
                ntab = list(tab)
                ntab[c], ntab[e] = ntab[e], ntab[c]
                p1 = distribucion(agent, tuple(ntab), h)
                imp[c] += tv(p0, p1)
                cuentas[c] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cuentas > 0, imp / np.maximum(cuentas, 1), 0.0)


def importancia_latente(agent, dec, n=120):
    """(2) importancia de cada dimension interna (anular y medir)."""
    D = agent.hidden_dim
    imp = np.zeros(D)
    for d in dec[:n]:
        h, p0 = d["h"], d["p"]
        for dim in range(D):
            h2 = h.clone()
            h2[0, dim] = 0.0
            p1 = distribucion(agent, d["tab"], h2)
            imp[dim] += tv(p0, p1)
    return imp / max(1, n)


def fidelidad_vs_k(agent, dec, imp, ks=(1, 3, 10, 30, 100, 300)):
    """(2b) con las k dims mas importantes, ¿cambia la decision?"""
    orden = np.argsort(-imp)
    out = {}
    for k in ks:
        sel = torch.zeros_like(dec[0]["h"])
        sel[0, orden[:k]] = 1.0
        mismos = 0
        for d in dec:
            h2 = d["h"] * sel
            p1 = distribucion(agent, d["tab"], h2)
            mismos += int(np.argmax(p1) == d["a"])
        out[k] = mismos / max(1, len(dec))
    return out


def intercambio(agent, dec, n=120, rng=None):
    """(3) se le da el estado interno de otra posicion: ¿que decision toma?"""
    rng = rng or np.random.default_rng(0)
    sigue_otro = sigue_propio = otro = 0
    for i, d in enumerate(dec[:n]):
        j = int(rng.integers(0, len(dec)))
        if j == i:
            continue
        p1 = distribucion(agent, d["tab"], dec[j]["h"])
        a1 = int(np.argmax(p1))
        if a1 == d["a"]:
            sigue_propio += 1
        elif a1 == dec[j]["a"]:
            sigue_otro += 1
        else:
            otro += 1
    tot = max(1, sigue_otro + sigue_propio + otro)
    return {"sigue_propio": sigue_propio / tot, "sigue_al_otro": sigue_otro / tot,
            "mezcla": otro / tot}


def contrafactual(agent, dec, n=120):
    """(4) el cambio mas pequeno del tablero que le hace cambiar la jugada."""
    con_contra = 0
    ejemplos = []
    for d in dec[:n]:
        tab, h, p0, a0 = d["tab"], d["h"], d["p"], d["a"]
        vacias = [i for i in range(9) if tab[i] == VACIO]
        encontrado = None
        for c in range(9):
            if tab[c] == VACIO:
                continue
            for e in vacias:
                ntab = list(tab)
                ntab[c], ntab[e] = ntab[e], ntab[c]
                p1 = distribucion(agent, tuple(ntab), h)
                if int(np.argmax(p1)) != a0:
                    encontrado = (c, e, int(np.argmax(p1)))
                    break
            if encontrado:
                break
        if encontrado:
            con_contra += 1
            if len(ejemplos) < 5:
                ejemplos.append({"tablero": list(tab), "mover": encontrado[0],
                                 "a": encontrado[1], "jugada_original": a0,
                                 "jugada_nueva": encontrado[2]})
    return {"frac_con_contrafactual": con_contra / max(1, n), "ejemplos": ejemplos}


def sonda(X, y, grupos=None, frac_train=0.7, ridge=1e-3):
    """Sonda lineal: X (N,D) -> y (N,9) one-hot. Devuelve (acc, R2).

    Si se pasan `grupos` (id de partida), el split es POR PARTIDA: ninguna
    partida aparece en train y test a la vez."""
    N = len(X)
    if grupos is not None:
        ug = sorted(set(grupos))
        n_g = max(1, int(round(frac_train * len(ug))))
        gtrain = set(ug[:n_g])
        idx_tr = [i for i in range(N) if grupos[i] in gtrain]
        idx_te = [i for i in range(N) if grupos[i] not in gtrain]
        if not idx_te:            # pocas partidas: al menos una para test
            idx_te = idx_tr[-max(1, len(idx_tr) // 5):]
            idx_tr = idx_tr[:len(idx_tr) - len(idx_te)]
    else:
        ntr = int(N * frac_train)
        idx_tr, idx_te = list(range(ntr)), list(range(ntr, N))
    Xtr, ytr = X[idx_tr], y[idx_tr]
    Xte, yte = X[idx_te], y[idx_te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
    A = Xtr_s.T @ Xtr_s + ridge * np.eye(Xtr_s.shape[1])
    W = np.linalg.solve(A, Xtr_s.T @ ytr)
    pred = Xte_s @ W
    acc = float((pred.argmax(1) == yte.argmax(1)).mean())
    r2 = float(1 - ((yte - pred) ** 2).sum() / max(1e-9, ((yte - yte.mean(0)) ** 2).sum()))
    return acc, r2


def legibilidad(agent, dec, imp, ks=(1, 3, 10, 30, 100, 300)):
    """(5) cuantos numeros internos bastan para predecir la jugada."""
    H = np.stack([d["h"].numpy().reshape(-1) for d in dec])
    y = np.zeros((len(dec), 9))
    for i, d in enumerate(dec):
        y[i, d["a"]] = 1.0
    tab = np.array([d["tab"] for d in dec], dtype=float)
    orden = np.argsort(-imp)
    grupos = [d.get("partida", 0) for d in dec]
    out = {"tablero": sonda(tab, y, grupos)}
    for k in ks:
        if k > H.shape[1]:
            continue
        out[str(k)] = sonda(H[:, orden[:k]], y, grupos)
    return out


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
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--decisions", type=int, default=400)
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--configs", default="A,B,C,D")
    ap.add_argument("--out", default=str(ROOT / "data" / "legibilidad.json"))
    ap.add_argument("--fig", default=str(ROOT / "paper" / "fig4_legibilidad.pdf"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    mentor = StochasticMinimax(depth=6, noise_prob=0.0)
    out = {"args": vars(args), "resultados": {}}
    mapas = {}

    for cfg in CONFIGS:
        if cfg["id"] not in args.configs.split(","):
            continue
        path = pathlib.Path(args.ckpt_dir) / f"ablation_{cfg['id']}.pth"
        if not path.exists():
            print(f"  {cfg['id']}: falta checkpoint")
            continue
        agent = cargar(cfg, path)
        dec = recolectar(agent, mentor, args.games, args.decisions)
        if len(dec) < 20:
            print(f"  {cfg['id']}: muy pocas decisiones ({len(dec)})")
            continue
        mapa = mapa_tablero(agent, dec, n=100)
        imp = importancia_latente(agent, dec, n=60)
        fid = fidelidad_vs_k(agent, dec, imp)
        sw = intercambio(agent, dec, n=100)
        cf = contrafactual(agent, dec, n=100)
        leg = legibilidad(agent, dec, imp)
        res = {
            "n_decisiones": len(dec),
            "mapa_tablero": mapa.tolist(),
            "celda_mas_importante": int(np.argmax(mapa)),
            "importancia_latente_top10": np.argsort(-imp)[:10].tolist(),
            "fidelidad_vs_k": {str(k): v for k, v in fid.items()},
            "intercambio": sw,
            "contrafactual": {k: v for k, v in cf.items() if k != "ejemplos"},
            "legibilidad": {k: {"acc": v[0], "r2": v[1]} for k, v in leg.items()},
        }
        out["resultados"][cfg["id"]] = {"label": cfg["label"], **res}
        mapas[cfg["id"]] = mapa
        print(f"  {cfg['id']} ({cfg['label']}):")
        print(f"     mapa del tablero: {np.round(mapa.reshape(3,3), 3).tolist()}")
        print(f"     fidelidad con las k dims mas importantes: "
              f"{ {k: round(v,2) for k,v in fid.items()} }")
        print(f"     intercambio de estado: sigue al propio {sw['sigue_propio']:.0%} | "
              f"sigue al prestado {sw['sigue_al_otro']:.0%} | mezcla {sw['mezcla']:.0%}")
        print(f"     contrafactual minimo (mover 1 pieza): "
              f"{cf['frac_con_contrafactual']:.0%} de las decisiones cambian")
        acc_t = leg['tablero'][0]
        acc_10 = leg['10'][0] if '10' in leg else float('nan')
        acc_all = (leg[str(agent.hidden_dim)][0]
                    if str(agent.hidden_dim) in leg else float('nan'))
        print(f"     legibilidad (acc de la sonda): tablero {acc_t:.2f} | "
              f"10 dims {acc_10:.2f} | todas {acc_all:.2f}", flush=True)

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nCrudo: {p}")

    # figura: mapas de atención al tablero
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(mapas)
        fig, axes = plt.subplots(1, max(1, n), figsize=(3.1 * max(1, n), 3.3))
        axes = np.atleast_1d(axes)
        for ax, (cid, m) in zip(axes, mapas.items()):
            ax.imshow(m.reshape(3, 3), cmap="viridis")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{m[3*i+j]:.2f}", ha="center", va="center",
                            color="w", fontsize=9)
            ax.set_title(f"config {cid}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Where the agent looks: influence of each cell on the decision", fontsize=11)
        fig.tight_layout()
        fig.savefig(args.fig)
        print(f"Figura: {args.fig}")
    except Exception as exc:  # noqa: BLE001
        print(f"(sin figura: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
