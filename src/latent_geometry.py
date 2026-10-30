"""
latent_geometry.py — ¿existe de verdad el "mapa hash"?

El paper (seccion "Mechanistic Interpretation") AFIRMA que entrenar contra un
adversario determinista y optimo produce un "latent hash map": una tabla hash
que responde bien solo en los estados alcanzables bajo ese adversario
(eq. hash_map) y se comporta caoticamente fuera. Esa prediccion es medible y
hasta ahora no se midio. Este script la mide.

Para cada config de la ablation (A: CfC+ACT, B: CfC fix3, C: GRU, D: CfC-32)
juega contra DOS oponentes y compara:

 1. ENTROPIA de la politica H[pi(a|s)] (nats, solo movimientos legales).
    Prediccion del mapa hash: ~0 vs M6 (en distribucion) y alta vs Random
    (fuera de distribucion).
 2. DIMENSION EFECTIVA del estado latente: participation ratio de la
    covarianza de h y exp(H[lambda/lambda_sum]) (la "geometria" que el paper
    llama latente rasante).
 3. COLISIONES DE HASH: cuantos tableros DISTINTOS caen en el mismo cluster
    del estado latente (k-means). La firma del mapa hash: el latente no
    discrimina el tablero -> H[tablero | cluster] alto.
 4. COBERTURA: |V_M6 ∩ V_R| / |V_R|, o sea, que fraccion de los estados que la
    politica ve fuera de distribucion (vs Random) ya habia visto vs M6. Cuanto
    mas bajo, mas "fino" es el submanifold que aprendio.
 5. ACUERDO CON LA JUGADA OPTIMA: fraccion de decisiones en que la politica
    elige el movimiento minimax (referencia: minimax de profundidad 6).
    En distribucion deberia ser alto; fuera, bajo.

Uso:
  python src/latent_geometry.py --games 300 --out data/latent_geometry.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_ttt import CONFIGS, AblationLiquidAgent  # noqa: E402
from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "ablation_logs"


# ----------------------------------------------------------------- utilidades
def kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 0):
    """k-means simple (numpy) sobre x (N, D). Devuelve (labels, centros)."""
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros((k, x.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    k = int(min(k, n))
    idx = rng.choice(n, size=k, replace=False)
    c = x[idx].copy()
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        for s in range(0, n, 256):            # por bloques: no explodes la RAM
            d = ((x[s:s + 256, None, :] - c[None, :, :]) ** 2).sum(-1)
            lab[s:s + 256] = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = x[m].mean(0)
    return lab, c


def dimension_efectiva(x: np.ndarray) -> tuple[float, float]:
    """(participation ratio, exp(entropia del espectro)) de la covarianza."""
    if len(x) < 3:
        return 0.0, 0.0
    xc = x - x.mean(0, keepdims=True)
    cov = (xc.T @ xc) / max(1, len(x) - 1)
    try:
        w = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 0:
        return 0.0, 0.0
    pr = float(s ** 2 / max(1e-12, (w ** 2).sum()))
    p = w / s
    p = p[p > 0]
    return pr, float(np.exp(-(p * np.log(p)).sum()))


def entropia(logits: torch.Tensor) -> float:
    """Entropia (nats) de una distribucion categorica sobre movimientos legales."""
    p = torch.softmax(logits, dim=-1)
    p = p[p > 0]
    return float(-(p * p.log()).sum())


# ----------------------------------------------------------------- experimento
def cargar_agente(cfg: dict, path: pathlib.Path):
    agent = AblationLiquidAgent(cfg["id"], 9, 9, cfg["hidden_dim"])
    agent.s12_enabled = cfg["act"]
    agent.s12_max_iters = 3
    sd = torch.load(str(path), map_location="cpu")
    agent.load_state_dict(sd)
    agent.eval()
    return agent


def jugar_y_medir(agent, opponent, n_games: int, ref_minimax, device="cpu"):
    """Juega n_games y devuelve las mediciones de la condicion."""
    env = TicTacToeEnv()
    tableros, hiddens, ents, acc_opt = [], [], [], []
    n_legales = 0
    for _ in range(n_games):
        hx = torch.zeros(1, agent.hidden_dim, device=device)
        obs = env.reset()
        done = False
        turn = 0
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            mask = torch.zeros(agent.action_dim, dtype=torch.bool)
            for a in legal:
                mask[a] = True
            if turn % 2 == 0:  # el agente
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).float()
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                with torch.no_grad():
                    if getattr(agent, "s12_enabled", False):
                        logits, _v, hx_new, _pen, _it = agent.forward_adaptive(obs_t, hx)
                    else:
                        logits, _v, hx_new = agent.forward_step(obs_t, hx)
                logits = logits.masked_fill(~mask, -float("inf"))
                hx = hx_new
                probs = torch.softmax(logits, dim=-1)
                ents.append(entropia(logits))
                accion = int(logits.argmax(-1).item())
                opt = ref_minimax.get_action(env, legal)
                acc_opt.append(1.0 if accion == opt else 0.0)
                tableros.append(tuple(int(x) for x in np.asarray(obs).reshape(-1)))
                hiddens.append(hx.detach().cpu().numpy().reshape(-1).copy())
                n_legales += len(legal)
                obs, _, done, _ = env.step(accion, player=env.AGENTE)
            else:  # oponente
                accion = opponent.get_action(env, legal)
                obs, _, done, _ = env.step(accion, player=env.HUMANO)
            turn += 1
            if done:
                break
    H = np.array(hiddens, dtype=np.float32) if hiddens else np.zeros((0, 1), np.float32)
    return {
        "n_estados": len(tableros),
        "tableros": set(tableros),
        "H": H,
        "entropia_media": float(np.mean(ents)) if ents else float("nan"),
        "frac_determinista": float(np.mean([e < 0.05 for e in ents])) if ents else float("nan"),
        "acuerdo_optimo": float(np.mean(acc_opt)) if acc_opt else float("nan"),
        "estados_legales_medios": n_legales / max(1, len(tableros)),
    }


def medir_config(cfg, path, n_games, k_clusters, ref_minimax):
    agent = cargar_agente(cfg, path)
    res = {"config": cfg["id"], "label": cfg["label"], "hidden_dim": cfg["hidden_dim"],
           "backbone": cfg["backbone"], "act": cfg["act"]}
    condiciones = {
        "m6": StochasticMinimax(depth=6, noise_prob=0.0),
        "random": StochasticMinimax(depth=0),
    }
    datos = {}
    for nombre, opp in condiciones.items():
        d = jugar_y_medir(agent, opp, n_games, ref_minimax)
        datos[nombre] = d
        res[nombre] = {k: v for k, v in d.items() if k not in ("tableros", "H")}
    # geometria y colisiones por condicion
    for nombre, d in datos.items():
        H = d["H"]
        pr, ed = dimension_efectiva(H)
        lab, _c = kmeans(H, k_clusters) if len(H) else (np.zeros(0, int), None)
        # cada cluster -> conjunto de tableros distintos que caen en el
        cl2b = {}
        for b, l in zip(d["tableros"], lab.tolist()):
            cl2b.setdefault(l, set()).add(b)
        boards_por_cluster = Counter({c: len(s) for c, s in cl2b.items()})
        if cl2b:
            medio = float(np.mean(list(boards_por_cluster.values())))
            maximo = int(max(boards_por_cluster.values()))
            puros = float(np.mean([1.0 if v == 1 else 0.0 for v in boards_por_cluster.values()]))
        else:
            medio, maximo, puros = float("nan"), 0, float("nan")
        res[nombre].update({
            "rank_efectivo": pr,
            "dim_efectiva_espectro": ed,
            "clusters_ocupados": len(cl2b),
            "tableros_por_cluster_medio": medio,
            "tableros_por_cluster_max": maximo,
            "frac_clusters_puros": puros,
        })
    # cobertura: de los estados vistos vs Random, cuantos se habian visto vs M6
    vm, vr = datos["m6"]["tableros"], datos["random"]["tableros"]
    res["cobertura"] = {
        "estados_m6": len(vm),
        "estados_random": len(vr),
        "interseccion": len(vm & vr),
        "frac_random_vistos_contra_m6": len(vm & vr) / max(1, len(vr)),
        "frac_m6_vistos_contra_random": len(vm & vr) / max(1, len(vm)),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="partidas por condicion")
    ap.add_argument("--clusters", type=int, default=64)
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--configs", default="A,B,C,D")
    ap.add_argument("--out", default=str(ROOT / "data" / "latent_geometry.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    ref = StochasticMinimax(depth=6, noise_prob=0.0)   # referencia "jugada optima"
    out = {"args": vars(args), "configs": {}}
    for cfg in CONFIGS:
        if cfg["id"] not in args.configs.split(","):
            continue
        path = pathlib.Path(args.ckpt_dir) / f"ablation_{cfg['id']}.pth"
        if not path.exists():
            print(f"  {cfg['id']}: falta {path.name}")
            continue
        r = medir_config(cfg, path, args.games, args.clusters, ref)
        out["configs"][cfg["id"]] = r
        print(f"  {cfg['id']} {cfg['label']}")
        for cond in ("m6", "random"):
            c = r[cond]
            print(f"     vs {cond:<7} H={c['entropia_media']:.3f} "
                  f"det={c['frac_determinista']:.0%} "
                  f"acuerdo_optimo={c['acuerdo_optimo']:.0%} "
                  f"dim_ef={c['dim_efectiva_espectro']:.1f} "
                  f"clusters={c['clusters_ocupados']} "
                  f"tableros/cluster={c['tableros_por_cluster_medio']:.1f} "
                  f"(max {c['tableros_por_cluster_max']})")
        cov = r["cobertura"]
        print(f"     cobertura: |V_m6|={cov['estados_m6']} |V_rand|={cov['estados_random']} "
              f"-> {cov['frac_random_vistos_contra_m6']:.0%} de los estados vs Random ya "
              f"se habian visto vs M6")
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nCrudo: {p}")


if __name__ == "__main__":
    main()
