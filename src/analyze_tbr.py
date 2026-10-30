#!/usr/bin/env python3
"""analyze_tbr.py — Agrega data/tbr_eval.json y produce tablas con incertidumbre.

Unidades de replicacion:
  - semilla de entrenamiento (n=3-5)
  - mentor gemelo (n=3, dentro de cada checkpoint)
  - partida (n=200-400, dentro de cada mentor)

Reporta:
  - DR contra M6 fijo y contra gemelos (media, rango, bootstrap de dos niveles)
  - twin_gap = DR(gemelos) - DR(M6)
  - regret y tasa de accion optima por nivel
  - diferencias pareadas TBR - fijo por semilla + test exacto de signos
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def two_level_bootstrap(opponents: dict, twin_names: list[str], n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = {"dr_twin": [], "regret_twin": [], "opt_twin": [], "dr_m6": [], "regret_m6": []}
    m6 = opponents["m6_fixed"]["per_game"]
    for _ in range(n_boot):
        twin_pick = rng.choice(len(twin_names), len(twin_names), replace=True)
        drs, regs, opts = [], [], []
        for j in twin_pick:
            pg = opponents[twin_names[j]]["per_game"]
            idx = rng.integers(0, len(pg["outcome"]), len(pg["outcome"]))
            outs = np.asarray(pg["outcome"])[idx]
            reg = np.asarray(pg["regret"], dtype=float)[idx]
            dec = np.asarray(pg["decisions"], dtype=float)[idx]
            opt = np.asarray(pg["optimal"], dtype=float)[idx]
            drs.append(float(np.mean(outs == 0)))
            regs.append(float(reg.sum() / max(dec.sum(), 1)))
            opts.append(float(opt.sum() / max(dec.sum(), 1)))
        out["dr_twin"].append(float(np.mean(drs)))
        out["regret_twin"].append(float(np.mean(regs)))
        out["opt_twin"].append(float(np.mean(opts)))
        idx = rng.integers(0, len(m6["outcome"]), len(m6["outcome"]))
        out["dr_m6"].append(float(np.mean(np.asarray(m6["outcome"])[idx] == 0)))
        reg = np.asarray(m6["regret"], dtype=float)[idx]
        dec = np.asarray(m6["decisions"], dtype=float)[idx]
        out["regret_m6"].append(float(reg.sum() / max(dec.sum(), 1)))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def exact_signflip_p(deltas: list[float]) -> float:
    d = [x for x in deltas if abs(x) > 1e-12]
    if not d:
        return float("nan")
    obs = abs(float(np.mean(d)))
    hits = 0
    total = 1 << len(d)
    for signs in itertools.product([-1, 1], repeat=len(d)):
        m = float(np.mean([s * x for s, x in zip(signs, d)]))
        if abs(m) >= obs - 1e-12:
            hits += 1
    return hits / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(ROOT / "data" / "tbr_eval.json"),
                    help="one JSON file or a comma-separated list (e.g. B and C evals)")
    ap.add_argument("--out", default=str(ROOT / "data" / "tbr_analysis.json"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    inputs = [pathlib.Path(p.strip()) for p in args.inp.split(",") if p.strip()]
    merged = {"results": {}, "twin_seeds": None}
    for p in inputs:
        d = json.loads(p.read_text(encoding="utf-8"))
        if merged["twin_seeds"] is None:
            merged["twin_seeds"] = d["twin_seeds"]
        elif d["twin_seeds"] != merged["twin_seeds"]:
            raise SystemExit(f"twin seeds differ in {p}")
        dup = set(merged["results"]) & set(d["results"])
        if dup:
            raise SystemExit(f"duplicate labels {sorted(dup)} in {p}")
        merged["results"].update(d["results"])
    data = merged
    twin_names = [f"twin_{s}" for s in data["twin_seeds"]]

    parsed = []
    for label, row in data["results"].items():
        m = re.match(r"^(?P<cfg>[A-D])_(?P<cond>[a-z]+)_s(?P<seed>\d+)$", label)
        if not m:
            print(f"[warn] etiqueta no reconocida: {label}")
            continue
        parsed.append((m.group("cfg"), m.group("cond"), int(m.group("seed")), label, row))

    groups: dict[str, dict[str, list]] = {}
    for cfg, cond, seed, label, row in parsed:
        r = {
            "label": label,
            "seed": seed,
            "dr_m6": row["summary"]["dr_m6_fixed"],
            "dr_twin": row["summary"]["dr_twin_mean"],
            "twin_gap": row["summary"]["twin_gap_mean"],
            "regret_m6": row["summary"]["regret_m6_fixed"],
            "regret_twin": row["summary"]["regret_twin_mean"],
            "regret_gap": row["summary"]["regret_gap_twin"],
            "opt_rate_m6": row["summary"]["opt_rate_m6_fixed"],
            "opt_rate_twin": row["summary"]["opt_rate_twin_mean"],
            "regret_tie_twin": row["summary"]["regret_tie_twin_mean"],
            "regret_notie_twin": row["summary"]["regret_notie_twin_mean"],
            "tie_share_twin": row["summary"]["tie_share_twin_mean"],
            "wr_random": row["summary"]["wr_random_uniform"],
            "wr_weak": row["summary"]["wr_weak_minimax"],
            "boot": two_level_bootstrap(row["opponents"], twin_names, args.boot, seed=seed + 11),
        }
        groups.setdefault(cfg, {}).setdefault(cond, []).append(r)

    out = {"input": args.inp, "groups": {}, "paired": {}}
    for cfg, conds in groups.items():
        out["groups"][cfg] = {}
        for cond, rows in conds.items():
            keys = ["dr_m6", "dr_twin", "twin_gap", "regret_m6", "regret_twin", "regret_gap",
                    "opt_rate_m6", "opt_rate_twin", "regret_tie_twin", "regret_notie_twin",
                    "tie_share_twin", "wr_random", "wr_weak"]
            agg = {}
            for k in keys:
                vals = [r[k] for r in rows]
                agg[k] = {
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "values": vals,
                }
            agg["n_seeds"] = len(rows)
            agg["labels"] = [r["label"] for r in rows]
            out["groups"][cfg][cond] = agg
        if "fixed" in conds and "tbr" in conds:
            fix = {r["seed"]: r for r in conds["fixed"]}
            tbr = {r["seed"]: r for r in conds["tbr"]}
            seeds = sorted(set(fix) & set(tbr))
            paired = {}
            for k in ["dr_twin", "twin_gap", "regret_twin", "regret_gap", "opt_rate_twin"]:
                deltas = [tbr[s][k] - fix[s][k] for s in seeds]
                paired[k] = {
                    "seeds": seeds,
                    "deltas_tbr_minus_fixed": deltas,
                    "mean_delta": float(np.mean(deltas)),
                    "min_delta": float(np.min(deltas)),
                    "max_delta": float(np.max(deltas)),
                    "signflip_p_two_sided": exact_signflip_p(deltas),
                }
            out["paired"][cfg] = paired

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n{'cfg':3s} {'cond':7s} {'seeds':5s} {'DR(M6)':>8s} {'DR(twin)':>9s} "
          f"{'gap':>7s} {'regret_M6':>10s} {'regret_twin':>12s} {'opt_twin':>9s}")
    for cfg, conds in out["groups"].items():
        for cond, a in conds.items():
            print(f"{cfg:3s} {cond:7s} {a['n_seeds']:<5d} "
                  f"{a['dr_m6']['mean']*100:7.1f}% {a['dr_twin']['mean']*100:8.1f}% "
                  f"{a['twin_gap']['mean']*100:+6.1f}p {a['regret_m6']['mean']:10.4f} "
                  f"{a['regret_twin']['mean']:12.4f} {a['opt_rate_twin']['mean']:8.3f}")
    for cfg, paired in out["paired"].items():
        print(f"\nPareado {cfg} (TBR - fijo):")
        for k, v in paired.items():
            print(f"  {k:14s} delta {v['mean_delta']:+.4f} "
                  f"[{v['min_delta']:+.4f},{v['max_delta']:+.4f}] "
                  f"signflip p={v['signflip_p_two_sided']:.4f} seeds={v['seeds']}")
    print(f"\nCrudo: {p}")


if __name__ == "__main__":
    main()
