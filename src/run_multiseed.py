"""
run_multiseed.py — ¿la fragilidad es de la ARQUITECTURA o de la SEMILLA?

Al reentrenar el pipeline del paper en otra maquina, el config A (CfC+ACT,
"el robusto": 96% vs random en el paper) tambien colapso (13% vs random),
mientras que C (GRU, "el fragil": 11% en el paper) reproduces el colapso
(13%). Eso sugiere que la diferencia entre "robusto" y "fragil" del paper
puede ser ruido de semilla, no arquitectura.

Este script entrena las configs indicadas con N semillas distintas (cada
semilla en su propia carpeta, para no pisar los resultados del paper) y
agrega la distribucion de "win rate vs random" por config.

Uso:
  python src/run_multiseed.py --seeds 42,43,44,45,46 --configs A,C --games 2500
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RUNS = ROOT / "runs_multiseed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--configs", default="A,C")
    ap.add_argument("--games", type=int, default=2500)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=str(ROOT / "data" / "multiseed_summary.json"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = []
    for seed in seeds:
        d = RUNS / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        res = d / "ablation_results.json"
        if res.exists():
            print(f"[multiseed] seed {seed}: ya existe, salto", flush=True)
        else:
            cmd = [args.python, str(SRC / "ablation_ttt.py"), "--all",
                   "--seed", str(seed), "--games", str(args.games),
                   "--configs", args.configs]
            print(f"[multiseed] seed {seed}: {' '.join(cmd)}", flush=True)
            t0 = time.time()
            with open(d / "train.log", "w", encoding="utf-8") as f:
                rc = subprocess.run(cmd, cwd=str(d), stdout=f,
                                    stderr=subprocess.STDOUT).returncode
            print(f"[multiseed] seed {seed}: rc {rc} | {(time.time()-t0)/60:.1f} min",
                  flush=True)
        try:
            for r in json.loads(res.read_text(encoding="utf-8")):
                todo.append({"seed": seed, "config": r["config"], "label": r["label"],
                             "time": r["time_str"],
                             "wr_random": r["metrics"]["wr_vs_random"],
                             "dr_random": r["metrics"]["dr_vs_random"],
                             "wr_m6": r["metrics"]["wr_vs_m6"],
                             "dr_m6": r["metrics"]["dr_vs_m6"]})
        except Exception as exc:  # noqa: BLE001
            print(f"[multiseed] seed {seed}: no se pudo leer ({exc})", flush=True)

    print(f"\n{'config':<6}{'seed':>6}{'wr vs random':>15}{'dr vs random':>15}{'wr vs M6':>10}")
    for r in sorted(todo, key=lambda x: (x["config"], x["seed"])):
        print(f"{r['config']:<6}{r['seed']:>6}{r['wr_random']*100:>14.1f}%"
              f"{r['dr_random']*100:>14.1f}%{r['wr_m6']*100:>9.1f}%")
    print("\npor config (WR vs random):")
    resumen = {}
    for cfg in sorted({r["config"] for r in todo}):
        v = np.array([r["wr_random"] for r in todo if r["config"] == cfg]) * 100
        d = np.array([r["dr_random"] for r in todo if r["config"] == cfg]) * 100
        resumen[cfg] = {"n": len(v), "wr_random_mean": float(v.mean()),
                        "wr_random_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                        "wr_random_min": float(v.min()), "wr_random_max": float(v.max()),
                        "dr_random_mean": float(d.mean()), "vals": v.tolist()}
        print(f"  {cfg}: n={len(v)}  WR vs random {v.mean():.1f}% ± {resumen[cfg]['wr_random_std']:.1f} "
              f"(min {v.min():.1f}%, max {v.max():.1f}%)  | DR {d.mean():.1f}%")
    if len(resumen) == 2:
        a, b = list(resumen.values())
        solapan = not (a["wr_random_max"] < b["wr_random_min"] or
                       b["wr_random_max"] < a["wr_random_min"])
        print(f"\n  las distribuciones de las dos configs "
              f"{'SE SOLAPAN' if solapan else 'NO se solapan'} "
              f"-> la diferencia del paper {'puede ser ruido de semilla' if solapan else 'es real'}")
        resumen["solapan"] = bool(solapan)
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"runs": todo, "resumen": resumen}, indent=2), encoding="utf-8")
    print(f"\nCrudo: {p}")


if __name__ == "__main__":
    main()
