#!/usr/bin/env python3
"""rps_figure.py — hidden-type RPS: certificate curves and per-type payoffs.

Reads data/rps_s*.json (one file per seed) and writes
paper/figs_v2/fig9_rps.pdf with three panels:

  (a) exact normalized certificate vs. number of added histories
  (b) payoff against each hidden type, base vs. repaired
  (c) final certificate by method

Usage:
  python src/rps_figure.py --seeds 0,1,2 --out figs_v2/fig9_rps.pdf
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent.parent
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.6,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#1f4e79"
ORANGE = "#c55a11"
GREEN = "#2e7d32"
GRAY = "#6f6f6f"
PURPLE = "#7030a0"
RED = "#b02418"


def forward_fill(xs, ys, grid):
    out, j = [], 0
    for x in grid:
        while j + 1 < len(xs) and xs[j + 1] <= x:
            j += 1
        out.append(ys[j])
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="figs_v2/fig9_rps.pdf")
    args = ap.parse_args()

    runs = []
    for s in [x.strip() for x in args.seeds.split(",") if x.strip()]:
        p = ROOT / "data" / f"rps_s{s}.json"
        if p.exists():
            runs.append((int(s), json.loads(p.read_text(encoding="utf-8"))))
    if not runs:
        raise SystemExit("no data/rps_s*.json files found")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35),
                             gridspec_kw={"width_ratios": [1.55, 1.05, 1.0]})
    ax = axes[0]

    groups: dict[str, tuple[str, str]] = {
        "CGR": (BLUE, "-"),
        "control copy": (ORANGE, "--"),
        "control all": (GRAY, ":"),
        "control opt": (PURPLE, "-."),
        "DAgger all": (GREEN, "--"),
        "DAgger worst": (RED, ":"),
    }
    curves: dict[str, list[tuple[list[int], list[float]]]] = {k: [] for k in groups}

    for _, d in runs:
        log = d["repair"]["log"]
        curves["CGR"].append(([int(e["aggregate"]) for e in log],
                              [float(e["normalized"]) for e in log]))
        for c in d.get("controls", []):
            key = f"control {c['pool']}"
            if key in curves:
                curves[key].append(([int(e["added"]) for e in c["log"]],
                                    [float(e["normalized"]) for e in c["log"]]))
        for c in d.get("dagger", []):
            key = "DAgger all" if c["pool"].endswith("all") else "DAgger worst"
            if key in curves:
                curves[key].append(([int(e["added"]) for e in c["log"]],
                                    [float(e["normalized"]) for e in c["log"]]))

    all_x = [x for k in curves for xs, _ in curves[k] for x in xs]
    grid = np.arange(0, max(all_x) + 1)
    for key, (color, ls) in groups.items():
        runs_k = [forward_fill(xs, ys, grid) for xs, ys in curves[key] if xs]
        if not runs_k:
            continue
        mean = np.mean(runs_k, axis=0)
        lw = 1.8 if key == "CGR" else 1.0
        ax.plot(grid, mean, color=color, ls=ls, lw=lw,
                label=key if key == "CGR" else key.replace("control ", "rand. "))
        if key == "CGR":
            lo = np.min(runs_k, axis=0)
            hi = np.max(runs_k, axis=0)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.16, lw=0)
    ax.axhline(1.0, color=GREEN, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel("added histories")
    ax.set_ylabel("certificate $\\hat V$")
    ax.set_ylim(0.30, 1.04)
    ax.set_title("(a) exact certificate", pad=3)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2,
              borderaxespad=0.2)

    # ---- per-type payoffs ----
    ax = axes[1]
    types = runs[0][1]["game"]["types"]
    base = np.array([r["base"]["certificate"]["per_type"] for _, r in runs])
    rep = np.array([r["repair"]["final"]["per_type"] for _, r in runs])
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import rps_game as g  # noqa: E402
    o = g.Oracle(int(runs[0][1]["game"]["horizon"]))
    bayes = np.array(g.per_type(g.bayes_policy(o), o.horizon))
    x = np.arange(len(types))
    w = 0.38
    ax.bar(x - w / 2, base.mean(0), w, color=GRAY, alpha=0.85, label="base",
           yerr=base.std(0), error_kw={"lw": 0.5, "capsize": 1.5})
    ax.bar(x + w / 2, rep.mean(0), w, color=BLUE, alpha=0.9, label="repaired",
           yerr=rep.std(0), error_kw={"lw": 0.5, "capsize": 1.5})
    ax.plot(x, bayes, "k_", ms=8, mew=1.4, label="Bayes-optimal")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=32, ha="right")
    ax.set_ylabel(f"payoff over {o.horizon} rounds")
    ax.set_ylim(-2.5, 8.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("(b) against each hidden type", pad=3)
    ax.legend(loc="lower left", frameon=False, ncol=2, handlelength=1.2,
              columnspacing=0.8, borderaxespad=0.2)

    # ---- final certificates ----
    ax = axes[2]
    methods = [("CGR", [r["repair"]["final"]["normalized"] for _, r in runs], BLUE)]
    for pool, label, color in (("copy", "rand.\ncopy", ORANGE),
                               ("all", "rand.\nall", GRAY),
                               ("opt", "rand.\nopt", PURPLE)):
        v = [c["final"]["normalized"] for _, r in runs
             for c in r.get("controls", []) if c["pool"] == pool]
        if v:
            methods.append((label, v, color))
    for pool, label, color in (("dagger-all", "DAgger\nall", GREEN),
                               ("dagger-worst", "DAgger\nworst", RED)):
        v = [c["final"]["normalized"] for _, r in runs
             for c in r.get("dagger", []) if c["pool"] == pool]
        if v:
            methods.append((label, v, color))
    for i, (_, v, col) in enumerate(methods):
        ax.bar(i, statistics.mean(v), 0.7, color=col, alpha=0.9)
        ax.errorbar(i, statistics.mean(v),
                    yerr=statistics.stdev(v) if len(v) > 1 else 0,
                    color="k", lw=0.6, capsize=1.8)
        ax.scatter([i] * len(v), v, s=5, color="k", zorder=5)
    ax.axhline(1.0, color=GREEN, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m[0] for m in methods], rotation=32, ha="right")
    ax.set_ylim(0.30, 1.03)
    ax.set_ylabel("final $\\hat V$")
    ax.set_title("(c) same state budget", pad=3)

    fig.tight_layout(w_pad=1.0)
    out = ROOT / "paper" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
