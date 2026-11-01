#!/usr/bin/env python3
"""eps_figure.py — what an approximate certificate can and cannot promise.

Panel A: how far the tolerance is from opening the teacher's action class, measured
as a multiple of the smallest payoff gap Delta. The proposition's safe regime is
epsilon < Delta/(2H); the class does not move until epsilon reaches Delta.
Panel B: what repair costs as a function of that same multiple, and whether it closes.
Panel C: the deficit the certificate measures as a fraction of the deficit the
additive bound 2*epsilon*H predicts, i.e. how much of the allowed slack is used.

Reads only data/c4_eps_int_s*.json (integer payoffs, per-board value error) and
data/c4_eps_grad_s*.json (graded payoffs, critic blended toward random play).

Usage:
  python src/eps_figure.py --out figs_v2/fig10_eps.pdf
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parent.parent
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#1f4e79"
ORANGE = "#c55a11"
GREEN = "#2e7d32"
RED = "#a4262c"
GRAY = "#6f6f6f"


def load(prefix: str, seeds: str):
    out = []
    for s in [x for x in seeds.split(",") if x.strip()]:
        p = ROOT / "data" / f"{prefix}_s{s}.json"
        if p.exists():
            out.append((int(s), json.loads(p.read_text(encoding="utf-8"))))
    if not out:
        raise SystemExit(f"no files match data/{prefix}_s*.json")
    return out


def curve(doc: dict, key: str):
    """(epsilon in units of Delta, value) for one arm, sorted by the dial.

    ``key`` may be dotted, e.g. ``teacher_class.widened_states``. The epsilon = 0
    row is dropped so the abscissa can be read on a log scale.
    """
    delta = doc["game"]["min_gap"]

    def get(row):
        v = row
        for part in key.split("."):
            v = v[part]
        return v

    pts = [(r["epsilon"] / delta, get(r)) for r in doc["rows"] if r["epsilon"] > 0]
    return zip(*sorted(pts))


def logx(ax):
    ax.set_xscale("log")
    ax.set_xlim(0.02, 9.0)
    ax.set_xticks([0.05, 0.1, 0.5, 1.0, 5.0])
    ax.set_xticklabels(["0.05", "0.1", "0.5", "1", "5"])
    ax.minorticks_off()


def thresholds(ax, safe: float):
    ax.axvline(safe, color=GRAY, lw=0.9, ls=(0, (3, 2)))
    ax.text(safe * 1.25, 0.055, r"proof needs $\epsilon<\frac{\Delta}{2H}$",
            color=GRAY, fontsize=5.5, va="center", rotation=90)
    ax.axvline(1.0, color=RED, lw=0.9, ls=":")
    ax.text(1.13, 0.055, r"class opens at $\epsilon=\Delta$", color=RED,
            fontsize=5.5, va="center", rotation=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--int-prefix", default="c4_eps_int")
    ap.add_argument("--grad-prefix", default="c4_eps_grad")
    ap.add_argument("--out", default="figs_v2/fig10_eps.pdf")
    args = ap.parse_args()

    ints = load(args.int_prefix, args.seeds)
    grads = load(args.grad_prefix, args.seeds)
    horizon = ints[0][1]["game"]["horizon"]
    safe = 1.0 / (2.0 * horizon)
    n_teacher = ints[0][1]["rows"][0]["teacher_class"]["teacher_states"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    # A: the class the certificate hands the teacher.
    ax = axes[0]
    for docs, color, mark, name in ((ints, BLUE, "o", "integer payoffs"),
                                    (grads, ORANGE, "s", "graded payoffs")):
        seed, doc = docs[0]
        xs, ys = curve(doc, "teacher_class.widened_states")
        ax.plot(xs, [w / n_teacher for w in ys], color=color, lw=1.5, marker=mark,
                ms=3.2, label=f"{name} ($\\Delta={doc['game']['min_gap']:.3g}$)")
    ax.set_xlabel(r"tolerance $\epsilon$, in units of $\Delta$")
    ax.set_ylabel(f"fraction of the {n_teacher} teacher states widened")
    ax.set_ylim(0, 0.35)
    logx(ax)
    thresholds(ax, safe)
    ax.legend(frameon=False, loc="upper left")

    # B: what repair costs, and whether it closes.
    ax = axes[1]
    for tag, docs, color, dx in (("integer", ints, BLUE, 0.97), ("graded", grads, ORANGE, 1.03)):
        for i, (seed, doc) in enumerate(docs):
            xs, added = curve(doc, "added")
            added = [max(a, 0.8) for a in added]  # log axis: zero histories sit on the floor
            xs, ok = curve(doc, "W_ref_exact")
            label = f"{tag} repair" if seed == min(s for s, _ in docs) else None
            ax.plot([x * dx for x in xs], added, color=color, lw=1.4 if label else 0.7,
                    alpha=1.0 if label else 0.5, label=label)
            for x, a, g in zip(xs, added, ok):
                ax.plot([x * dx], [a], marker="o", ms=3.4,
                        color=GREEN if g == 1 else RED,
                        markerfacecolor="none" if g == 1 else color,
                        lw=1.0, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel(r"tolerance $\epsilon$, in units of $\Delta$")
    ax.set_ylabel("histories added (log)")
    ax.set_ylim(0.7, 400)
    logx(ax)
    ax.legend(frameon=False, loc="upper left")

    # C: how far the observed deficit sits below the predicted one.
    ax = axes[2]
    for docs, color, mark, name in ((ints, BLUE, "o", "integer"), (grads, ORANGE, "s", "graded")):
        for i, (seed, doc) in enumerate(docs):
            xs, worst = curve(doc, "worst_value")
            eps = [x * doc["game"]["min_gap"] for x in xs]
            ratio = [(2.0 * e * horizon) / abs(w) if w < 0 else math.nan
                     for w, e in zip(worst, eps)]
            first = seed == min(s for s, _ in docs)
            ax.plot(xs, ratio, color=color, lw=1.4 if first else 0.7,
                    alpha=1.0 if first else 0.5, marker=mark if first else None, ms=3.2,
                    label=f"bound / measured, {name}" if first else None)
    ax.set_yscale("log")
    ax.set_xlabel(r"tolerance $\epsilon$, in units of $\Delta$")
    ax.set_ylabel(r"$2\epsilon H$ / measured deficit")
    ax.set_ylim(15, 60)
    ax.set_yticks([20, 50])
    ax.set_yticklabels(["$20\\times$", "$50\\times$"])
    ax.minorticks_off()
    logx(ax)
    ax.set_xlim(0.02, 11.0)
    ax.legend(frameon=False, loc="upper left")

    handles = [
        Line2D([], [], marker="o", ms=4, color=GREEN, markerfacecolor="none", lw=0,
               label="repair certified ($W=1$)"),
        Line2D([], [], marker="o", ms=4, color=RED, lw=0, label="repair not certified"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ROOT / "paper" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"wrote {out}  (integer seeds {[s for s,_ in ints]}, graded seeds {[s for s,_ in grads]})")


if __name__ == "__main__":
    main()
