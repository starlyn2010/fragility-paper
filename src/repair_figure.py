#!/usr/bin/env python3
"""repair_figure.py — certificate-guided repair trajectories.

Reads data/repair_*.json produced by src/repair.py and plots, per target, the
exact uniform-tie draw rate after each round of state aggregation.  The repair
arm adds the certificate's violating frontier states; the controls add the same
number of states sampled from all 180,361 histories ("all") or from the
optimal-play tree ("tree") without looking at violations.

Uso:
  python src/repair_figure.py --runs "C-42=data/repair_C42_s0.json" \
      --out figs_v2/fig6_repair.pdf
"""
from __future__ import annotations

import argparse
import json
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
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#1f4e79"
ORANGE = "#c55a11"
GREEN = "#2e7d32"
GRAY = "#6f6f6f"


def trajectory(log):
    # One point per completed fit; log[0] is the certificate before any fit.
    return list(range(len(log))), [e["random_draw_rate_exact"] for e in log]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="label=path;...")
    ap.add_argument("--out", default="figs_v2/fig6_repair.pdf")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--panel-width", type=float, default=3.45)
    args = ap.parse_args()

    runs = []
    for item in args.runs.split(";"):
        item = item.strip()
        if not item:
            continue
        label, path = item.split("=", 1)
        runs.append((label.strip(), json.loads(pathlib.Path(path).read_text(encoding="utf-8"))))

    ncols = min(args.cols, len(runs))
    nrows = (len(runs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(args.panel_width * ncols, 2.25 * nrows),
                             squeeze=False, sharey=True)
    axes = axes.ravel()

    for ax, (label, d) in zip(axes, runs):
        base_rate = d["base"]["certificate"]["random_draw_rate_exact"]
        log = d["repair"]["log"]
        xs, ys = trajectory(log)
        rounds = len(log) - 1
        added = len(d["repair"]["aggregate"])
        ax.plot(xs, ys, "-o", color=BLUE, lw=1.8, ms=3.5, zorder=5,
                label="CGR (certificate states)")
        ax.annotate(f"{added} states", xy=(xs[-1], ys[-1]),
                    xytext=(-5, -13), textcoords="offset points", fontsize=7,
                    color=BLUE, ha="right",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

        for j, c in enumerate(d["controls"]):
            cxs, cys = trajectory(c["log"])
            cxs = [v + (j - 2.5) * 0.015 for v in cxs]
            if c["pool"] == "tree":
                style = dict(color=ORANGE, lw=1.0, ls="--", alpha=0.85)
            else:
                style = dict(color=GRAY, lw=1.0, ls=":", alpha=0.95)
            ax.plot(cxs, cys, marker=".", ms=2.5, **style)

        ax.axhline(1.0, color=GREEN, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_title(label, pad=3)
        ax.set_xlabel("repair round")
        ax.set_ylim(-0.04, 1.08)
        ax.set_xticks(range(max(xs[-1], max(len(c["log"]) for c in d["controls"])) + 1))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25", "50", "75", "100"])
        if len(runs) > 1 and ax is not axes[0]:
            ax.tick_params(labelleft=False)

    for ax in axes[len(runs):]:
        ax.axis("off")
    axes[0].set_ylabel("draw rate vs. uniform\noptimal teacher (\\%)")

    handles = [
        Line2D([], [], color=BLUE, lw=1.8, marker="o", ms=3.5,
               label="CGR: certificate-selected states"),
        Line2D([], [], color=ORANGE, lw=1.0, ls="--", marker=".", ms=3,
               label="random, optimal-play tree (3 seeds)"),
        Line2D([], [], color=GRAY, lw=0.9, ls=":", marker=".", ms=3,
               label="random, all 180k histories (3 seeds)"),
        Line2D([], [], color=GREEN, lw=0.9, ls=(0, (4, 3)), label="certified ($W=1$)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = ROOT / "paper" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
