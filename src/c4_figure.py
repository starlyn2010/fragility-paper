#!/usr/bin/env python3
"""c4_figure.py — CGR trajectories for Connect Three and the scaling panels.

By default reads data/c4_canon_s*.json (Connect Three, three seeds) and plots
the exact uniform optimal-teacher draw rate after each round for the CGR arm
and the matched random controls. --files overrides the inputs, e.g. to place
the 4x4 and 7x6 endgame runs side by side.

Usage:
  python src/c4_figure.py --out figs_v2/fig7_c4.pdf
  python src/c4_figure.py --files "data/c4x4_ctrl_s0.json:4x4,data/c4_end_ctrl_s0.json:7x6"       --panel-width 2.5 --out figs_v2/fig8_scaling.pdf

A panel's path may name more than one file, joined by ``+'': the first is the
arm and the rest only contribute their ``controls'' entries, which is how the
deeper suites get their matched controls --- they are run separately, with
--controls-from, because the laptop cannot fit them in one process.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--prefix", default="c4_canon")
    ap.add_argument("--files", default="",
                    help="comma-separated path:label pairs; overrides --prefix")
    ap.add_argument("--out", default="figs_v2/fig7_c4.pdf")
    ap.add_argument("--panel-width", type=float, default=2.45)
    args = ap.parse_args()

    runs = []
    if args.files:
        for spec in [x for x in args.files.split(",") if x.strip()]:
            path, label = spec.rsplit(":", 1)
            parts = path.split("+")
            d = json.loads((ROOT / parts[0]).read_text(encoding="utf-8"))
            for extra in parts[1:]:
                d["controls"] = (d.get("controls") or []) + json.loads(
                    (ROOT / extra).read_text(encoding="utf-8"))["controls"]
            runs.append((label, d))
    else:
        for s in [x.strip() for x in args.seeds.split(",") if x.strip()]:
            p = ROOT / "data" / f"{args.prefix}_s{s}.json"
            runs.append((f"seed {s}", json.loads(p.read_text(encoding="utf-8"))))

    fig, axes = plt.subplots(1, len(runs), figsize=(args.panel_width * len(runs), 2.3),
                             squeeze=False, sharey=True)
    axes = axes.ravel()

    for ax, (label, d) in zip(axes, runs):
        log = d["repair"]["log"]
        xs = list(range(len(log)))
        ys = [100 * e["random_draw_rate_exact"] for e in log]
        ax.plot(xs, ys, "-o", color=BLUE, lw=1.8, ms=3.5, zorder=5,
                label="CGR (certificate states)")
        ax.text(0.97, 0.06, f"{len(d['repair']['aggregate'])} added states",
                transform=ax.transAxes, fontsize=7, color=BLUE,
                ha="right", va="bottom")

        random_controls = [c for c in d["controls"]
                           if c["pool"] in ("all", "tree")]
        for j, c in enumerate(random_controls):
            cxs = [e["round"] for e in c["log"]]
            cys = [100 * e["random_draw_rate_exact"] for e in c["log"]]
            cxs = [v + (j - 2.5) * 0.012 for v in cxs]
            style = (dict(color=ORANGE, lw=1.0, ls="--", alpha=0.85)
                     if c["pool"] == "tree"
                     else dict(color=GRAY, lw=1.0, ls=":", alpha=0.95))
            ax.plot(cxs, cys, marker=".", ms=2.5, **style)

        ax.axhline(100.0, color=GREEN, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_title(label, pad=3)
        ax.set_xlabel("repair round")
        ax.set_ylim(-6, 108)
        last = xs[-1]
        if random_controls:
            last = max(last, max(len(c["log"]) - 1 for c in random_controls))
        step = 5 if last > 12 else 1
        ticks = list(range(0, last + 1, step))
        if ticks[-1] != last:
            ticks.append(last)
        ax.set_xticks(ticks)
        ax.set_yticks([0, 25, 50, 75, 100])
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)

    axes[0].set_ylabel("draw rate vs. uniform\noptimal teacher (\\%)")
    handles = [
        Line2D([], [], color=BLUE, lw=1.8, marker="o", ms=3.5,
               label="CGR: certificate-selected states"),
        Line2D([], [], color=ORANGE, lw=1.0, ls="--", marker=".", ms=3,
               label="random, optimal-play tree"),
        Line2D([], [], color=GRAY, lw=0.9, ls=":", marker=".", ms=3,
               label="random, constrained histories"),
        Line2D([], [], color=GREEN, lw=0.9, ls=(0, (4, 3)), label="certified ($W=1$)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = ROOT / "paper" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
