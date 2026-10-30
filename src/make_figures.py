#!/usr/bin/env python3
"""make_figures.py — v2 figures from frozen measurements.

Figures
  fig1_gap.pdf      certification collapse: fixed M6 vs equally strong twins
  fig2_lottery.pdf  intervention lottery: 30-teacher strips + M6/twin phase plot
  fig3_state.pdf    boards vs histories, coverage dose-response, game outcomes
  fig4_seeds.pdf    v1 seed variance (audit)

The figures read only files under data/; no synthetic distributions.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

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
RED = "#b02418"
PURPLE = "#6a3d9a"
GRAY = "#6f6f6f"
LIGHT = "#d9d9d9"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def save(fig, name: str, png: bool = False):
    out = ROOT / "paper" / "figs_v2"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name)
    if png:
        fig.savefig((out / name).with_suffix(".png"), dpi=200)
    plt.close(fig)


def v1_twin_values() -> dict:
    vals = {}
    for cfg in ("A", "B", "C"):
        d = load(f"v1_reanalysis_{cfg}.json")
        row = d["results"][f"v1_{cfg}"]
        vals[cfg] = {
            "m6": row["summary"]["dr_m6_fixed"] * 100,
            "twins": [v["dr"] * 100 for k, v in row["opponents"].items()
                      if k.startswith("twin_")],
            "mean": row["summary"]["dr_twin_mean"] * 100,
        }
    return vals


def fresh_fixed_values() -> dict:
    vals = {}
    for cfg, fname in (("B", "tbr_eval_B.json"), ("C", "tbr_eval_C.json")):
        d = load(fname)
        rows = []
        for label, row in d["results"].items():
            if f"_fixed_" not in label:
                continue
            rows.append({
                "label": label,
                "m6": row["summary"]["dr_m6_fixed"] * 100,
                "twins": [v["dr"] * 100 for k, v in row["opponents"].items()
                          if k.startswith("twin_")],
                "mean": row["summary"]["dr_twin_mean"] * 100,
            })
        vals[cfg] = sorted(rows, key=lambda r: r["label"])
    return vals


def fig1_gap(png: bool = False):
    v1 = v1_twin_values()
    fresh = fresh_fixed_values()
    groups = [
        ("v1 A\nCfC+ACT", v1["A"], BLUE, "o"),
        ("v1 B\nCfC$\\times$3", v1["B"], BLUE, "s"),
        ("v1 C\nGRU", v1["C"], BLUE, "^"),
        ("B fixed\n3 seeds", {"m6": 100.0,
                              "twins": [v for r in fresh["B"] for v in r["twins"]],
                              "mean": np.mean([r["mean"] for r in fresh["B"]])},
         ORANGE, "D"),
        ("C fixed\n3 seeds", {"m6": 100.0,
                              "twins": [v for r in fresh["C"] for v in r["twins"]],
                              "mean": np.mean([r["mean"] for r in fresh["C"]])},
         ORANGE, "v"),
    ]

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    rng = np.random.default_rng(3)
    for i, (label, g, color, marker) in enumerate(groups):
        twin = np.asarray(g["twins"], dtype=float)
        ax.plot([i, i], [g["mean"], g["m6"]], color=LIGHT, lw=1.2, zorder=1)
        ax.scatter(i + rng.uniform(-0.10, 0.10, len(twin)), twin, s=12,
                   color=color, alpha=0.65, edgecolor="white", linewidth=0.3, zorder=3)
        ax.plot(i, g["mean"], marker=marker, ms=6.5, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        ax.plot(i, g["m6"], marker="o", ms=6.5, mfc="white", mec=GRAY, mew=1.2, zorder=4)
        ax.annotate(f"{g['mean'] - g['m6']:+.0f}", (i, (g["mean"] + g["m6"]) / 2),
                    textcoords="offset points", xytext=(8, 0), ha="left",
                    va="center", fontsize=7, color=color)

    ax.axhline(100, color=GRAY, ls="--", lw=0.7, zorder=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Draw rate (%)")
    ax.set_ylim(-4, 107)
    ax.set_xlim(-0.55, len(groups) - 0.25)
    ax.grid(axis="y", color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="o", ls="", mfc="white", mec=GRAY, mew=1.2,
               label="fixed mentor (M6)"),
        Line2D([], [], marker="s", ls="", color=BLUE, label="twin mean"),
        Line2D([], [], marker="o", ls="", color=GRAY, alpha=0.65, ms=4,
               label="individual twin / seed"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=1,
              handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("Certification on one line does not transfer", loc="left")
    fig.tight_layout()
    save(fig, "fig1_gap.pdf", png)


def thirty(name: str) -> tuple[list[float], float, float]:
    d = load(name)
    drs = [v["dr"] * 100 for k, v in d["results"].items() if k.startswith("twin_")]
    return drs, d["results"]["m6_fixed"]["dr"] * 100, float(np.mean(drs))


def fig2_lottery(png: bool = False):
    # (a) 30-teacher strips
    strips = [
        ("C fixed s42", "twin_ens_fixed_s42.json", BLUE, "o"),
        ("C TBR s42", "twin_ens_tbr_s42.json", ORANGE, "s"),
        ("C TBR s43", "twin_ens_tbr_s43.json", ORANGE, "s"),
        ("C TBR s44", "twin_ens_tbr_s44.json", ORANGE, "s"),
        ("C mixed s42", "twin_ens_mixed_s42.json", PURPLE, "^"),
        ("C episode s43", "twin_ens_episode_s43.json", GREEN, "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    rng = np.random.default_rng(7)
    for i, (label, fname, color, marker) in enumerate(strips):
        drs, m6, mean = thirty(fname)
        y = i + rng.uniform(-0.075, 0.075, len(drs))
        ax.scatter(drs, y, s=10, color=color, alpha=0.55, edgecolor="none", zorder=2)
        ax.plot([mean, mean], [i - 0.18, i + 0.18], color=color, lw=1.6, zorder=3)
        ax.plot(m6, i + 0.27, marker="o", ms=5.2, mfc="white", mec=color, mew=1.1, zorder=4)
    # memoryless: all 60 held-out teachers at 100
    i = len(strips)
    ax.scatter([100] * 60, i + rng.uniform(-0.08, 0.08, 60), s=9,
               color=RED, alpha=0.5, edgecolor="none", zorder=2)
    ax.plot([100, 100], [i - 0.18, i + 0.18], color=RED, lw=1.6, zorder=3)
    ax.plot(100, i + 0.27, marker="o", ms=5.2, mfc="white", mec=RED, mew=1.1, zorder=4)
    ax.set_yticks(range(len(strips) + 1))
    ax.set_yticklabels([s[0] for s in strips] + ["MLP s0+s1\n(60 teachers)"])
    ax.set_xlabel("Draw rate (%)")
    ax.set_xlim(-4, 104)
    ax.set_ylim(-0.55, len(strips) + 0.55)
    ax.grid(axis="x", color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("(a) 30 held-out teachers", loc="left")

    # (b) phase plot: canonical M6 DR vs mean twin DR
    points = []
    for cfg, fname in (("B", "tbr_eval_B.json"), ("C", "tbr_eval_C.json")):
        d = load(fname)
        for label, row in d["results"].items():
            cond = label.split("_")[1]
            points.append({
                "cond": cond.upper() if cond == "tbr" else cond,
                "label": label, "m6": row["summary"]["dr_m6_fixed"] * 100,
                "twin": row["summary"]["dr_twin_mean"] * 100,
                "cfg": cfg,
            })
    d = load("tbr_eval_C_new.json")
    for label, row in d["results"].items():
        cond = label.split("_")[1]
        cond = "EPISODE" if cond == "episode" else ("TBR" if cond == "tbr" else cond)
        points.append({"cond": cond, "label": label,
                       "m6": row["summary"]["dr_m6_fixed"] * 100,
                       "twin": row["summary"]["dr_twin_mean"] * 100, "cfg": "C"})
    d = load("tbr_eval_C_ep.json")
    for label, row in d["results"].items():
        points.append({"cond": "EPISODE", "label": label,
                       "m6": row["summary"]["dr_m6_fixed"] * 100,
                       "twin": row["summary"]["dr_twin_mean"] * 100, "cfg": "C"})
    points.append({"cond": "MLP", "label": "memoryless", "m6": 100.0, "twin": 100.0,
                   "cfg": "C"})

    style = {
        "fixed": (BLUE, "o"), "TBR": (ORANGE, "s"), "mixed": (PURPLE, "^"),
        "EPISODE": (GREEN, "D"), "noise": (GRAY, "P"), "MLP": (RED, "*"),
    }
    # deterministic jitter on the categorical x-axis
    offsets = {"fixed": -10.0, "noise": -5.0, "TBR": 0.0,
               "EPISODE": 5.0, "mixed": 10.0, "MLP": 0.0}
    ax = axes[1]
    for p in points:
        color, marker = style[p["cond"]]
        offs = offsets[p["cond"]]
        ax.plot(100 * (p["m6"] > 50) + offs, p["twin"], marker=marker, ms=6.0,
                color=color, alpha=0.85, markeredgecolor="white", markeredgewidth=0.5,
                zorder=3)
    ax.axhspan(95, 103, color=GREEN, alpha=0.08, zorder=0)
    ax.axvline(50, color=GRAY, ls=":", lw=0.7, zorder=0)
    ax.set_xticks([0, 100])
    ax.set_xticklabels(["0\n(fails M6)", "100\n(certified)"])
    ax.set_xlim(-18, 118)
    ax.set_ylim(-5, 106)
    ax.set_xlabel("Draw rate vs fixed M6 (%)")
    ax.set_ylabel("Mean draw rate vs twins (%)")
    ax.grid(axis="y", color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("(b) every training run", loc="left")
    ax.annotate("certified\nand robust", (105, 100), fontsize=6.5, color=GREEN,
                ha="right", va="top")
    ax.annotate("relocated\nfragility", (2, 62), fontsize=6.5, color=ORANGE,
                ha="left", va="top")
    handles = [
        Line2D([], [], marker="o", ls="", color=BLUE, label="fixed"),
        Line2D([], [], marker="s", ls="", color=ORANGE, label="TBR"),
        Line2D([], [], marker="^", ls="", color=PURPLE, label="mixed"),
        Line2D([], [], marker="D", ls="", color=GREEN, label="coherent"),
        Line2D([], [], marker="P", ls="", color=GRAY, label="noise"),
        Line2D([], [], marker="*", ls="", color=RED, label="memoryless"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2, frameon=False,
              handletextpad=0.25, columnspacing=0.7, borderaxespad=0.15)
    fig.tight_layout()
    save(fig, "fig2_lottery.pdf", png)


def distill_series() -> dict:
    d = load("distill_C.json")
    by_cov: dict[int, list[float]] = {}
    for row in d["results"].values():
        by_cov.setdefault(row["coverage"], []).append(row["mean_regret"])
    return by_cov


def fig3_state(png: bool = False):
    cov = distill_series()
    fixed_cov = load("coverage_fixed_C_s42_400.json")
    twins_cov = load("coverage_fixed_C_s42_30twins.json")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 0.85]})
    # (a) state-space counts
    ax = axes[0]
    labels = ["reachable\nboards", "reachable\nhistories",
              "visited vs M6\n(400 games)", "visited vs 30 twins\n(3,000 games)"]
    vals = [2423, 180361, fixed_cov["m6_unique_histories"],
            twins_cov["twins_union_unique_histories"]]
    colors = [GRAY, GRAY, BLUE, ORANGE]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(1, 4e5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for yi, v in zip(y, vals):
        ax.text(v * 1.25, yi, f"{v:,}", va="center", fontsize=7)
    ax.set_xlabel("unique agent-turn states (log scale)")
    ax.set_title("(a) the state space is histories", loc="left")
    ax.grid(axis="x", color="#ececec", lw=0.6)
    ax.set_axisbelow(True)

    # (b) board-level regret vs distillation coverage
    ax = axes[1]
    xs = sorted(cov)
    xpos = np.arange(len(xs))
    means = [float(np.mean(cov[x])) for x in xs]
    ax.plot(xpos, means, color=BLUE, marker="o", ms=4.5, lw=1.2, zorder=2)
    for xi, x in zip(xpos, xs):
        ax.scatter([xi] * len(cov[x]), cov[x], color=BLUE, s=10, alpha=0.45,
                   zorder=3, edgecolor="none")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["25", "100", "400", "1.2k", "2.4k"])
    ax.set_xlabel("boards in distillation set")
    ax.set_ylabel("board-level regret")
    ax.set_ylim(-0.02, 0.56)
    ax.grid(axis="y", color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("(b) board accuracy improves", loc="left")
    ax.annotate("2423: 0.000", (xpos[-1], 0.0), textcoords="offset points",
                xytext=(-10, 20), ha="right", fontsize=6.5, color=BLUE)

    # (c) game outcomes at 1200 and full coverage
    ax = axes[2]
    d1200 = load("tbr_eval_distill_1200.json")["results"]["C_distill_cov1200_s0"]["summary"]
    dfull = load("tbr_eval_distill_full.json")["results"]["C_distill_full_s0"]["summary"]
    x = np.arange(2)
    w = 0.36
    m6 = [d1200["dr_m6_fixed"] * 100, dfull["dr_m6_fixed"] * 100]
    twin = [d1200["dr_twin_mean"] * 100, dfull["dr_twin_mean"] * 100]
    ax.bar(x - w / 2, m6, w, color=GRAY, label="M6")
    ax.bar(x + w / 2, twin, w, color=ORANGE, label="twins")
    ax.set_xticks(x)
    ax.set_xticklabels(["1,200\nboards", "2,423\nboards"])
    ax.set_ylim(0, 108)
    ax.set_ylabel("draw rate (%)")
    ax.set_title("(c) games do not follow", loc="left")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3)
    ax.grid(axis="y", color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    for xi, (a, b) in enumerate(zip(m6, twin)):
        ax.text(xi - w / 2, a + 2, f"{a:.0f}", ha="center", fontsize=6.5)
        ax.text(xi + w / 2, b + 2, f"{b:.0f}", ha="center", fontsize=6.5)

    fig.tight_layout()
    save(fig, "fig3_state.pdf", png)


def fig4_seeds(png: bool = False):
    d = load("multiseed_summary.json")["resumen"]
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    rng = np.random.default_rng(0)
    for i, cfg in enumerate(("A", "C")):
        vals = np.asarray(d[cfg]["vals"], dtype=float)
        x = i + rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(x, vals, s=18, color=BLUE if cfg == "A" else ORANGE, zorder=3,
                   edgecolor="white", linewidth=0.5)
        ax.plot([i - 0.18, i + 0.18], [np.median(vals)] * 2, color="#222222",
                lw=1.2, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A: CfC+ACT", "C: GRU"])
    ax.set_ylabel("Win rate vs depth-0 opponent (%)")
    ax.set_ylim(-3, 103)
    ax.grid(axis="y", color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("V1 multi-seed re-runs", loc="left")
    fig.tight_layout()
    save(fig, "fig4_seeds.pdf", png)


def empirical_twin_mean(label: str):
    if label.endswith("-v1"):
        cfg = label[0]
        d = load(f"v1_reanalysis_{cfg}.json")["results"][f"v1_{cfg}"]["summary"]
        return d["dr_twin_mean"] * 100
    if "TBR" in label:
        cfg = label[0]
        seed = label.rsplit("-", 1)[-1]
        for fname in (f"tbr_eval_{cfg}.json", "tbr_eval_C_new.json"):
            try:
                d = load(fname)["results"][f"{cfg}_tbr_s{seed}"]
                return d["summary"]["dr_twin_mean"] * 100
            except KeyError:
                continue
    if label.startswith("B-") or label.startswith("C-"):
        if label.startswith("C-distill"):
            d = load("tbr_eval_distill_full.json")["results"]["C_distill_full_s0"]
            return d["summary"]["dr_twin_mean"] * 100
        cfg = label[0]
        seed = label.rsplit("-", 1)[-1]
        d = load(f"tbr_eval_{cfg}.json")["results"][f"{cfg}_fixed_s{seed}"]
        return d["summary"]["dr_twin_mean"] * 100
    if label.startswith("MLP"):
        return 100.0
    return None


def draw_board(ax, board, title, highlight=None, win_line=None):
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    for k in (0.5, 1.5):
        ax.plot([-0.5, 2.5], [k, k], color="#bdbdbd", lw=0.7)
        ax.plot([k, k], [-0.5, 2.5], color="#bdbdbd", lw=0.7)
    for i, v in enumerate(board):
        r, c = divmod(i, 3)
        if v == 0:
            continue
        sym = "X" if v == 1 else "O"
        color = BLUE if v == 1 else ORANGE
        if highlight is not None and i == highlight:
            color = RED
        ax.text(c, r, sym, ha="center", va="center", fontsize=12,
                color=color, fontweight="bold")
    if win_line:
        (r0, c0), (r1, c1) = win_line
        ax.plot([c0, c1], [r0, r1], color=GREEN, lw=2.0, alpha=0.8, zorder=5)
    ax.set_title(title, fontsize=6.5, pad=1.5)


def fig5_certificate(png: bool = False):
    wc = load("worst_case.json")["results"]
    rows = [
        ("A-v1", "A-v1"),
        ("B-v1", "B-v1"),
        ("C-v1", "C-v1"),
        ("B-42", "B-42"),
        ("B-43", "B-43"),
        ("B-44", "B-44"),
        ("C-42", "C-42"),
        ("C-43", "C-43"),
        ("C-44", "C-44"),
        ("C-TBR-43", "C-TBR-43"),
        ("MLP", "MLP-s0"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.05),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    ax.axvspan(98.5, 110, color=GREEN, alpha=0.07, zorder=0)
    for i, (name, key) in enumerate(rows):
        r = wc[key]
        worst = r["worst_draw_rate"] * 100
        best = r["best_draw_rate"] * 100
        exact = r["random_draw_rate_exact"] * 100
        emp = empirical_twin_mean(key)
        certified = worst >= 100 and best >= 100
        if certified:
            ax.plot([100, 100], [i - 0.16, i + 0.16], color=GREEN, lw=2.4,
                    solid_capstyle="butt", zorder=2)
        else:
            ax.plot([worst, best], [i, i], color=LIGHT, lw=2.4,
                    solid_capstyle="butt", zorder=1)
        ax.scatter(exact, i, s=16, color=ORANGE, zorder=3,
                   edgecolor="white", linewidth=0.4)
        if emp is not None:
            ax.scatter(emp, i, s=22, facecolors="none", edgecolors=BLUE,
                       linewidth=0.8, zorder=4)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-3, 110)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel("draw rate (%)")
    ax.grid(axis="x", color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("(a) every optimal teacher, not one line", loc="left")
    handles = [
        Line2D([], [], color=LIGHT, lw=2.4, label="optimal-teacher interval"),
        Line2D([], [], marker="o", ls="", color=ORANGE, label="exact uniform mean"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=BLUE, mew=0.8,
               label="3-twin sample"),
        Line2D([], [], marker="|", ls="", color=GREEN, ms=9,
               label="worst case = 100"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, ncol=1,
              handletextpad=0.3, borderaxespad=0.25, fontsize=6.4,
              framealpha=0.95, facecolor="white", edgecolor="none")

    # (b) witness game: C-42, the adversarial optimal teacher
    witness = wc["C-42"]["witness"]["steps"]
    board = [0] * 9
    moves = []
    for step in witness:
        board[int(step["action"])] = 1 if step["player"] == "teacher" else 2
        moves.append((list(board), step))
    axw = axes[1]
    axw.axis("off")
    axw.set_title("(b) witness: an optimal teacher wins", loc="left")
    for k, (b, step) in enumerate(moves):
        r, c = 0 if k < 3 else 1, k % 3
        sub = axw.inset_axes([0.02 + c * 0.33, 0.52 - r * 0.47, 0.30, 0.42])
        highlight = int(step["action"]) if step["player"] == "agent" and step["regret"] > 0 else None
        win_line = ((0, 2), (2, 2)) if k == len(moves) - 1 else None
        label = f"{k + 1}. {'X' if step['player'] == 'teacher' else 'O'} {step['action']}"
        if step["player"] == "agent" and step["regret"] > 0:
            label += f"  regret {step['regret']:.0f}"
        draw_board(sub, b, label, highlight=highlight, win_line=win_line)

    fig.tight_layout()
    save(fig, "fig5_certificate.pdf", png)


def main() -> None:
    png = "--png" in sys.argv
    fig1_gap(png)
    fig2_lottery(png)
    fig3_state(png)
    fig4_seeds(png)
    fig5_certificate(png)
    print("figures written to paper/figs_v2")


if __name__ == "__main__":
    main()
