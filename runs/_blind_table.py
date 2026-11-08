"""Emit and check the rows of `tab:blind`: every repaired arm that has been read on both suites.

Every cell comes from the archived JSONs, and the script refuses to print an arm whose two
implementations disagree. For each arm it takes the certified side (suite A) from the memo-free
verifier when that read exists and from the loop's own `repair.final` when it does not, and the blind
side (suite B) always from the verifier. As of the reads archived on 2026-11-05 every arm in the
table has an independent read of its own suite, so no row carries the dagger any more; the mechanism
stays because a new depth would arrive with its first read from the loop. Where both exist for
suite A they must agree on the worst case, the best case, the uniform-optimal mean and the roots
drawn, which is what licenses calling the table a measurement rather than a report.

The claim the table carries is one-sided, so it is asserted here too: an arm that certifies on B
would contradict `sec:blind`, so this script exits non-zero if any arm reaches W=1 on the blind
suite.

The lower block of `tab:blind` is the paired control, emitted by the same run: for each seed in
PAIRED the memo-free read of `c4_gru_base.pt` on both suites, next to the repaired row above it.
Two things are asserted rather than printed---that no unrepaired base certifies (the base rows would
otherwise stop being a control), and that the training loop's own base certificate agrees with the
verifier on suite A for every seed in `PAIRED`. That list is read off `data/`, not remembered: an arm
is in it exactly when its two base reads are on disk, and the run prints how many that is. The six
arms that trained before `c4_gru_base.pt` was ever saved joined it on 2026-11-05, when
`runs/_refit_bases.py` re-derived their bases on the machine that trained them and read them here.
That agreement is what lets the gain be called a measurement twice over.

    python runs/_blind_table.py
"""
import json
import math
import pathlib
import sys

R = pathlib.Path("data")

# (empties, seed, arm JSON, verifier read on A or None, verifier read on B)
ARMS = [(8, 0, "c4_end_ctrl_s0", "c4_end_verify_s0", "c4_end8_holdoutB_s0"),
        (8, 1, "c4_end_s1", "c4_end8_verify_s1", "c4_end8_holdoutB_s1"),
        (8, 2, "c4_end_s2", "c4_end8_verify_s2", "c4_end8_holdoutB_s2"),
        (8, 3, "c4_end8_s3", "c4_end8_verify_s3", "c4_end8_holdoutB_s3"),
        (8, 4, "c4_end8_s4", "c4_end8_verify_s4", "c4_end8_holdoutB_s4"),
        (8, 5, "c4_end8_s5", "c4_end8_verify_s5", "c4_end8_holdoutB_s5"),
        (10, 0, "c4_end10_s0", "c4_end10_verify_s0", "c4_end10_holdoutB_s0"),
        (10, 1, "c4_end10_s1", "c4_end10_verify_s1", "c4_end10_holdoutB_s1"),
        (10, 2, "c4_end10_s2", "c4_end10_verify_s2", "c4_end10_holdoutB_s2"),
        (10, 3, "c4_end10_s3", "c4_end10_verify_s3", "c4_end10_holdoutB_s3"),
        (10, 4, "c4_end10_s4", "c4_end10_verify_s4", "c4_end10_holdoutB_s4"),
        (10, 5, "c4_end10_s5", "c4_end10_verify_s5", "c4_end10_holdoutB_s5"),
        (12, 0, "c4_end12_s0", "c4_end12_verify_s0", "c4_end12_holdoutB_s0"),
        (12, 1, "c4_end12_s1", "c4_end12_verify_s1", "c4_end12_holdoutB_s1"),
        (12, 2, "c4_end12_s2", "c4_end12_verify_s2", "c4_end12_holdoutB_s2"),
        (12, 3, "c4_end12_s3", "c4_end12_verify_s3", "c4_end12_holdoutB_s3"),
        (12, 4, "c4_end12_s4", "c4_end12_verify_s4", "c4_end12_holdoutB_s4"),
        (12, 5, "c4_end12_s5", "c4_end12_verify_s5", "c4_end12_holdoutB_s5"),
        (12, 6, "c4_end12_s6", "c4_end12_verify_s6", "c4_end12_holdoutB_s6"),
        (12, 7, "c4_end12_s7", "c4_end12_verify_s7", "c4_end12_holdoutB_s7"),
        (12, 8, "c4_end12_s8", "c4_end12_verify_s8", "c4_end12_holdoutB_s8"),
        (12, 9, "c4_end12_s9", "c4_end12_verify_s9", "c4_end12_holdoutB_s9")]

# The paired control that rules out "suite B is simply harder": the same seeds read before any
# repair, on both suites. Derived rather than typed, because a hand-kept list of arms keeps its old
# shape after the repository changes - an arm is paired exactly when its two base reads are on disk,
# and the run prints how many that is. Each of them still has to agree with the loop's own base
# certificate on A, which is asserted below and not assumed here.
PAIRED = [(E, s, arm_name) for (E, s, arm_name, _vA, _vB) in ARMS
          if all((R / ("c4_end%d_base_s%d_on%s.json" % (E, s, w))).is_file() for w in "AB")]


def load(name):
    return json.loads((R / (name + ".json")).read_text(encoding="utf-8"))


def pct(x):
    return "%.1f" % (100 * x)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


rows, fails = [], []
for E, s, arm_name, vA_name, vB_name in ARMS:
    arm = load(arm_name)
    fin = arm["repair"]["final"]
    vB = load(vB_name)
    loop = dict(W=fin["worst_draw_rate"], B=fin["best_draw_rate"], D=fin["random_draw_rate_exact"],
                dr=fin["n_roots_drawn_worst"])
    row = dict(E=E, s=s, loop=loop, loopA=vA_name is None)
    if vA_name:
        vA = load(vA_name)
        agree = (("W", loop["W"], vA["worst_draw_rate"]),
                 ("B", loop["B"], vA["best_draw_rate"]),
                 ("drawn", loop["dr"], vA["n_roots_drawn_worst"]))
        for key, a, b in agree:
            if a != b:
                fails.append((arm_name, key, a, b))
        if abs(loop["D"] - vA["random_draw_rate_exact"]) > 1e-6:
            fails.append((arm_name, "D", loop["D"], vA["random_draw_rate_exact"]))
        row.update(W=vA["worst_draw_rate"], B=vA["best_draw_rate"], D=vA["random_draw_rate_exact"],
                   dr=vA["n_roots_drawn_worst"])
    else:
        row.update(**loop)
    row.update(BW=vB["worst_draw_rate"], BB=vB["best_draw_rate"], BD=vB["random_draw_rate_exact"],
               Bdr=vB["n_roots_drawn_worst"], Bmin=vB["order_min"], Badapt=vB["adaptive_min"])
    rows.append(row)

assert not fails, "loop and verifier disagree on suite A: %r" % (fails,)
certified_A = sum(1 for r in rows if r["W"] == 1.0)
for r in rows:
    if r["BW"] == 1.0:
        fails.append(("arm certifies on the blind suite", r["E"], r["s"]))
assert not fails, repr(fails)

print("%d arms read on both suites; %d of them hold W=1 on the suite they were repaired "
      "against; %d hold W=1 on the blind one" % (len(rows), certified_A,
                                                 sum(1 for r in rows if r["BW"] == 1.0)))
print("loop-vs-verifier on suite A checked in %d arms with an independent read; "
      "%d arms report their own suite from the loop (%s)" % (
          sum(1 for r in rows if not r["loopA"]), sum(1 for r in rows if r["loopA"]),
          ", ".join("%d/s%d" % (r["E"], r["s"]) for r in rows if r["loopA"])
          or "none: every arm has an independent read of the suite it certified"))
print()
print("Empty & Seed & $W$ & $\\bar D$ & drawn & $W$ & $B$ & $\\bar D$ & drawn   (A then B)")
for r in rows:
    print("%d & %d & $%g$%s & %s & %d & $%g$ & $%g$ & %s & %d \\\\   [blind order_min %.3f, "
          "adaptive_min %.3f]" % (
              r["E"], r["s"], r["W"], "$^{\\dagger}$" if r["loopA"] else "", pct(r["D"]), r["dr"],
              r["BW"], r["BB"], pct(r["BD"]), r["Bdr"], r["Bmin"], r["Badapt"]))
print()
for depth in sorted({r["E"] for r in rows}):
    sub = [r for r in rows if r["E"] == depth]
    lo, hi = wilson(0, len(sub))
    print("depth %2d: %d arms blind-read, mean blind $\\bar D$ %s, drawn %d of %d roots in "
          "the worst case; 0/%d certified, Wilson 95%% upper %s" % (
              depth, len(sub),
              pct(sum(r["BD"] for r in sub) / len(sub)),
              sum(r["Bdr"] for r in sub), 40 * len(sub), len(sub), "%.3f" % hi))
lo, hi = wilson(0, len(rows))
print("all %d arms: 0 certified on the blind suite, Wilson 95%% upper %.3f" % (len(rows), hi))

by_arm = {(r["E"], r["s"]): r for r in rows}
print()
print("paired base rows (same seeds before repair, memo-free verifier on c4\\_gru\\_base.pt)")
gains = []
for E, s, arm_name in PAIRED:
    r = by_arm[(E, s)]
    bA, bB = load("c4_end%d_base_s%d_onA" % (E, s)), load("c4_end%d_base_s%d_onB" % (E, s))
    loopb = load(arm_name)["base"]["certificate"]
    for key, v in (("worst_draw_rate", "W"), ("best_draw_rate", "B"),
                   ("n_roots_drawn_worst", "drawn")):
        if loopb[key] != bA[key]:
            fails.append(("base loop-vs-verifier on A", E, s, v, loopb[key], bA[key]))
    if abs(loopb["random_draw_rate_exact"] - bA["random_draw_rate_exact"]) > 1e-9:
        fails.append(("base loop-vs-verifier on A", E, s, "D",
                      loopb["random_draw_rate_exact"], bA["random_draw_rate_exact"]))
    if bA["worst_draw_rate"] == 1.0 or bB["worst_draw_rate"] == 1.0:
        fails.append(("an unrepaired base certifies", E, s))
    dA, dB = bA["random_draw_rate_exact"], bB["random_draw_rate_exact"]
    gains.append((E, s, 100 * (r["D"] - dA), 100 * (r["BD"] - dB), 100 * (dA - dB)))
    print("%d & %d & $%g$ & %s & %d & $%g$ & $%g$ & %s & %d \\\\   [base gap %.1f, gain A %+.1f, "
          "gain B %+.1f]" % (
              E, s, bA["worst_draw_rate"], pct(dA), bA["n_roots_drawn_worst"],
              bB["worst_draw_rate"], bB["best_draw_rate"], pct(dB), bB["n_roots_drawn_worst"],
              gains[-1][4], gains[-1][2], gains[-1][3]))
assert not fails, repr(fails)


def span(xs):
    return "%.1f to %.1f" % (min(xs), max(xs))


def mean(xs):
    return "%+.1f" % (sum(xs) / len(xs))


for depth in sorted({g[0] for g in gains}):
    sub = [g for g in gains if g[0] == depth]
    if sub:
        print("depth %2d: %d paired bases, B sits %s points below A unrepaired, repair gains "
              "%s on A (mean %s) and %s on B (mean %s)" % (
                  depth, len(sub), span([g[4] for g in sub]), span([g[2] for g in sub]),
                  mean([g[2] for g in sub]), span([g[3] for g in sub]), mean([g[3] for g in sub])))
print("all %d paired arms: the base gap is %s points, the repair gains %s on A (mean %s) against "
      "%s on B (mean %s), negative in %d of %d" % (
          len(gains), span([g[4] for g in gains]), span([g[2] for g in gains]),
          mean([g[2] for g in gains]), span([g[3] for g in gains]), mean([g[3] for g in gains]),
          sum(1 for g in gains if g[3] < 0), len(gains)))


