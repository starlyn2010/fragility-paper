"""Emit the LaTeX rows for the reverse direction of `tab:blind` (fit against suite B).

Every number is read from the archived JSONs: the six repaired arms' verifier reads on both
suites (`c4_end{8,10}B_s*_verify{A,B}.json`), the same pair of reads taken on each arm's
`c4_gru_base.pt` (`*_base_on{A,B}.json`), and the loop's own bookkeeping for rounds and added
histories. Rounds follow `tab:scaling76`'s convention: the loop's last logged row is the
re-certification taken after the final refit, so an arm that breaks on a certificate is
counted at len(log)-1 and an arm the cap stopped is counted at the cap.

    python runs/_scope_table.py
"""
import json
import pathlib

R = pathlib.Path("data")
CAP = {8: 12, 10: 24}


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def pct(x):
    return "%.1f" % (100 * x)


def rounds_added(arm):
    rep = arm["repair"]
    log = rep["log"]
    broke = log[-1]["worst_draw_rate"] == 1.0
    return (len(log) - 1 if broke else len(log)), len(rep["aggregate"])


# the rule above has to reproduce tab:scaling76, whose rows were read off the published arms
PUBLISHED = {(8, 0): "c4_end_ctrl_s0", (8, 1): "c4_end_s1", (8, 2): "c4_end_s2",
             (10, 0): "c4_end10_s0", (10, 1): "c4_end10_s1", (10, 2): "c4_end10_s2",
             (12, 0): "c4_end12_s0", (12, 1): "c4_end12_s1", (12, 2): "c4_end12_s2",
             (12, 3): "c4_end12_s3", (12, 4): "c4_end12_s4", (12, 5): "c4_end12_s5",
             (12, 6): "c4_end12_s6"}
TABLE = {(8, 0): (8, 111), (8, 1): (6, 102), (8, 2): (7, 95),
         (10, 0): (17, 214), (10, 1): (14, 166), (10, 2): (7, 166),
         (12, 0): (22, 385), (12, 1): (24, 413), (12, 2): (24, 401), (12, 3): (22, 391),
         (12, 4): (24, 406), (12, 5): (24, 420), (12, 6): (24, 406)}
for key, name in PUBLISHED.items():
    got = rounds_added(load(name + ".json"))
    assert got == TABLE[key], (key, got, TABLE[key])
print("rounds rule reproduces tab:scaling76 in %d/%d arms" % (len(PUBLISHED), len(PUBLISHED)))

for E in (8, 10):
    for s in (0, 1, 2):
        stem = "c4_end%dB_s%d" % (E, s)
        arm = load(stem + ".json")
        rep = arm["repair"]
        log, final = rep["log"], rep["final"]
        # a row is logged at the top of each iteration and carries the certificate *before*
        # that iteration's refit, so the loop broke on a certificate exactly when the last
        # logged row already reads W=1; otherwise the cap ran out and every row was a refit
        broke = log[-1]["worst_draw_rate"] == 1.0
        rounds, capped = (len(log) - 1, False) if broke else (len(log), True)
        vA, vB = load(stem + "_verifyA.json"), load(stem + "_verifyB.json")
        bA, bB = load(stem + "_base_onA.json"), load(stem + "_base_onB.json")
        print("%% rounds %d%s added %d" % (
            rounds, " (cap)" if capped else "", len(rep["aggregate"])))
        print("   latex: %d & %d & $%g$ & %s & %d & $%g$ & %s & %d & %s & %d \\\\" % (
            E, s, vB["worst_draw_rate"], pct(vB["random_draw_rate_exact"]),
            vB["n_roots_drawn_worst"],
            vA["worst_draw_rate"], pct(vA["random_draw_rate_exact"]),
            vA["n_roots_drawn_worst"],
            "%d%s" % (rounds, "$^{c}$" if capped else ""), len(rep["aggregate"])))
        print("   base : %d & %d & %s & %d & %s & %d \\\\   gain B %+.1f  gain A %+.1f" % (
            E, s, pct(bB["random_draw_rate_exact"]), bB["n_roots_drawn_worst"],
            pct(bA["random_draw_rate_exact"]), bA["n_roots_drawn_worst"],
            100 * (vB["random_draw_rate_exact"] - bB["random_draw_rate_exact"]),
            100 * (vA["random_draw_rate_exact"] - bA["random_draw_rate_exact"])))
        print("   loop-vs-verifier on the blind base: loop %.4f vs verifier %.4f  "
              "violations %d" % (arm["base"]["certificate"]["random_draw_rate_exact"],
                                 bB["random_draw_rate_exact"],
                                 len(arm["base"]["certificate"]["violations"])))

# --- the union arms: certify A cup B, then read the result on A, on B and on the union -----
# `certify` takes a minimum over roots of a per-root worst case, so on two suites that share no
# board W over the union is min(W_A, W_B) and Dbar over the union is the mean of the two Dbar.
# Both are checked here against the verifier's three independent reads of the same weights.
print("\n=== union arms (certificate over A cup B, 80 roots) ===")
HALF = 40  # the committed union file is suite A's forty roots followed by suite B's forty


def per_suite(per_root):
    out = []
    for block in (per_root[:HALF], per_root[HALF:]):
        ws = [r["w"] for r in block]
        ds = [r["d"] for r in block]
        out.append((min(ws), max(ws), sum(ds) / len(ds), sum(1 for w in ws if w == 1.0)))
    return out


for E in (8, 10):
    for s in (0, 1, 2):
        stem = "c4_end%dU_s%d" % (E, s)
        try:
            arm = load(stem + ".json")
        except FileNotFoundError:
            continue
        rep = arm["repair"]
        broke = rep["log"][-1]["worst_draw_rate"] == 1.0
        rounds = len(rep["log"]) - 1 if broke else len(rep["log"])
        fin = rep["final"]
        (WA, BA, DA, drA), (WB, BB, DB, drB) = per_suite(rep["per_root"])
        # the loop's own union certificate must be the min/mean/sum of its own per-root split
        assert fin["worst_draw_rate"] == min(WA, WB), (stem, "W over the union is not the min")
        assert fin["best_draw_rate"] == max(BA, BB), (stem, "B over the union is not the max")
        assert abs(fin["random_draw_rate_exact"] - (DA + DB) / 2) < 1e-9, (
            stem, "Dbar over the union is not the mean")
        assert fin["n_roots_drawn_worst"] == drA + drB, (stem, "drawn counts are not additive")
        print("%s refits %2d%s added %3d | loop split  A: W=%g D=%s dr=%2d/40 | "
              "B: W=%g D=%s dr=%2d/40   identities checked on per_root" % (
                  stem, rounds, " (cap)" if not broke else "", len(rep["aggregate"]),
                  WA, pct(DA), drA, WB, pct(DB), drB))
        done = []
        for which, want in (("A", (WA, DA, drA)), ("B", (WB, DB, drB))):
            try:
                v = load("%s_%s.json" % (stem, which))
            except FileNotFoundError:
                continue
            got = (v["worst_draw_rate"], v["random_draw_rate_exact"],
                   v["n_roots_drawn_worst"])
            assert abs(got[1] - want[1]) < 1e-6 and got[0] == want[0] and got[2] == want[2], (
                stem, which, "verifier disagrees with the loop's per-root split", got, want)
            done.append(which)
        try:
            v = load("%s_U.json" % stem)
        except FileNotFoundError:
            v = None
        if v:
            assert v["worst_draw_rate"] == fin["worst_draw_rate"], (stem, "U read vs loop")
            assert abs(v["random_draw_rate_exact"]
                       - fin["random_draw_rate_exact"]) < 1e-6, (stem, "U Dbar")
            assert v["n_roots_drawn_worst"] == fin["n_roots_drawn_worst"], (stem, "U drawn")
            done.append("U")
        print("   memo-free verifier agrees on: %s   outstanding: %s" % (
            ", ".join(done) or "nothing yet",
            ", ".join(w for w in ("A", "B", "U") if w not in done) or "nothing"))
