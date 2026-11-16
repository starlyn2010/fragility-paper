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

# The mirror section's prose: per (depth, seed), (refits, added histories, Dbar on the suite the
# arm never saw, roots drawn there, open violations in the blind-aimed base). The last value is the
# "29, 28, 31, 56, 60 and 61" sentence; the first four are the table's columns and the cost prose.
BLIND_PROSE = {(8, 0): (7, 103, 41.4, 12, 29),
               (8, 1): (4, 77, 40.2, 11, 28),
               (8, 2): (6, 66, 29.9, 7, 31),
               (10, 0): (23, 247, 27.0, 7, 56),
               (10, 1): (16, 223, 26.4, 6, 60),
               (10, 2): (19, 218, 35.2, 8, 61)}

FORTY = {}
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
        FORTY[(E, s)] = (rounds, len(rep["aggregate"]))
        vA, vB = load(stem + "_verifyA.json"), load(stem + "_verifyB.json")
        bA, bB = load(stem + "_base_onA.json"), load(stem + "_base_onB.json")
        assert (rounds, len(rep["aggregate"]), round(100 * vA["random_draw_rate_exact"], 1),
                vA["n_roots_drawn_worst"],
                len(arm["base"]["certificate"]["violations"])) == BLIND_PROSE[(E, s)], stem
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

# The abstract, the body and the conclusion quote the cost of forty positions for the three
# seeds the union arms share, aimed in either direction. Those six arms are the three published
# eight-cell arms (`TABLE`) and the three blind-aimed ones just read, so pin both ends here.
matched = [TABLE[(8, s)] for s in (0, 1, 2)] + [FORTY[(8, s)] for s in (0, 1, 2)]
lo_h, hi_h = min(h for _, h in matched), max(h for _, h in matched)
lo_r, hi_r = min(r for r, _ in matched), max(r for r, _ in matched)
assert (lo_h, hi_h) == (66, 111), matched
assert (lo_r, hi_r) == (4, 8), matched
print("forty positions, same three seeds in both directions: %d--%d histories in %d--%d refits"
      % (lo_h, hi_h, lo_r, hi_r))

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


# Every number Sect. 7 prints about the union arms, pinned. Dbar is a percent rounded the way
# the paper rounds it; a/b are (W, Dbar%, roots drawn) of the repair read on each suite. `lost`
# counts the blind roots the union base loses to some optimal tie-break, `left` those still open
# after repair. The abstract, body and conclusion also quote the 161--211 / 4--10 cost range.
UNION_PROSE = {
    0: dict(rounds=12, capped=True, added=191, base_d=79.3, base_viol=67,
            base_a=79.5, base_b=79.0, base_b_drawn=24,
            a=(1.0, 100.0, 40), b=(0.0, 99.2, 39), lost=16, left=1),
    1: dict(rounds=10, capped=False, added=211, base_d=78.2, base_viol=83,
            base_a=74.1, base_b=82.2, base_b_drawn=24,
            a=(1.0, 100.0, 40), b=(1.0, 100.0, 40), lost=16, left=0),
    2: dict(rounds=4, capped=False, added=161, base_d=80.6, base_viol=69,
            base_a=77.0, base_b=84.2, base_b_drawn=26,
            a=(1.0, 100.0, 40), b=(1.0, 100.0, 40), lost=14, left=0),
}
UNION_COST = {}

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
        if E == 8:
            pin = UNION_PROSE[s]
            base = arm["base"]["certificate"]
            assert (rounds, len(rep["aggregate"])) == (pin["rounds"], pin["added"]), stem
            assert (not broke) == pin["capped"], (stem, "stopped by the cap")
            assert round(100 * base["random_draw_rate_exact"], 1) == pin["base_d"], stem
            assert len(base["violations"]) == pin["base_viol"], stem
            bA, bB = load(stem + "_base_A.json"), load(stem + "_base_B.json")
            assert round(100 * bA["random_draw_rate_exact"], 1) == pin["base_a"], stem
            assert round(100 * bB["random_draw_rate_exact"], 1) == pin["base_b"], stem
            assert bB["n_roots_drawn_worst"] == pin["base_b_drawn"], stem
            assert (bA["worst_draw_rate"], bB["worst_draw_rate"]) == (0.0, 0.0), stem
            for which in ("A", "B"):
                v = load("%s_%s.json" % (stem, which))
                got = (v["worst_draw_rate"], round(100 * v["random_draw_rate_exact"], 1),
                       v["n_roots_drawn_worst"])
                assert got == pin[which.lower()], (stem, which, got, pin[which.lower()])
            lost = len({v["root"] for v in base["violations"] if v["root"] >= HALF})
            left = len({v["root"] for v in fin["violations"] if v["root"] >= HALF})
            assert (lost, left) == (pin["lost"], pin["left"]), (stem, lost, left)
            UNION_COST[s] = (rounds, len(rep["aggregate"]), not broke)
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

cert = [UNION_COST[s] for s in (0, 1, 2) if not UNION_COST[s][2]]
assert [s for s in (0, 1, 2) if UNION_COST[s][2]] == [0], UNION_COST
lo_h, hi_h = min(h for _, h, _ in cert), max(h for _, h, _ in cert)
lo_r, hi_r = min(r for r, _, _ in cert), max(r for r, _, _ in cert)
assert (lo_h, hi_h) == (161, 211), cert
assert (lo_r, hi_r) == (4, 10), cert
assert UNION_COST[0][:2] == (12, 191), UNION_COST[0]
print("union arms: the two that certify cost %d--%d histories in %d--%d refits; seed 0 spends "
      "the %d-refit cap" % (lo_h, hi_h, lo_r, hi_r, UNION_COST[0][0]))
