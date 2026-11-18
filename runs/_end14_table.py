"""Emit and check the rows of `tab:end14`: the fourteen-empty arms run on a T4.

Every cell comes from the JSONs of those arms, and the script refuses to print the table if the
two implementations disagree or if the claims the text makes about the arms stop holding:

* for each arm, the loop's base certificate and the memo-free read of `c4_gru_base.pt` on suite A
  must be the same number, which is what makes the base row a measurement and the gain a
  difference;
* for seed 3, the loop's certificate at the end of the first session (round 24, where the cap
  stopped it) and the memo-free read taken then - kept as `_r24_` when the arm was resumed - must
  agree;
* for each arm, the loop's final certificate and the memo-free read of the repaired weights on
  suite A must agree to 1e-9, root counts included;
* the arms must still be uncertified (W=0), and the blind suite must read W=0 at every repaired
  checkpoint - if a later session certifies one, this script fails so the section is rewritten
  rather than left saying "it does not certify" over a certificate. The best case on the blind
  suite is B=0 for seeds 3, 1 and 0, but seed 2 reaches one draw there (1 of 40, B=1) from a base
  that reaches none, and the section prints that exception rather than hiding it under a blanket
  "no draw";
* seed 3 must have stopped the way the section says it stopped: at the wall, with every violating
  history already in the aggregate, which is what leaves the loop with no move left;
* seeds 1 and 0 must have stopped at the 24-round cap with every violating history of their final
  certificates already in the aggregate - the section says a longer cap would have stopped both
  one round later on "no new violations", and that is exactly this check;
* seed 2 must have stopped at the same cap with the opposite signature - 64 of its 72 final
  violations absent from the aggregate - because the section says the cap, not the wall, ended
  that arm: a check that it stopped at the cap is not enough, the new material is the point;
* the refit rule the notebook adds must still be the one the JSON records: the stated numbers of
  fits reach exact optimality and are cut by patience, the aggregate regret is the list the text
  prints, and none of the fits reached the 200-epoch cap;
* the last fit of each arm must have kept what the text says it kept: seeds 3, 1 and 0 handed back
  the weights they were given (`kept: incumbent`, best equal to entry), while seed 2's took one
  epoch from 155 stop-set mistakes to 120 and nothing after beat that, which is what makes its
  last sentence different from the other three.

    python runs/_end14_table.py
"""
import json
import pathlib

R = pathlib.Path("data")
SUITE = "data/c4_endgame_roots_14_40_s0.json"
SUITE_B = "data/c4_endgame_roots_14_40_holdoutB.json"


def load(name):
    return json.loads((R / (name + ".json")).read_text(encoding="utf-8"))


def feet(arm, tag, blind_best=0.0):
    """The checks every arm shares: base and final read on A, no certificate, blind at zero.

    `blind_best` is the best case the section prints for the arm's repaired weights on the blind
    suite: 0 for seeds 3, 1 and 0, and 1 for seed 2, which reaches one draw there in 40 roots.
    """
    fin = arm["repair"]["final"]
    base = arm["base"]["certificate"]
    vA = load(tag + "_verify")
    vB = load(tag + "_holdoutB")
    bA = load(tag + "_base_onA")
    bB = load(tag + "_base_onB")
    out = []
    if arm["args"]["roots_file"] != SUITE:
        out.append(("the arm did not read the published suite", arm["args"]["roots_file"]))
    prot = arm["protocol"]
    if prot.get("published") is not False or not prot.get("keep_best"):
        out.append(("the arm does not carry the rule the section describes", prot))

    for key, a, b in (("W", base["worst_draw_rate"], bA["worst_draw_rate"]),
                      ("B", base["best_draw_rate"], bA["best_draw_rate"]),
                      ("drawn", base["n_roots_drawn_worst"], bA["n_roots_drawn_worst"])):
        if a != b:
            out.append(("base loop-vs-verifier on A", tag, key, a, b))
    if abs(base["random_draw_rate_exact"] - bA["random_draw_rate_exact"]) > 1e-9:
        out.append(("base loop-vs-verifier on A", tag, "D",
                    base["random_draw_rate_exact"], bA["random_draw_rate_exact"]))

    for key, a, b in (("W", fin["worst_draw_rate"], vA["worst_draw_rate"]),
                      ("B", fin["best_draw_rate"], vA["best_draw_rate"]),
                      ("drawn", fin["n_roots_drawn_worst"], vA["n_roots_drawn_worst"]),
                      ("roots", fin["n_roots"], vA["roots"])):
        if a != b:
            out.append(("repair loop-vs-verifier on A", tag, key, a, b))
    if abs(fin["random_draw_rate_exact"] - vA["random_draw_rate_exact"]) > 1e-9:
        out.append(("repair loop-vs-verifier on A", tag, "D",
                    fin["random_draw_rate_exact"], vA["random_draw_rate_exact"]))

    if fin["worst_draw_rate"] == 1.0 or vA["worst_draw_rate"] == 1.0:
        out.append(("the arm certifies: the section says it does not", tag))
    if vB["worst_draw_rate"] != 0.0 or vB["best_draw_rate"] != blind_best:
        out.append(("the blind suite reads W=%r B=%r; the section says W=0 B=%r"
                    % (vB["worst_draw_rate"], vB["best_draw_rate"], blind_best), tag))
    if bA["worst_draw_rate"] == 1.0 or bB["worst_draw_rate"] == 1.0:
        out.append(("an unrepaired base certifies", tag))
    return fin, vA, vB, bA, bB, out


def fits_check(arm, tag, optimal_n, patience_n, regret):
    """The refit rule, pinned to the counts and the round-by-round regret the text prints."""
    fits = arm["repair"]["fits"]
    optimal = [f for f in fits if f["stopped_by"] == "optimal"]
    cut = [f for f in fits if f["stopped_by"] == "patience"]
    out = []
    if not optimal or len(optimal) + len(cut) != len(fits):
        out.append(("the rule did not behave as the section says: %d optimal, %d patience, "
                    "%d fits" % (len(optimal), len(cut), len(fits)), tag))
    if any(f["stopped_by"] == "cap" for f in fits):
        out.append(("a fit ran to its 200-epoch cap", tag))
    if (len(optimal), len(cut)) != (optimal_n, patience_n):
        out.append(("the text says %d optimal and %d patience, the JSON has %d and %d"
                    % (optimal_n, patience_n, len(optimal), len(cut)), tag))
    wall = [(f["round"], f["agg_mistakes"]) for f in fits if f["agg_mistakes"] > 0]
    if wall != regret:
        out.append(("the text prints aggregate regret %r, the JSON has %r" % (regret, wall), tag))
    return optimal, cut, wall, out


fails = []

# ---- seed 3: the arm stopped at the wall, resumed once and read at both checkpoints ----------
S3 = "c4_end14_t4_s3"
s3 = load(S3)
fin3, vA3, vB3, bA3, bB3, f3 = feet(s3, S3)
fails += f3
r24 = load(S3 + "_r24_verify")
r24B = load(S3 + "_r24_holdoutB")
at24 = next(r for r in s3["repair"]["log"] if r["round"] == 24)
# The round-24 state is the loop's own certificate of that round, and the two `_r24_` files are
# the independent reads the first session took of the same weights before the arm was resumed.
for key, a, b in (("W", at24["worst_draw_rate"], r24["worst_draw_rate"]),
                  ("B", at24["best_draw_rate"], r24["best_draw_rate"]),
                  ("drawn", at24["n_roots_drawn_worst"], r24["n_roots_drawn_worst"])):
    if a != b:
        fails.append(("round-24 loop-vs-verifier on A", key, a, b))
if r24["roots"] != 40:
    fails.append(("the round-24 read covered %d roots" % r24["roots"], S3))
if abs(at24["random_draw_rate_exact"] - r24["random_draw_rate_exact"]) > 1e-9:
    fails.append(("round-24 loop-vs-verifier on A", "D",
                  at24["random_draw_rate_exact"], r24["random_draw_rate_exact"]))
if at24["worst_draw_rate"] == 1.0 or r24["worst_draw_rate"] == 1.0:
    fails.append(("the round-24 checkpoint certifies: the section says it does not", S3))
if r24B["worst_draw_rate"] == 1.0 or r24B["best_draw_rate"] == 1.0:
    fails.append(("the blind suite reaches a draw in some case", "seed 3 round 24"))
if s3["stop"] != "no new violations":
    fails.append(("seed 3 did not stop at the wall as the section says", s3["stop"]))
known3 = {tuple(h) for h in s3["repair"]["aggregate"]}
open_new = [v for v in fin3["violations"] if tuple(v["history"]) not in known3]
if open_new:
    fails.append(("%d violations at seed 3's stop are not in the aggregate" % len(open_new), S3))
opt3, cut3, wall3, f3 = fits_check(
    s3, S3, 20, 12,
    [(21, 1), (22, 3), (25, 5), (27, 7), (28, 34), (29, 51), (30, 44), (31, 82), (32, 167)])
fails += f3

# ---- seed 1: the arm that spent the whole cap, certified at the wall in disguise -------------
S1 = "c4_end14_t4_s1"
s1 = load(S1)
fin1, vA1, vB1, bA1, bB1, f1 = feet(s1, S1)
fails += f1
if s1.get("complete") is not True or s1["stop"] != "max_iters":
    fails.append(("seed 1 did not finish its 24 rounds at the cap", s1["stop"]))
known1 = {tuple(h) for h in s1["repair"]["aggregate"]}
new1 = [v for v in fin1["violations"] if tuple(v["history"]) not in known1]
if new1:
    fails.append(("%d violations at seed 1's last certificate are not in the aggregate - the "
                  "section says a longer cap would have stopped it on 'no new violations'"
                  % len(new1), S1))
opt1, cut1, wall1, f1 = fits_check(
    s1, S1, 15, 9, [(17, 2), (18, 2), (19, 6), (21, 6), (22, 39), (23, 45), (24, 134)])
fails += f1

# ---- seed 0: same cap, same signature as seed 1, reached in one session --------------------
S0 = "c4_end14_t4_s0"
s0 = load(S0)
fin0, vA0, vB0, bA0, bB0, f0 = feet(s0, S0)
fails += f0
if s0.get("complete") is not True or s0["stop"] != "max_iters":
    fails.append(("seed 0 did not finish its 24 rounds at the cap", s0["stop"]))
known0 = {tuple(h) for h in s0["repair"]["aggregate"]}
new0 = [v for v in fin0["violations"] if tuple(v["history"]) not in known0]
if (len(fin0["violations"]), len(new0)) != (70, 0):
    fails.append(("seed 0's last certificate holds %d violations, %d of them new; the section "
                  "says 70 with none new - a longer cap would have stopped it on 'no new "
                  "violations'" % (len(fin0["violations"]), len(new0)), S0))
opt0, cut0, wall0, f0 = fits_check(
    s0, S0, 14, 10,
    [(16, 1), (17, 2), (18, 4), (19, 6), (20, 2), (21, 19), (22, 2), (23, 12), (24, 81)])
fails += f0

# ---- seed 2: the cap with new material still on the table ---------------------------------
S2 = "c4_end14_t4_s2"
s2 = load(S2)
fin2, vA2, vB2, bA2, bB2, f2 = feet(s2, S2, blind_best=1.0)
fails += f2
if s2.get("complete") is not True or s2["stop"] != "max_iters":
    fails.append(("seed 2 did not finish its 24 rounds at the cap", s2["stop"]))
known2 = {tuple(h) for h in s2["repair"]["aggregate"]}
new2 = [v for v in fin2["violations"] if tuple(v["history"]) not in known2]
if (len(fin2["violations"]), len(new2)) != (72, 64):
    fails.append(("seed 2's last certificate holds %d violations, %d of them new; the section "
                  "says 72 with 64 new" % (len(fin2["violations"]), len(new2)), S2))
opt2, cut2, wall2, f2 = fits_check(
    s2, S2, 9, 15,
    [(11, 2), (13, 3), (14, 11), (15, 23), (16, 19), (17, 36), (18, 97), (19, 51), (20, 74),
     (21, 91), (22, 82), (23, 59), (24, 91)])
fails += f2

assert not fails, repr(fails)

# The exact values the section prints. The table carries them rounded, so without these a read
# that moved to another value would still pass every check above. The blind and base rows are
# pinned too, because the prose quotes them to two or four decimals.
assert [s["minutes"] for s in s3["provenance"]["sessions"]] == [46.9, 38.4]
assert [s["minutes"] for s in s1["provenance"]["sessions"]] == [61.2]
assert (len(s3["repair"]["log"]), len(s3["repair"]["fits"])) == (33, 32)
assert (len(s1["repair"]["log"]), len(s1["repair"]["fits"])) == (24, 24)
assert round(at24["random_draw_rate_exact"], 6) == 0.975934
assert (at24["aggregate"], at24["n_roots_drawn_worst"], at24["n_violations"]) == (2474, 29, 25)
assert round(fin3["random_draw_rate_exact"], 6) == 0.767834
assert (fin3["n_roots_drawn_worst"], fin3["n_violations"]) == (15, 91)
assert len(s3["repair"]["aggregate"]) == 3030
assert round(fin1["random_draw_rate_exact"], 6) == 0.784453
assert (fin1["n_roots_drawn_worst"], fin1["n_violations"]) == (14, 93)
assert len(s1["repair"]["aggregate"]) == 2513
best3 = max(s3["repair"]["log"], key=lambda r: r["random_draw_rate_exact"])
best1 = max(s1["repair"]["log"], key=lambda r: r["random_draw_rate_exact"])
assert (best3["round"], round(best3["random_draw_rate_exact"], 4)) == (18, 0.9782)
assert (best1["round"], round(best1["random_draw_rate_exact"], 4)) == (10, 0.9586)
assert [f["epochs"] for f in s3["repair"]["fits"]
        if f["stopped_by"] == "patience" and f["round"] <= 24] == [27, 30, 26, 27]
assert s3["repair"]["fits"][-1]["entry_mistakes"] == 196
assert s1["repair"]["fits"][-1]["entry_mistakes"] == 168
# The text says seeds 3, 1 and 0 ended by handing back the weights they were given, and that seed
# 2's last fit instead took one epoch and kept it; `kept` is the loop's own record of which
# happened.
assert s3["repair"]["fits"][-1]["kept"] == "incumbent"
assert s3["repair"]["fits"][-1]["best_mistakes"] == 196
assert s1["repair"]["fits"][-1]["kept"] == "incumbent"
assert s1["repair"]["fits"][-1]["best_mistakes"] == 168
assert (min(f["epochs"] for f in s1["repair"]["fits"] if f["stopped_by"] == "optimal"),
        max(f["epochs"] for f in s1["repair"]["fits"] if f["stopped_by"] == "optimal"),
        min(f["epochs"] for f in s1["repair"]["fits"] if f["stopped_by"] == "patience"),
        max(f["epochs"] for f in s1["repair"]["fits"] if f["stopped_by"] == "patience")) == (2, 30, 25, 40)
assert (round(r24B["random_draw_rate_exact"], 6), r24B["n_roots_drawn_worst"],
        r24B["best_draw_rate"]) == (0.055646, 0, 0.0)
assert (round(vB3["random_draw_rate_exact"], 6), vB3["n_roots_drawn_worst"],
        vB3["best_draw_rate"]) == (0.051517, 0, 0.0)
assert (round(bB3["random_draw_rate_exact"], 5), bB3["n_roots_drawn_worst"],
        bB3["best_draw_rate"]) == (0.09234, 1, 1.0)
assert (round(vB1["random_draw_rate_exact"], 6), vB1["n_roots_drawn_worst"],
        vB1["best_draw_rate"]) == (0.079244, 0, 0.0)
assert (round(bB1["random_draw_rate_exact"], 5), bB1["n_roots_drawn_worst"],
        bB1["best_draw_rate"]) == (0.05331, 1, 1.0)
assert [s["minutes"] for s in s0["provenance"]["sessions"]] == [66.8]
assert (len(s0["repair"]["log"]), len(s0["repair"]["fits"])) == (24, 24)
assert round(fin0["random_draw_rate_exact"], 6) == 0.875451
assert (fin0["n_roots_drawn_worst"], fin0["n_violations"]) == (15, 70)
assert len(s0["repair"]["aggregate"]) == 2827
best0 = max(s0["repair"]["log"], key=lambda r: r["random_draw_rate_exact"])
assert (best0["round"], round(best0["random_draw_rate_exact"], 4)) == (14, 0.9698)
assert (best0["n_roots_drawn_worst"], best0["n_violations"]) == (25, 74)
assert s0["repair"]["fits"][-1]["entry_mistakes"] == 94
assert s0["repair"]["fits"][-1]["epochs"] == 25
# The text's seed-0 sentence: it entered with 94 stop-set mistakes, no epoch beat that, and it
# returned the weights it was handed.
assert s0["repair"]["fits"][-1]["best_mistakes"] == 94
assert s0["repair"]["fits"][-1]["kept"] == "incumbent"
assert min(s0["repair"]["fits"][-1]["mistake_trace"]) >= 94
assert (round(bA0["random_draw_rate_exact"], 6), bA0["n_roots_drawn_worst"]) == (0.364652, 5)
assert (round(vB0["random_draw_rate_exact"], 6), vB0["n_roots_drawn_worst"],
        vB0["best_draw_rate"]) == (0.039229, 0, 0.0)
assert (round(bB0["random_draw_rate_exact"], 6), bB0["n_roots_drawn_worst"],
        bB0["best_draw_rate"]) == (0.086245, 1, 1.0)
assert [s["minutes"] for s in s2["provenance"]["sessions"]] == [75.1, 13.1]
assert (len(s2["repair"]["log"]), len(s2["repair"]["fits"])) == (24, 24)
assert round(fin2["random_draw_rate_exact"], 6) == 0.864226
assert (fin2["n_roots_drawn_worst"], fin2["n_violations"]) == (20, 72)
assert len(s2["repair"]["aggregate"]) == 3206
best2 = max(s2["repair"]["log"], key=lambda r: r["random_draw_rate_exact"])
assert (best2["round"], round(best2["random_draw_rate_exact"], 4)) == (11, 0.9295)
assert (best2["n_roots_drawn_worst"], best2["n_violations"]) == (22, 63)
# The text's seed-2 sentence: one epoch took 155 to 120, nothing beat 120, epoch 1 was kept.
assert s2["repair"]["fits"][-1]["entry_mistakes"] == 155
assert s2["repair"]["fits"][-1]["best_mistakes"] == 120
assert s2["repair"]["fits"][-1]["epochs"] == 26
assert s2["repair"]["fits"][-1]["kept"] == "best@1"
trace2 = s2["repair"]["fits"][-1]["mistake_trace"]
assert trace2[0] == 120 and min(trace2[1:]) == 262
assert (round(bA2["random_draw_rate_exact"], 6), bA2["n_roots_drawn_worst"]) == (0.421931, 5)
assert (round(vB2["random_draw_rate_exact"], 6), vB2["n_roots_drawn_worst"],
        vB2["best_draw_rate"]) == (0.090484, 1, 1.0)
assert (round(bB2["random_draw_rate_exact"], 6), bB2["n_roots_drawn_worst"],
        bB2["best_draw_rate"]) == (0.093742, 0, 0.0)


def pct(x):
    return "%.1f" % (100 * x)


def report(tag, arm, fin, vA, vB, bA, bB, optimal, cut, wall):
    minutes = sum(s.get("minutes", 0.0) for s in arm["provenance"]["sessions"])
    print("%s: %d certificates, %d fits, %.1f min in %d session(s), %d aggregate histories, "
          "stop=%s, complete=%s"
          % (tag, len(arm["repair"]["log"]), len(arm["repair"]["fits"]), minutes,
             len(arm["provenance"]["sessions"]), len(arm["repair"]["aggregate"]),
             arm["stop"], arm.get("complete")))
    print("  fits: %d reached exact optimality in %d-%d epochs; %d were cut by patience at %d-%d "
          "epochs, keeping an epoch with %d-%d stop-set mistakes open; none reached the 200-epoch "
          "cap"
          % (len(optimal), min(f["epochs"] for f in optimal), max(f["epochs"] for f in optimal),
             len(cut), min(f["epochs"] for f in cut), max(f["epochs"] for f in cut),
             min(f["best_mistakes"] for f in cut), max(f["best_mistakes"] for f in cut)))
    print("  aggregate regret in %d of %d fits: %s"
          % (len(wall), len(arm["repair"]["fits"]),
             ", ".join("r%d %d" % (r, m) for r, m in wall)))
    print("  base %s -> repair %s on A (gain %+.1f); blind %s -> %s (gain %+.1f), drawn %d/%d"
          % (pct(bA["random_draw_rate_exact"]), pct(vA["random_draw_rate_exact"]),
             100 * (vA["random_draw_rate_exact"] - bA["random_draw_rate_exact"]),
             pct(bB["random_draw_rate_exact"]), pct(vB["random_draw_rate_exact"]),
             100 * (vB["random_draw_rate_exact"] - bB["random_draw_rate_exact"]),
             vB["n_roots_drawn_worst"], vB["roots"]))


minutes3 = sum(s.get("minutes", 0.0) for s in s3["provenance"]["sessions"])
print("%s is the arm in the text: %d certificates, %d fits, %.1f min, stop=%s; the eight refits "
      "after round 24 took the aggregate from %d to %d histories, the drawn roots from %d to %d "
      "and the violations from %d to %d"
      % (S3, len(s3["repair"]["log"]), len(s3["repair"]["fits"]), minutes3, s3["stop"],
         at24["aggregate"], len(s3["repair"]["aggregate"]), at24["n_roots_drawn_worst"],
         fin3["n_roots_drawn_worst"], at24["n_violations"], fin3["n_violations"]))
report(S3, s3, fin3, vA3, vB3, bA3, bB3, opt3, cut3, wall3)
report(S1, s1, fin1, vA1, vB1, bA1, bB1, opt1, cut1, wall1)
report(S0, s0, fin0, vA0, vB0, bA0, bB0, opt0, cut0, wall0)
report(S2, s2, fin2, vA2, vB2, bA2, bB2, opt2, cut2, wall2)
print()
print("Policy & Suite & $W$ & $B$ & $\\bar D$ & drawn & viol. \\\\")
print("s3: Base & A (published) & $%g$ & $%g$ & %s & %d & %d \\\\"
      % (bA3["worst_draw_rate"], bA3["best_draw_rate"],
         pct(bA3["random_draw_rate_exact"]), bA3["n_roots_drawn_worst"],
         s3["base"]["certificate"]["n_violations"]))
print("s3: Repair r24 loop & A & $%g$ & $%g$ & %s & %d & %d \\\\"
      % (at24["worst_draw_rate"], at24["best_draw_rate"],
         pct(at24["random_draw_rate_exact"]), at24["n_roots_drawn_worst"],
         at24["n_violations"]))
print("s3: Repair r24 verifier & A & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (r24["worst_draw_rate"], r24["best_draw_rate"],
         pct(r24["random_draw_rate_exact"]), r24["n_roots_drawn_worst"]))
print("s3: Repair r32 loop & A & $%g$ & $%g$ & %s & %d & %d \\\\"
      % (fin3["worst_draw_rate"], fin3["best_draw_rate"],
         pct(fin3["random_draw_rate_exact"]), fin3["n_roots_drawn_worst"],
         fin3["n_violations"]))
print("s3: Repair r32 verifier & A & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (vA3["worst_draw_rate"], vA3["best_draw_rate"],
         pct(vA3["random_draw_rate_exact"]), vA3["n_roots_drawn_worst"]))
print("s1: Base & A (published) & $%g$ & $%g$ & %s & %d & %d \\\\"
      % (bA1["worst_draw_rate"], bA1["best_draw_rate"],
         pct(bA1["random_draw_rate_exact"]), bA1["n_roots_drawn_worst"],
         s1["base"]["certificate"]["n_violations"]))
print("s1: Repair r24 loop & A & $%g$ & $%g$ & %s & %d & %d \\\\"
      % (fin1["worst_draw_rate"], fin1["best_draw_rate"],
         pct(fin1["random_draw_rate_exact"]), fin1["n_roots_drawn_worst"],
         fin1["n_violations"]))
print("s1: Repair r24 verifier & A & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (vA1["worst_draw_rate"], vA1["best_draw_rate"],
         pct(vA1["random_draw_rate_exact"]), vA1["n_roots_drawn_worst"]))
print("s0: Base & A (published) & $%g$ & $%g$ & %s & %d & %d \\\\"
      % (bA0["worst_draw_rate"], bA0["best_draw_rate"],
         pct(bA0["random_draw_rate_exact"]), bA0["n_roots_drawn_worst"],
         s0["base"]["certificate"]["n_violations"]))
print("s0: Repair r24 loop & A & $%g$ & $%g$ & %s & %d & %d \\\\"
      % (fin0["worst_draw_rate"], fin0["best_draw_rate"],
         pct(fin0["random_draw_rate_exact"]), fin0["n_roots_drawn_worst"],
         fin0["n_violations"]))
print("s0: Repair r24 verifier & A & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (vA0["worst_draw_rate"], vA0["best_draw_rate"],
         pct(vA0["random_draw_rate_exact"]), vA0["n_roots_drawn_worst"]))
print("s2: Base & A (published) & $%g$ & $%g$ & %s & %d & %d \\\\"
      % (bA2["worst_draw_rate"], bA2["best_draw_rate"],
         pct(bA2["random_draw_rate_exact"]), bA2["n_roots_drawn_worst"],
         s2["base"]["certificate"]["n_violations"]))
print("s2: Repair r24 loop & A & $%g$ & $%g$ & %s & %d & %d \\\\"
      % (fin2["worst_draw_rate"], fin2["best_draw_rate"],
         pct(fin2["random_draw_rate_exact"]), fin2["n_roots_drawn_worst"],
         fin2["n_violations"]))
print("s2: Repair r24 verifier & A & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (vA2["worst_draw_rate"], vA2["best_draw_rate"],
         pct(vA2["random_draw_rate_exact"]), vA2["n_roots_drawn_worst"]))
print("\\midrule")
print("s3: Base verifier & B (blind) & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (bB3["worst_draw_rate"], bB3["best_draw_rate"],
         pct(bB3["random_draw_rate_exact"]), bB3["n_roots_drawn_worst"]))
print("s3: Repair r24 verifier & B & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (r24B["worst_draw_rate"], r24B["best_draw_rate"],
         pct(r24B["random_draw_rate_exact"]), r24B["n_roots_drawn_worst"]))
print("s3: Repair r32 verifier & B & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (vB3["worst_draw_rate"], vB3["best_draw_rate"],
         pct(vB3["random_draw_rate_exact"]), vB3["n_roots_drawn_worst"]))
print("s1: Base verifier & B (blind) & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (bB1["worst_draw_rate"], bB1["best_draw_rate"],
         pct(bB1["random_draw_rate_exact"]), bB1["n_roots_drawn_worst"]))
print("s1: Repair r24 verifier & B & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (vB1["worst_draw_rate"], vB1["best_draw_rate"],
         pct(vB1["random_draw_rate_exact"]), vB1["n_roots_drawn_worst"]))
print("s0: Base verifier & B (blind) & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (bB0["worst_draw_rate"], bB0["best_draw_rate"],
         pct(bB0["random_draw_rate_exact"]), bB0["n_roots_drawn_worst"]))
print("s0: Repair r24 verifier & B & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (vB0["worst_draw_rate"], vB0["best_draw_rate"],
         pct(vB0["random_draw_rate_exact"]), vB0["n_roots_drawn_worst"]))
print("s2: Base verifier & B (blind) & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (bB2["worst_draw_rate"], bB2["best_draw_rate"],
         pct(bB2["random_draw_rate_exact"]), bB2["n_roots_drawn_worst"]))
print("s2: Repair r24 verifier & B & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (vB2["worst_draw_rate"], vB2["best_draw_rate"],
         pct(vB2["random_draw_rate_exact"]), vB2["n_roots_drawn_worst"]))
