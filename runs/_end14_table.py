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
* the arms must still be uncertified (W=0), and the blind suite must still read W=0 with B=0 at
  every repaired checkpoint - if a later session certifies one, this script fails so the section
  is rewritten rather than left saying "it does not certify" over a certificate;
* seed 3 must have stopped the way the section says it stopped: at the wall, with every violating
  history already in the aggregate, which is what leaves the loop with no move left;
* seed 1 must have stopped at the 24-round cap with every violating history of its final
  certificate already in the aggregate - the section says a longer cap would have stopped it one
  round later on "no new violations", and that is exactly this check;
* the refit rule the notebook adds must still be the one the JSON records: the stated numbers of
  fits reach exact optimality and are cut by patience, the aggregate regret is the list the text
  prints, and none of the fits reached the 200-epoch cap.

    python runs/_end14_table.py
"""
import json
import pathlib

R = pathlib.Path("data")
SUITE = "data/c4_endgame_roots_14_40_s0.json"
SUITE_B = "data/c4_endgame_roots_14_40_holdoutB.json"


def load(name):
    return json.loads((R / (name + ".json")).read_text(encoding="utf-8"))


def feet(arm, tag):
    """The checks every arm shares: base and final read on A, no certificate, blind at zero."""
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
    if vB["worst_draw_rate"] == 1.0 or vB["best_draw_rate"] == 1.0:
        out.append(("the blind suite reaches a draw in some case", tag))
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
