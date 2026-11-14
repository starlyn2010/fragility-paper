"""Emit and check the rows of `tab:end14`: the fourteen-empty arm run on a T4.

Every cell comes from the JSONs of that arm, and the script refuses to print the table if the
two implementations disagree or if the claims the text makes about the arm stop holding:

* the loop's base certificate and the memo-free read of `c4_gru_base.pt` on suite A must be the
  same number, which is what makes the base row a measurement and the gain a difference;
* the loop's certificate at the end of the first session (round 24, where the cap stopped it) and
  the memo-free read taken then - kept as `_r24_` when the arm was resumed - must agree;
* the loop's final certificate and the memo-free read of the repaired weights on suite A must
  agree to 1e-9, root counts included;
* the arm must still be uncertified (W=0) at both checkpoints, and the blind suite must still
  read W=0 with B=0 at both - if a later session certifies it, this script fails so the section
  is rewritten rather than left saying "it does not certify" over a certificate;
* the loop must have stopped the way the section says it stopped: at the wall, with every
  violating history already in the aggregate, which is what leaves the loop with no move left;
* the refit rule the notebook adds must still be the one the JSON records: some fits reach exact
  optimality, the rest were cut by patience, and none of them reached the 200-epoch cap.

    python runs/_end14_table.py
"""
import json
import pathlib

R = pathlib.Path("data")
ARM = "c4_end14_t4_s3"
SUITE = "data/c4_endgame_roots_14_40_s0.json"
SUITE_B = "data/c4_endgame_roots_14_40_holdoutB.json"


def load(name):
    return json.loads((R / (name + ".json")).read_text(encoding="utf-8"))


arm = load(ARM)
fin = arm["repair"]["final"]
base = arm["base"]["certificate"]
vA = load(ARM + "_verify")
vB = load(ARM + "_holdoutB")
r24 = load(ARM + "_r24_verify")
r24B = load(ARM + "_r24_holdoutB")
bA = load(ARM + "_base_onA")
bB = load(ARM + "_base_onB")
fits = arm["repair"]["fits"]
prot = arm["protocol"]
# The round-24 state is the loop's own certificate of that round, and the two `_r24_` files are
# the independent reads the first session took of the same weights before the arm was resumed.
at24 = next(r for r in arm["repair"]["log"] if r["round"] == 24)

fails = []
if arm["args"]["roots_file"] != SUITE:
    fails.append(("the arm did not read the published suite", arm["args"]["roots_file"]))
if prot.get("published") is not False or not prot.get("keep_best"):
    fails.append(("the arm does not carry the rule the section describes", prot))

for key, a, b in (("W", base["worst_draw_rate"], bA["worst_draw_rate"]),
                 ("B", base["best_draw_rate"], bA["best_draw_rate"]),
                 ("drawn", base["n_roots_drawn_worst"], bA["n_roots_drawn_worst"])):
    if a != b:
        fails.append(("base loop-vs-verifier on A", key, a, b))
if abs(base["random_draw_rate_exact"] - bA["random_draw_rate_exact"]) > 1e-9:
    fails.append(("base loop-vs-verifier on A", "D",
                  base["random_draw_rate_exact"], bA["random_draw_rate_exact"]))

for key, a, b in (("W", at24["worst_draw_rate"], r24["worst_draw_rate"]),
                  ("B", at24["best_draw_rate"], r24["best_draw_rate"]),
                  ("drawn", at24["n_roots_drawn_worst"], r24["n_roots_drawn_worst"])):
    if a != b:
        fails.append(("round-24 loop-vs-verifier on A", key, a, b))
if r24["roots"] != 40:
    fails.append(("the round-24 read covered %d roots" % r24["roots"], ARM))
if abs(at24["random_draw_rate_exact"] - r24["random_draw_rate_exact"]) > 1e-9:
    fails.append(("round-24 loop-vs-verifier on A", "D",
                  at24["random_draw_rate_exact"], r24["random_draw_rate_exact"]))

for key, a, b in (("W", fin["worst_draw_rate"], vA["worst_draw_rate"]),
                  ("B", fin["best_draw_rate"], vA["best_draw_rate"]),
                  ("drawn", fin["n_roots_drawn_worst"], vA["n_roots_drawn_worst"]),
                  ("roots", fin["n_roots"], vA["roots"])):
    if a != b:
        fails.append(("repair loop-vs-verifier on A", key, a, b))
if abs(fin["random_draw_rate_exact"] - vA["random_draw_rate_exact"]) > 1e-9:
    fails.append(("repair loop-vs-verifier on A", "D",
                  fin["random_draw_rate_exact"], vA["random_draw_rate_exact"]))

if fin["worst_draw_rate"] == 1.0 or vA["worst_draw_rate"] == 1.0:
    fails.append(("the arm certifies: the section says it does not", ARM))
if at24["worst_draw_rate"] == 1.0 or r24["worst_draw_rate"] == 1.0:
    fails.append(("the round-24 checkpoint certifies: the section says it does not", ARM))
for name, blind in (("round 24", r24B), ("final", vB)):
    if blind["worst_draw_rate"] == 1.0 or blind["best_draw_rate"] == 1.0:
        fails.append(("the blind suite reaches a draw in some case", name))
if bA["worst_draw_rate"] == 1.0 or bB["worst_draw_rate"] == 1.0:
    fails.append(("an unrepaired base certifies", ARM))

if arm["stop"] != "no new violations":
    fails.append(("the loop did not stop at the wall as the section says", arm["stop"]))
known = {tuple(h) for h in arm["repair"]["aggregate"]}
open_new = [v for v in fin["violations"] if tuple(v["history"]) not in known]
if open_new:
    fails.append(("%d violations at the stop are not in the aggregate" % len(open_new), ARM))

optimal = [f for f in fits if f["stopped_by"] == "optimal"]
cut = [f for f in fits if f["stopped_by"] == "patience"]
if not optimal or len(optimal) + len(cut) != len(fits):
    fails.append(("the rule did not behave as the section says: %d optimal, %d patience, %d fits"
                  % (len(optimal), len(cut), len(fits))))
if any(f["stopped_by"] == "cap" for f in fits):
    fails.append(("a fit ran to its 200-epoch cap", ARM))

assert not fails, repr(fails)


def pct(x):
    return "%.1f" % (100 * x)


wall = [f for f in fits if f["agg_mistakes"] > 0]
minutes = sum(s.get("minutes", 0.0) for s in arm["provenance"]["sessions"])

print("%s: %d certificates, %d fits, %.1f min in %d sessions, %d aggregate histories, stop=%s"
      % (ARM, len(arm["repair"]["log"]), len(fits), minutes,
         len(arm["provenance"]["sessions"]), len(arm["repair"]["aggregate"]), arm["stop"]))
print("fits: %d reached exact optimality in %d-%d epochs; %d were cut by patience at %d-%d "
      "epochs, keeping an epoch with %d-%d stop-set mistakes open; none reached the 200-epoch cap"
      % (len(optimal), min(f["epochs"] for f in optimal), max(f["epochs"] for f in optimal),
         len(cut), min(f["epochs"] for f in cut), max(f["epochs"] for f in cut),
         min(f["best_mistakes"] for f in cut), max(f["best_mistakes"] for f in cut)))
print("aggregate regret in %d of %d fits: %s"
      % (len(wall), len(fits), ", ".join("r%d %d" % (f["round"], f["agg_mistakes"]) for f in wall)))
print("patience rounds: " + ", ".join("r%d %d eps entry %d best %d kept %s"
                                     % (f["round"], f["epochs"], f["entry_mistakes"],
                                        f["best_mistakes"], f["kept"]) for f in cut))
print("base %s -> round 24 %s -> final %s on A; blind %s -> %s -> %s"
      % (pct(bA["random_draw_rate_exact"]), pct(r24["random_draw_rate_exact"]),
         pct(vA["random_draw_rate_exact"]), pct(bB["random_draw_rate_exact"]),
         pct(r24B["random_draw_rate_exact"]), pct(vB["random_draw_rate_exact"])))
print("the eight refits after round 24 took the aggregate from %d to %d histories, the drawn "
      "roots from %d to %d and the violations from %d to %d"
      % (at24["aggregate"], len(arm["repair"]["aggregate"]), at24["n_roots_drawn_worst"],
         fin["n_roots_drawn_worst"], at24["n_violations"], fin["n_violations"]))
print()
print("Policy & Suite & $W$ & $B$ & $\\bar D$ & drawn & viol. \\\\")
print("Base & A (published) & $%g$ & $%g$ & %s & %d & %d \\\\"
      % (bA["worst_draw_rate"], bA["best_draw_rate"],
         pct(bA["random_draw_rate_exact"]), bA["n_roots_drawn_worst"], base["n_violations"]))
print("Repair, round 24, loop & A & $%g$ & $%g$ & %s & %d & %d \\\\"
      % (at24["worst_draw_rate"], at24["best_draw_rate"],
         pct(at24["random_draw_rate_exact"]), at24["n_roots_drawn_worst"],
         at24["n_violations"]))
print("Repair, round 24, verifier & A & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (r24["worst_draw_rate"], r24["best_draw_rate"],
         pct(r24["random_draw_rate_exact"]), r24["n_roots_drawn_worst"]))
print("Repair, round 32, loop & A & $%g$ & $%g$ & %s & %d & %d \\\\"
      % (fin["worst_draw_rate"], fin["best_draw_rate"],
         pct(fin["random_draw_rate_exact"]), fin["n_roots_drawn_worst"], fin["n_violations"]))
print("Repair, round 32, verifier & A & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (vA["worst_draw_rate"], vA["best_draw_rate"],
         pct(vA["random_draw_rate_exact"]), vA["n_roots_drawn_worst"]))
print("Base, verifier & B (blind) & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (bB["worst_draw_rate"], bB["best_draw_rate"],
         pct(bB["random_draw_rate_exact"]), bB["n_roots_drawn_worst"]))
print("Repair, round 24, verifier & B & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (r24B["worst_draw_rate"], r24B["best_draw_rate"],
         pct(r24B["random_draw_rate_exact"]), r24B["n_roots_drawn_worst"]))
print("Repair, round 32, verifier & B & $%g$ & $%g$ & %s & %d & -- \\\\"
      % (vB["worst_draw_rate"], vB["best_draw_rate"],
         pct(vB["random_draw_rate_exact"]), vB["n_roots_drawn_worst"]))
