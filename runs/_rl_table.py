"""Emit and check the reward-trained $4\\times4$ arm the paper reports next to the distilled one.

`data/rl_c4x4_s0.json` is the REINFORCE base, `data/c4_rl_repair_x4_s0.json` the repair run that
produced the paragraph, and `data/c4_rl_repair_x4_s0_rerun.json` the same arm repeated from
scratch with a checkpoint before every refit. The paragraph prints numbers no table carries, so
nothing else guards them; this script refuses to print if they move.

What it holds, and why each check has teeth:

* the base record must still be the reward signal the text describes - no optimal sets, 5{,}000
  games, hidden 32, 4{,}245 parameters - and the certificate must still be W=0, B=1, D=16.52%,
  108 violations and 440/1,085 frontier states, with 8 of the 24 orders drawn and 14,520 of the
  66,297 boards mislabelled. A weaker or stronger base is a different experiment;
* the repair must still be the full-budget arm: 40 certificates, `stop_reason` a cap and not a
  certificate, final W=0, D=91.79%, 4,178 states, 1,705 on the frontier, 17 violations, 1,088
  aggregate histories that include all 108 base violations, and the best certificate at round 27
  with 4 violations;
* the sentences about the fit must still be true: 38 of the 40 logged certificates find aggregate
  regret between 1 and 92, the replay never drops below 3,446 mistakes, and - the contrast the
  paragraph draws - all 34 certificates of the distilled arm find aggregate regret 0;
* the replication claim is a per-round equality of every logged field, not of the summary: the
  two repair files are compared field by field over all 40 rounds, and the only numbers allowed
  to differ are the wall clocks the text reports.

    python runs/_rl_table.py
"""
import json
import pathlib

R = pathlib.Path("data")


def load(name):
    return json.loads((R / (name + ".json")).read_text(encoding="utf-8"))


rl = load("rl_c4x4_s0")
arm = load("c4_rl_repair_x4_s0")
rerun = load("c4_rl_repair_x4_s0_rerun")
distilled = load("c4x4_ctrl_s0")

fails = []

# ---- the reward-trained base: what the paragraph says it is, and how far it starts ----------
if rl["args"]["games"] != 5000 or rl["args"]["hidden"] != 32 or rl["args"]["emb"] != 8:
    fails.append(("the base is not the arm the text describes", rl["args"]))
if rl["parameters"]["rl"] != 4245:
    fails.append(("base parameter count", rl["parameters"]))
if "no optimal sets" not in rl["signal"]:
    fails.append(("the training signal changed", rl["signal"]))
if round(rl["elapsed_s"], 1) != 1756.6:
    fails.append(("base training seconds", rl["elapsed_s"]))
if (rl["game"]["learner_boards"], rl["game"]["tie_orders"]) != (66297, 24):
    fails.append(("the board counts moved", rl["game"]))
cert = rl["base"]["cert"]
if (cert["worst_draw_rate"], cert["best_draw_rate"], cert["n_violations"]) != (0.0, 1.0, 108):
    fails.append(("base certificate W/B/violations", cert["worst_draw_rate"],
                  cert["best_draw_rate"], cert["n_violations"]))
if round(cert["random_draw_rate_exact"], 6) != 0.16522:
    fails.append(("base certificate D", cert["random_draw_rate_exact"]))
if (cert["n_states"], cert["n_frontier"]) != (1085, 440):
    fails.append(("base certificate size", cert["n_states"], cert["n_frontier"]))
if (rl["base"]["teacher_orders_drawn"], rl["base"]["board_mistakes"]) != (8, 14520):
    fails.append(("the paragraph prints 8/24 orders and 14,520 mislabelled boards",
                  rl["base"]["teacher_orders_drawn"], rl["base"]["board_mistakes"]))
if round(rl["base"]["mean_draw_rate_vs_deterministic_orders"], 4) != 0.3333:
    fails.append(("mean over the 24 orders", rl["base"]["mean_draw_rate_vs_deterministic_orders"]))

# ---- the repair: same starting weights, full budget, no certificate -------------------------
for tag, d in (("archived", arm), ("rerun", rerun)):
    if d["args"]["init"] != "data/rl_c4x4_s0.pt":
        fails.append(("the arm did not start from the base checkpoint", tag, d["args"]["init"]))
    if (d["args"]["stop"], d["args"]["max_iters"]) != ("both", 40):
        fails.append(("the refit rule or the cap moved", tag, d["args"]["stop"],
                      d["args"]["max_iters"]))
    b = d["base"]["certificate"]
    if (b["worst_draw_rate"], b["best_draw_rate"], b["n_violations"]) != (0.0, 1.0, 108):
        fails.append(("the repair did not start from the certificate the text prints", tag))

if rerun["repair"]["stop_reason"] != "max_iters":
    fails.append(("the arm no longer ends at the cap", rerun["repair"]["stop_reason"]))
fin = rerun["repair"]["final"]
if (fin["worst_draw_rate"], fin["best_draw_rate"], fin["n_violations"]) != (0.0, 1.0, 17):
    fails.append(("final certificate W/B/violations", fin["worst_draw_rate"],
                  fin["best_draw_rate"], fin["n_violations"]))
if round(fin["random_draw_rate_exact"], 6) != 0.917933:
    fails.append(("final certificate D", fin["random_draw_rate_exact"]))
if (fin["n_states"], fin["n_frontier"]) != (4178, 1705):
    fails.append(("final certificate size", fin["n_states"], fin["n_frontier"]))

log = rerun["repair"]["log"]
if len(log) != 40 or [r["round"] for r in log] != list(range(40)):
    fails.append(("the arm no longer logs 40 rounds", len(log)))
best = min(log, key=lambda r: r["n_violations"])
if (best["round"], best["n_violations"]) != (27, 4):
    fails.append(("the best certificate is no longer round 27 with 4 violations",
                  best["round"], best["n_violations"]))
if log[-1]["n_violations"] != 18:
    fails.append(("the last logged certificate", log[-1]["n_violations"]))

agg = rerun["repair"]["aggregate"]
if len(agg) != 1088:
    fails.append(("aggregate size", len(agg)))
known = {tuple(h) for h in agg}
missing = [v for v in arm["base"]["certificate"]["violations"] if tuple(v["history"]) not in known]
if missing:
    fails.append(("%d base violations never entered the aggregate" % len(missing)))

# ---- the fit: the sentence the progress guarantee turns on ----------------------------------
regret = [(r["round"], r["aggregate_mistakes"]) for r in log if r["aggregate_mistakes"]]
if len(regret) != 38:
    fails.append(("the text says 38 rounds carry aggregate regret, the log has %d"
                  % len(regret)))
if (min(m for _, m in regret), max(m for _, m in regret)) != (1, 92):
    fails.append(("the text says 1 to 92 aggregate mistakes",
                  min(m for _, m in regret), max(m for _, m in regret)))
if min(r["replay_mistakes"] for r in log) < 3446:
    fails.append(("the replay fell below the 3,446 the text prints",
                  min(r["replay_mistakes"] for r in log)))
dlog = distilled["repair"]["log"]
if len(dlog) != 34 or any(r["aggregate_mistakes"] for r in dlog):
    fails.append(("the distilled arm's 34 rounds no longer fit their aggregate to zero",
                  len(dlog), max(r["aggregate_mistakes"] for r in dlog)))

# ---- the replication: every logged field of all 40 rounds, not the summary -----------------
for i, (a, b) in enumerate(zip(arm["repair"]["log"], log)):
    if a != b:
        fails.append(("round %d of the rerun is not the archived round" % i,
                      sorted(k for k in a if a.get(k) != b.get(k))))
if arm["repair"]["final"] != fin:
    fails.append(("the rerun's final certificate is not the archived one",
                  arm["repair"]["final"], fin))
if len(arm["repair"]["aggregate"]) != len(agg):
    fails.append(("the rerun's aggregate is not the archived one",
                  len(arm["repair"]["aggregate"]), len(agg)))
first, second = arm["repair"]["elapsed_s"], rerun["repair"]["elapsed_s"]
if (round(first), round(second)) != (85038, 63549):
    fails.append(("the two wall clocks the text prints", first, second))

assert not fails, repr(fails)


def pct(x):
    return "%.1f" % (100 * x)


print("rl_c4x4_s0: %d games, %d parameters, %.0f s; %d/%d deterministic orders draw, "
      "%d of %d boards mislabelled" % (rl["args"]["games"], rl["parameters"]["rl"],
                                       rl["elapsed_s"], rl["base"]["teacher_orders_drawn"],
                                       rl["game"]["tie_orders"], rl["base"]["board_mistakes"],
                                       rl["game"]["learner_boards"]))
print("c4_rl_repair_x4_s0: base W=%g B=%g D=%s%%, %d violations, %d/%d frontier states"
      % (cert["worst_draw_rate"], cert["best_draw_rate"], pct(cert["random_draw_rate_exact"]),
         cert["n_violations"], cert["n_frontier"], cert["n_states"]))
print("  repair: %d certificates in %.0f s, stop=%s, final W=%g B=%g D=%s%%, %d violations, "
      "%d states, %d on the frontier, %d aggregate histories"
      % (len(arm["repair"]["log"]), first, rerun["repair"]["stop_reason"],
         fin["worst_draw_rate"], fin["best_draw_rate"], pct(fin["random_draw_rate_exact"]),
         fin["n_violations"], fin["n_states"], fin["n_frontier"], len(agg)))
after = [r["replay_mistakes"] for r in log[1:]]
print("  aggregate regret in %d of 40 certificates (%d-%d mistakes open), replay %d-%d "
      "after the base; the distilled arm: regret 0 in all %d"
      % (len(regret), min(m for _, m in regret), max(m for _, m in regret),
         min(after), max(after), len(dlog)))
print("  rerun: %.0f s, all 40 logged rounds identical (best round %d with %d violations)"
      % (second, best["round"], best["n_violations"]))
print()
print("Round & violations & states & frontier & $\\bar D$ & aggregate & agg.\\ mistakes \\\\")
for r in (log[0], best, log[-1]):
    print("r%d & %d & %d & %d & %s & %d & %d \\\\"
          % (r["round"], r["n_violations"], r["n_states"], r["n_frontier"],
             pct(r["random_draw_rate_exact"]), r["aggregate"], r["aggregate_mistakes"]))
