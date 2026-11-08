import json, pathlib, math

R = pathlib.Path(".")


def g(p):
    return json.loads((R / p).read_text(encoding="utf-8")) if (R / p).exists() else None


for E in (12,):
    print("=== %d empties ===" % E)
    for s in range(0, 7):
        d = g("data/c4_end%d_s%d.json" % (E, s))
        if d is None:
            continue
        rep, base = d["repair"], d["base"]
        log = rep["log"]
        agg = d["repair"]["aggregate"]
        bf = base["certificate"]
        print("seed %d | learner sample %s constrained %s | base W %.0f B %.0f Dbar %.4f "
              "viol %d drawn %d" % (
                  s, d["repair"].get("replay_histories"), "?",
                  bf["worst_draw_rate"], bf["best_draw_rate"],
                  bf["random_draw_rate_exact"], bf["n_violations"],
                  bf["n_roots_drawn_worst"]))
        print("   rounds %d | added %d | final W %.2f Dbar %.4f viol %s drawn %s | "
              "rep_mis max %d | elapsed %.0fs" % (
                  len(log) - 1, len(agg), rep["final"]["worst_draw_rate"],
                  rep["final"]["random_draw_rate_exact"],
                  rep["final"]["n_violations"], rep["final"]["n_roots_drawn_worst"],
                  max([r["replay_mistakes"] for r in log] or [0]), rep["elapsed_s"]))
        print("   viol by round:", [r["n_violations"] for r in log])
        vA = g("data/c4_end%d_verify_s%d.json" % (E, s))
        vB = g("data/c4_end%d_holdoutB_s%d.json" % (E, s))
        if vA:
            print("   verify A: W %.2f B %.2f Dbar %.4f drawn %d/%d orders %d [%.3f-%.3f] "
                  "adaptive [%.3f-%.3f]" % (vA["worst_draw_rate"], vA["best_draw_rate"],
                                            vA["random_draw_rate_exact"],
                                            vA["n_roots_drawn_worst"], vA["roots"],
                                            vA["deterministic_orders"], vA["order_min"],
                                            vA["order_max"], vA["adaptive_min"],
                                            vA["adaptive_mean"]))
        if vB:
            print("   verify B: W %.2f B %.2f Dbar %.4f drawn %d/%d orders %d [%.3f-%.3f] "
                  "adaptive [%.3f-%.3f]" % (vB["worst_draw_rate"], vB["best_draw_rate"],
                                            vB["random_draw_rate_exact"],
                                            vB["n_roots_drawn_worst"], vB["roots"],
                                            vB["deterministic_orders"], vB["order_min"],
                                            vB["order_max"], vB["adaptive_min"],
                                            vB["adaptive_mean"]))
for E in (8, 10):
    print("=== %d empties: base on B ===" % E)
    for s in range(0, 3):
        b = g("data/c4_end%d_base_s%d_onB.json" % (E, s))
        print("  base onB s%d" % s, json.dumps(b)[:200] if b else "missing")
