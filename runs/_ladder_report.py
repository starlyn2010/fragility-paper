import json

for s in (3, 4, 5, 6):
    d = json.load(open("data/c4_end12_s%d.json" % s, encoding="utf-8"))
    log = d["repair"]["log"]
    print("--- seed %d | arm %d s | repair %d s | certificates %d | added %d"
          % (s, d["elapsed_s"], d["repair"]["elapsed_s"], len(log), len(d["repair"]["aggregate"])))
    print("  base (suite A):  W %.2f  B %.2f  Dbar %.1f%%  viol %d  drawn %d/40"
          % (log[0]["worst_draw_rate"], log[0]["best_draw_rate"],
             100 * log[0]["random_draw_rate_exact"], log[0]["n_violations"],
             log[0]["n_roots_drawn_worst"]))
    print("  violations per certificate:", [r["n_violations"] for r in log])
    print("  Dbar per certificate:      ", [round(100 * r["random_draw_rate_exact"], 1) for r in log])
    print("  replay mistakes:", set(r["replay_mistakes"] for r in log),
          "| aggregate grows:", all(log[i]["aggregate"] < log[i + 1]["aggregate"]
                                    for i in range(len(log) - 1)))
    f = d["repair"]["final"]
    print("  final (suite A): W %.2f  B %.2f  Dbar %.2f%%  viol %d  drawn %d/40"
          % (f["worst_draw_rate"], f["best_draw_rate"], 100 * f["random_draw_rate_exact"],
             f["n_violations"], f["n_roots_drawn_worst"]))

print()
for s in (3, 4, 5, 6):
    a = json.load(open("data/c4_end12_verify_s%d.json" % s, encoding="utf-8"))
    b = json.load(open("data/c4_end12_holdoutB_s%d.json" % s, encoding="utf-8"))
    print("seed %d  A: W %.2f B %.2f drawn %d/40 order_mean %.3f |  B: W %.2f B %.2f drawn %d/40"
          " order_mean %.3f order_min %.3f adaptive_mean %.3f"
          % (s, a["worst_draw_rate"], a["best_draw_rate"], a["n_roots_drawn_worst"],
             a["order_mean"], b["worst_draw_rate"], b["best_draw_rate"],
             b["n_roots_drawn_worst"], b["order_mean"], b["order_min"], b["adaptive_mean"]))
