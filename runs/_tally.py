import json, pathlib, math

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

for E in (8, 10, 12):
    rows = []
    for p in sorted(pathlib.Path("data").glob("c4_end%d_s*.json" % E)):
        if not p.stem.endswith(tuple("0123456789")):
            continue
        s = int(p.stem.split("_s")[1])
        d = json.loads(p.read_text(encoding="utf-8"))
        fin = d["repair"]["final"]
        log = d["repair"]["log"]
        W = fin["worst_draw_rate"]
        v = fin.get("n_violations")
        rows.append((s, W, fin["best_draw_rate"], fin["random_draw_rate_exact"],
                     len(log) - 1, v, d["repair"]["elapsed_s"]))
    k = sum(1 for r in rows if r[1] == 1.0)
    print("=== %d empties: %d/%d certify" % (E, k, len(rows)))
    for r in sorted(rows):
        print("  s%-2d W %.2f B %.2f Dbar %.4f rounds %2d viol %s %5.0fs"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    if rows:
        lo, hi = wilson(k, len(rows))
        print("  wilson %.3f-%.3f (width %.2f)" % (lo, hi, hi - lo))
