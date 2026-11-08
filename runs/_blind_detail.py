import json, pathlib

R = pathlib.Path("data")


def load(p):
    f = R / p
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def cert(d):
    return d["certificate"] if "certificate" in d else d


A_NAME = {8: {0: "c4_end_verify_s0.json", 1: "c4_end_s1.json", 2: "c4_end_s2.json"},
          10: {s: "c4_end10_verify_s%d.json" % s for s in (0, 1, 2)},
          12: {s: "c4_end12_verify_s%d.json" % s for s in range(7)}}


def fmt(c, seed, label):
    print("  %s s%d: W %.3f B %.3f Dbar %6.4f drawn %2d/40" % (
        label, seed, c["worst_draw_rate"], c["best_draw_rate"],
        c["random_draw_rate_exact"], c["n_roots_drawn_worst"]))


for E in (8, 10, 12):
    print("=== %d ===" % E)
    for s in range(0, 7):
        a = A_NAME.get(E, {}).get(s)
        d = load(a) if a else None
        if d is None:
            continue
        c = cert(d["repair"]["final"]) if "repair" in d else cert(d)
        fmt(c, s, "A")
        b = load("c4_end%d_holdoutB_s%d.json" % (E, s))
        if b:
            fmt(cert(b), s, "B")
        for w in ("A", "B"):
            bd = load("c4_end%d_base_s%d_on%s.json" % (E, s, w))
            if bd:
                fmt(cert(bd), s, "base" + w)
