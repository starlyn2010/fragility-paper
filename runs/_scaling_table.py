"""Emit the LaTeX rows of `tab:scaling76` (the CGR scaling ladder) straight from the run JSONs.

Every cell comes from disk: rounds and added histories from `repair`, the base and final
certificates from `base.certificate` and `repair.final`, and the controls column from how many
matched random-aggregation arms on that seed reached W=1. Rounds follow the published rule,
asserted below against all thirteen rows that were already in the table: a log row carries the
certificate *before* its refit, so an arm that broke on a certificate logs one row more than it
refitted and is counted at len(log)-1, while an arm the cap stopped is counted at the cap.

    python runs/_scaling_table.py
"""
import json
import pathlib

R = pathlib.Path("data")

ARMS = [(8, 0, "c4_end_ctrl_s0"), (8, 1, "c4_end_s1"), (8, 2, "c4_end_s2"),
        (8, 3, "c4_end8_s3"), (8, 4, "c4_end8_s4"), (8, 5, "c4_end8_s5"),
        (10, 0, "c4_end10_s0"), (10, 1, "c4_end10_s1"), (10, 2, "c4_end10_s2"),
        (10, 3, "c4_end10_s3"), (10, 4, "c4_end10_s4"), (10, 5, "c4_end10_s5"),
        (12, 0, "c4_end12_s0"), (12, 1, "c4_end12_s1"), (12, 2, "c4_end12_s2"),
        (12, 3, "c4_end12_s3"), (12, 4, "c4_end12_s4"), (12, 5, "c4_end12_s5"),
        (12, 6, "c4_end12_s6"), (12, 7, "c4_end12_s7"), (12, 8, "c4_end12_s8"),
        (12, 9, "c4_end12_s9")]

# what the table said before this script existed; the new nine rows are generated, not typed
PUBLISHED = {(8, 0): (8, 111, 76.6, 100.0), (8, 1): (6, 102, 78.9, 100.0),
             (8, 2): (7, 95, 79.4, 100.0),
             (10, 0): (17, 214, 68.2, 100.0), (10, 1): (14, 166, 72.9, 100.0),
             (10, 2): (7, 166, 72.6, 100.0),
             (12, 0): (22, 385, 58.2, 100.0), (12, 1): (24, 413, 53.0, 99.9),
             (12, 2): (24, 401, 55.5, 97.5), (12, 3): (22, 391, 61.5, 100.0),
             (12, 4): (24, 406, 63.9, 100.0), (12, 5): (24, 420, 57.0, 100.0),
             (12, 6): (24, 406, 71.5, 99.1)}

CONTROLS = {(8, 0): "0/6", (10, 0): "0/2", (12, 0): "0/2"}

CAP = {8: 12, 10: 24, 12: 24}

# the budget each depth was published with: history repetition, stopping set, round cap
BUDGET = {8: (20, "agg", 12), 10: (60, "both", 24), 12: (60, "both", 24)}


def load(name):
    return json.loads((R / (name + ".json")).read_text(encoding="utf-8"))


def row(arm):
    rep = arm["repair"]
    log = rep["log"]
    cap = arm["args"]["max_iters"]
    broke = log[-1]["worst_draw_rate"] == 1.0
    rounds = len(log) - 1 if broke else len(log)
    assert broke or rounds == cap, (rounds, cap)
    d0 = 100 * arm["base"]["certificate"]["random_draw_rate_exact"]
    d1 = 100 * rep["final"]["random_draw_rate_exact"]
    w0 = arm["base"]["certificate"]["worst_draw_rate"]
    w1 = rep["final"]["worst_draw_rate"]
    return rounds, len(rep["aggregate"]), w0, w1, d0, d1, broke


rows = []
for E, s, name in ARMS:
    arm = load(name)
    rf = arm["args"].get("roots_file")
    if rf is not None:
        assert rf.endswith("_40_s0.json"), (name, rf)
    assert arm["args"]["empties"] == E and arm["args"]["seed"] == s, name
    b = BUDGET[E]
    for key, want in (("agg_repeat", b[0]), ("stop", b[1]), ("max_iters", b[2]),
                      ("epochs", 200), ("lr", 0.002), ("hidden", 128), ("emb", 16),
                      ("roots", 40), ("root_value", 0)):
        assert arm["args"].get(key) == want, (name, key, arm["args"].get(key), want)
    r = row(arm)
    if (E, s) in PUBLISHED:
        got = (r[0], r[1], round(r[4], 1), round(r[5], 1))
        assert got == PUBLISHED[(E, s)], (name, got, PUBLISHED[(E, s)])
    rows.append(((E, s), name, r))
print("rounds/D rule reproduces the published %d rows of tab:scaling76" % len(PUBLISHED))

last = None
for (E, s), name, (rounds, added, w0, w1, d0, d1, broke) in rows:
    if last is not None and last != E:
        print(r"\midrule")
    last = E
    print("%d & %d & %d%s & %d & $%d\\to%d$ & $%.1f\\to%s$ & %s \\\\" % (
        E, s, rounds, "$^{c}$" if not broke else "", added, w0, w1, d0,
        "100" if d1 > 99.95 else "%.1f" % d1,
        CONTROLS.get((E, s), "--")))

for E in (8, 10, 12):
    sel = [(s, rounds, w1, broke) for (e, s), _, (rounds, _a, _w0, w1, _d0, _d1, broke) in rows
           if e == E]
    print("depth %2d: %2d seeds | W=1 final %d (of which only after the last refit: %s) "
          "| capped %s | rounds %s" % (
              E, len(sel), sum(1 for x in sel if x[2] == 1.0),
              " ".join("%d/%d" % (s, r) for s, r, w, b in sel if w == 1.0 and not b),
              " ".join(str(s) for s, r, w, b in sel if not b) or "-",
              " ".join(str(r) for _s, r, _w, _b in sel)))
