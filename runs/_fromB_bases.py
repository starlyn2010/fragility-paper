"""The missing paired cells: the six blind-seeded bases, read on both suites.

Phase two of `_fromB_driver.py` reports each repaired policy on both suites. That alone
cannot say whether a repair *costs* the other suite or merely fails to help it, because the
number to compare against is the same policy before repair. The six `c4_gru_base.pt` files
are on disk, so this reads each of them over both suites with the same verifier, the same 120
column orders and the same 40 adaptive teachers per root, two at a time -- which is also what
makes the reverse direction of `tab:blind` entirely verifier-reported, with no cell taken from
the training loop's own bookkeeping.
"""
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
SLOTS = 2


def jobs():
    for E in (8, 10):
        for s in (0, 1, 2):
            for blind in ("A", "B"):
                stem = "end%dB_s%d" % (E, s)
                out = "data/c4_%s_base_on%s.json" % (stem, blind)
                yield dict(
                    name="base %s on %s" % (stem, blind),
                    log=RUNS / ("_fromB_base_on%s_%s.log" % (blind, stem)),
                    out=REPO / out,
                    cmd=[sys.executable, "src/c4_endgame_verify.py",
                         "--ckpt", "runs/c4_%s/c4_gru_base.pt" % stem,
                         "--roots", "data/c4_endgame_roots_%d_40_%s.json"
                                    % (E, "s0" if blind == "A" else "holdoutB"),
                         "--hidden", "128", "--emb", "16",
                         "--orders", "120", "--games", "40", "--out", out])


if __name__ == "__main__":
    todo = [j for j in jobs() if not j["out"].exists()]
    print("%d verificaciones pendientes" % len(todo), flush=True)
    running, t0 = [], time.time()
    while todo or running:
        while todo and len(running) < SLOTS:
            j = todo.pop(0)
            j["t"] = time.time()
            j["p"] = subprocess.Popen(j["cmd"], cwd=REPO,
                                      stdout=open(j["log"], "w"), stderr=subprocess.STDOUT)
            running.append(j)
            print("  inico %s" % j["name"], flush=True)
        time.sleep(15)
        for j in list(running):
            if j["p"].poll() is not None:
                running.remove(j)
                ok = j["out"].exists()
                print("  %s %-18s %5.0f s" % ("acabo" if ok else "FALLO", j["name"],
                                              time.time() - j["t"]), flush=True)
    print("listo en %.2f h" % ((time.time() - t0) / 3600), flush=True)
