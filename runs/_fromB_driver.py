"""Repair against the blind suite, then read the weights on the published one.

The published arms were repaired against suite A and certified on suite A, and every one of
them collapses to zero on suite B. That is only a suite-indexed certificate if the same run
pointed at suite B refuses to certify suite A, so this drives the paired experiment: same
seed, same budget, same network, same forty roots per suite - only the suite the repair is
aimed at changes. Phase one runs the six repairs, phase two the twelve verifications, so a
verification never reads a checkpoint that is still being written.
"""
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
BUDGET = {8: (20, "agg", 12), 10: (60, "both", 24)}
THREADS = 1
SLOTS = 2


def suite(which, E):
    return "data/c4_endgame_roots_%d_40_%s.json" % (E, "s0" if which == "A" else "holdoutB")


def repairs():
    for E in (8, 10):
        agg, stop, iters = BUDGET[E]
        for s in (0, 1, 2):
            stem = "end%dB_s%d" % (E, s)
            ckpt = "runs/c4_%s" % stem
            yield dict(
                name="repair " + stem,
                log=RUNS / ("_fromB_%s.log" % stem),
                out=REPO / "data" / ("c4_%s.json" % stem),
                cmd=[sys.executable, "src/c4_endgame.py",
                     "--rows", "6", "--cols", "7", "--win", "4", "--empties", str(E),
                     "--roots", "40", "--root-value", "0", "--branch", "3",
                     "--hidden", "128", "--emb", "16", "--epochs", "200", "--lr", "2e-3",
                     "--agg-repeat", str(agg), "--stop", stop, "--max-iters", str(iters),
                     "--seed", str(s), "--no-controls", "--threads", str(THREADS),
                     "--roots-file", suite("B", E), "--ckpt-out", ckpt,
                     "--out", "data/c4_%s.json" % stem])


def verifies():
    for E in (8, 10):
        for s in (0, 1, 2):
            stem = "end%dB_s%d" % (E, s)
            for blind in ("A", "B"):
                yield dict(
                    name="verify %s %s" % (stem, blind),
                    log=RUNS / ("_fromB_verify%s_s%d_end%d.log" % (blind, s, E)),
                    out=REPO / "data" / ("c4_%s_verify%s.json" % (stem, blind)),
                    cmd=[sys.executable, "src/c4_endgame_verify.py",
                         "--ckpt", "runs/c4_%s/c4_gru.pt" % stem,
                         "--roots", suite(blind, E), "--hidden", "128", "--emb", "16",
                         "--orders", "120", "--games", "40",
                         "--out", "data/c4_%s_verify%s.json" % (stem, blind)])


def drain(jobs, tag):
    todo = [j for j in jobs if not j["out"].exists()]
    skipped = len(jobs) - len(todo)
    if skipped:
        print("%s: %d ya estaban, no se repiten" % (tag, skipped), flush=True)
    running = []
    t0 = time.time()
    while todo or running:
        while todo and len(running) < SLOTS:
            j = todo.pop(0)
            j["t"] = time.time()
            j["p"] = subprocess.Popen(j["cmd"], cwd=REPO,
                                      stdout=open(j["log"], "w"), stderr=subprocess.STDOUT)
            print("  inico %s (%d corriendo)" % (j["name"], len(running)), flush=True)
            running.append(j)
        time.sleep(20)
        for j in list(running):
            if j["p"].poll() is not None:
                running.remove(j)
                ok = j["out"].exists()
                print("  %s %-22s %5.0f s -> %s" % ("acabo" if ok else "FALLO",
                                                    j["name"], time.time() - j["t"], j["out"].name),
                      flush=True)
                if not ok:
                    print("     cola en %s" % j["log"], flush=True)
    print("%s: todo en %.2f h" % (tag, (time.time() - t0) / 3600), flush=True)


if __name__ == "__main__":
    (RUNS / "c4_end8B_s0").mkdir(parents=True, exist_ok=True)
    drain(list(repairs()), "reparaciones sobre el mazo ciego")
    drain(list(verifies()), "verificaciones en los dos mazos")
    print("fin", flush=True)
