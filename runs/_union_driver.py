"""Do the two-suite arms: repair against A cup B, then read the result on A, on B, on both.

The question this answers is the one `tab:blind` raises. That table certifies against suite A
and finds nothing at $W=1$ on a board-disjoint suite B, in ten arms across three depths. The
reading it forces is that the certificate is a statement about a policy *and a set of
positions*. The test of that reading is to move the blind positions inside the set: run the
identical loop with `--roots-file data/c4_endgame_roots_<E>_80_AplusB.json`, which makes every
blind root a position the certificate can complain about and the repair can collect. Because
`certify` takes a minimum over roots of a per-root worst case and the two suites share no
board, $W_{A\\cup B}=\\min(W_A,W_B)$ while $\\bar D_{A\\cup B}$ is the mean of the two $\\bar D$,
so a union certificate at $W=1$ is a certificate at $W=1$ on the blind suite too. What the
experiment measures is the price: the budget is the one each depth was published with,
unchanged, so rounds and histories per certified root are directly comparable to the
single-suite arms in `tab:scaling76`.

    python runs/_union_driver.py --depths 8
    python runs/_union_driver.py --depths 8 10 --phase arms
    python runs/_union_driver.py --depths 8 --phase reads
"""
import argparse
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
SLOTS = 2

# the budget each depth was published with, so the arms stay comparable to tab:scaling76
BUDGET = {8: (20, "agg", 12), 10: (60, "both", 24)}
SEEDS = {8: (0, 1, 2), 10: (0, 1, 2)}


def suite(E, which):
    return "data/c4_endgame_roots_%d_%s.json" % (
        E, {"A": "40_s0", "B": "40_holdoutB", "U": "80_AplusB"}[which])


def arms(E):
    agg, stop, iters = BUDGET[E]
    for s in SEEDS[E]:
        stem = "end%dU_s%d" % (E, s)
        out = "data/c4_%s.json" % stem
        yield dict(name="arm " + stem, log=RUNS / ("_union_%s.log" % stem),
                   out=REPO / out,
                   cmd=[sys.executable, "src/c4_endgame.py", "--roots", "40",
                        "--empties", str(E), "--root-value", "0", "--branch", "3",
                        "--hidden", "128", "--emb", "16", "--epochs", "200", "--lr", "2e-3",
                        "--agg-repeat", str(agg), "--stop", stop, "--max-iters", str(iters),
                        "--seed", str(s), "--threads", "1", "--no-controls",
                        "--roots-file", suite(E, "U"),
                        "--ckpt-out", "runs/c4_%s" % stem, "--out", out])


def reads(E, ckpt):
    for s in SEEDS[E]:
        stem = "end%dU_s%d" % (E, s)
        for which in ("A", "B", "U"):
            out = "data/c4_%s_%s.json" % (stem, which)
            yield dict(name="read %s on %s" % (stem, which),
                       log=RUNS / ("_union_read_%s_%s.log" % (which, stem)),
                       out=REPO / out,
                       cmd=[sys.executable, "src/c4_endgame_verify.py", "--ckpt",
                            "runs/c4_%s/%s" % (stem, ckpt), "--roots", suite(E, which),
                            "--hidden", "128", "--emb", "16", "--orders", "120",
                            "--games", "40", "--out", out])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[8])
    ap.add_argument("--phase", choices=["all", "arms", "reads"], default="all")
    ap.add_argument("--ckpt", default="c4_gru.pt", help="c4_gru_base.pt for the base reads")
    args = ap.parse_args()

    todo = []
    if args.phase in ("all", "arms"):
        todo += [j for E in args.depths for j in arms(E)]
    if args.phase in ("all", "reads"):
        todo += [j for E in args.depths for j in reads(E, args.ckpt)]
    todo = [j for j in todo if not j["out"].exists()]
    print("%d jobs pendientes" % len(todo), flush=True)
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
                print("  %s %-24s %6.0f s" % ("acabo" if ok else "FALLO", j["name"],
                                              time.time() - j["t"]), flush=True)
    print("listo en %.2f h" % ((time.time() - t0) / 3600), flush=True)
