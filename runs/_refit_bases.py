"""Re-derive on this machine the six base checkpoints that make an arm's control paired.

`tab:blind` compares a repaired policy with the weights it started from, so an arm whose base was
never saved cannot be paired at all. `src/c4_endgame.py` grew `--base-fit` for exactly this case:
the same seed, the same recipe and the same forty positions reproduce the archived base, and the
script refuses the run unless the recovered policy certifies the numbers the arm logged beside it.
Six arms are in that case, and all six trained here under this interpreter -- python 3.11.9, torch
2.14.1+cpu -- so the reproduction is not a cross-version leap of faith but the same computation,
which is what the archived `args` pin and what this script re-derives rather than retypes.

Then the paired reads run here too, because a memo-free read is a recursion and not a training
run. Each one is checked against the arm's own base certificate with the same function the Colab
ingestion uses, and a read that disagrees is deleted instead of kept. That check does double duty for
the one arm whose archive predates `--roots-file` and so sampled its suite in-process: the read is
taken over the published file, so if the sample and the file were different forty positions the two
numbers could not agree, and the read is deleted rather than filed.

    python runs/_refit_bases.py            # refit what is missing, then read it
    python runs/_refit_bases.py --check    # report what would run, run nothing
"""
import argparse, json, os, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "colab"))
import reads_plan as rp

sys.path.insert(0, str(REPO / "runs"))
from _integrate_reads import check_reads

SCRATCH = HERE / "_basefit"
# the flags that decide what the base fit computes; everything else only decides when the repair
# loop stops, and passing it anyway keeps the scratch JSON's args block identical to the archive.
FIT_FLAGS = ("rows", "cols", "win", "empties", "roots", "root_value", "branch",
             "hidden", "emb", "epochs", "lr", "agg_repeat", "stop", "max_iters",
             "seed", "threads")


def cmd_for(row, arch):
    """The archived command, with --base-fit and --controls-from so the script itself checks it."""
    a = arch["args"]
    cmd = [sys.executable, "src/c4_endgame.py"]
    for k in FIT_FLAGS:
        cmd += ["--" + k.replace("_", "-"), str(a[k])]
    if a.get("roots_file"):
        cmd += ["--roots-file", a["roots_file"].replace("\\", "/")]
    return cmd + ["--no-controls", "--base-fit", "--controls-from", row["arm_json"],
                  "--ckpt-out", str(pathlib.Path(row["ckpt"]).parent),
                  "--out", str(SCRATCH / (pathlib.Path(row["arm_json"]).stem + ".json"))]


def verify_cmd(row):
    """One memo-free read, with the notebook's flags: same script, same orders, same budget."""
    return [sys.executable, "src/c4_endgame_verify.py", "--ckpt", row["ckpt"],
            "--roots", row["suite_file"], "--hidden", "128", "--emb", "16",
            "--orders", "120", "--games", "40", "--out", row["out"]]


def run(cmd, log):
    """A subprocess whose transcript lands in a file, so a killed session still leaves evidence."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh:
        return subprocess.call(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report what would run, run nothing")
    args = ap.parse_args()

    owed = rp.pending()
    bases = [r for r in owed if r["kind"] in ("base-on-A", "base-on-B")]

    done = 0
    for r in bases:
        arch = json.loads((REPO / r["arm_json"]).read_text(encoding="utf-8"))
        need_fit = not rp.exists(r["ckpt"])
        need_read = not rp.exists(r["out"])
        if not (need_fit or need_read):
            continue
        print("%s on %s: %s%s" % (r["arm"], r["suite"],
                                  "refit base" if need_fit else "base in place",
                                  ", read" if need_read else ""))
        if args.check:
            print("   " + " ".join(cmd_for(r, arch)))
            done += 1
            continue

        if need_fit:
            cmd = cmd_for(r, arch)
            rc = run(cmd, SCRATCH / (pathlib.Path(r["arm_json"]).stem + ".log"))
            transcript = (SCRATCH / (pathlib.Path(r["arm_json"]).stem + ".log")).read_text(
                encoding="utf-8")
            ok = "reproduces the archived base certificate: True" in transcript
            if rc != 0 or not ok or not rp.exists(r["ckpt"]):
                print("".join("   | " + l + "\n" for l in transcript.splitlines()))
                raise SystemExit("%s: el base fit no reprodujo el certificado archivado "
                                 "(rc=%d); nada de lo que depende de el peso se ley"
                                 % (r["arm"], rc))
            print("   base re-derived: %s" % r["ckpt"])

        if need_read:
            log = SCRATCH / (pathlib.Path(r["out"]).stem + ".log")
            rc = run(verify_cmd(r), log)
            if rc != 0:
                print("".join("   | " + l + "\n" for l in
                               log.read_text(encoding="utf-8").splitlines()))
                raise SystemExit("%s: la lectura salio con rc=%d" % (r["out"], rc))
            _lines, fail = check_reads({}, [r])
            if fail:
                (REPO / r["out"]).unlink()
                raise SystemExit("%s: %s -> la lectura se elimino, no se guardo"
                                 % (r["out"], "; ".join(fail)))
            v = json.loads((REPO / r["out"]).read_text(encoding="utf-8"))
            print("   read W %.3f B %.3f D %.4f drawn %d/%d -> %s"
                  % (v["worst_draw_rate"], v["best_draw_rate"], v["random_draw_rate_exact"],
                     v["n_roots_drawn_worst"], v["roots"], r["out"]))
        done += 1

    if not bases:
        print("ninguna lectura base falta en data/: el control pareado ya está completo")
    else:
        print("%s: %d de %d lecturas base pendientes"
              % ("quedarían por hacer" if args.check else "resueltas", done, len(bases)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
