"""Run the notebook's queue on this machine, with the notebook's own commands and guards.

`colab/c4_finish_all.ipynb` exists because fourteen empty cells was thought to need a runtime
with more cores than this laptop has. What it queues is nine jobs whose commands, budget and
parallelism rule are all derived from `colab/reads_plan.py`, and nothing in that derivation is
Colab-specific: the budget is read off the archived twelve-cell arms, the suites are committed
files, and a memo-free read is a recursion that runs anywhere. So the same queue can run here,
and this is the driver that runs it - one process per job, the notebook's `MAX_PAR` and
`THREADS` unless told otherwise, and the same two promises checked before a result is kept.

The queue is re-derived from `data/` on every pass, so a session that dies mid-arm resumes at
the jobs still owed rather than re-training a repair that already landed. Transcripts go to
`runs/_localq/`, and `status.json` beside them is rewritten from those transcripts while the
processes run: reading a log does not touch the process that writes it, so a snapshot taken
here is evidence even if the machine is shut down a minute later.

    python runs/_local_queue.py --check    # print every command, run nothing
    python runs/_local_queue.py            # run what data/ still owes
    python runs/_local_queue.py --par 3    # override the notebook's two-at-a-time rule
"""
import argparse, ctypes, json, os, pathlib, re, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "colab"))
import reads_plan as rp

sys.path.insert(0, str(HERE))
from _integrate_reads import check_reads

SCRATCH = HERE / "_localq"
ROUND = re.compile(r"\[repair it\s+(\d+)\] W (\d+) B (\d+) D ([\d.]+) drawn (\d+)/(\d+) "
                   r"viol\s+(\d+) agg\s+(\d+) rep_mis (\d+)")
FIT = re.compile(r"fit loss ([\d.]+) epochs (\d+).*?\[(\d+)s\]")
# a job is only started while this much RAM is still free: three trainings that page are slower
# than two that do not, and the stall this rule exists to prevent is measured in hours
MIN_FREE_GB = 2.5


def free_gb():
    """Free physical RAM in GB, or None where this interpreter cannot ask for it."""
    if os.name != "nt":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return st.ullAvailPhys / 1024 ** 3


def parallelism(par=0):
    """The notebook's rule: as many jobs as cores, capped at two when free RAM is unknown.

    The notebook's helper cannot measure free RAM on Windows and falls back to two; this one can,
    so `--par` lifts the cap only once the resident size of a running arm has been measured. The
    file `runs/_localq/par` overrides it mid-run, which is how a third slot is opened without
    orphaning the two arms already training.
    """
    cores = os.cpu_count() or 1
    jobs = par if par > 0 else max(1, min(cores, 2))
    return jobs, max(1, cores // jobs)


def cmd_for(row, threads):
    """The command the notebook's pump would run for this row, flag for flag."""
    if row["kind"] != "arm-repair":
        return [sys.executable, "src/c4_endgame_verify.py", "--ckpt", row["ckpt"],
                "--roots", row["suite_file"], "--hidden", "128", "--emb", "16",
                "--orders", "120", "--games", "40", "--out", row["out"]]
    b = rp.arm_budget(row["E"])
    cmd = [sys.executable, "src/c4_endgame.py", "--rows", "6", "--cols", "7", "--win", "4",
           "--empties", str(row["E"])]
    for flag in ("roots", "root_value", "branch", "hidden", "emb", "epochs", "lr",
                 "agg_repeat", "stop", "max_iters"):
        cmd += ["--" + flag.replace("_", "-"), str(b[flag])]
    return cmd + ["--roots-file", row["suite_file"], "--threads", str(threads),
                  "--seed", str(row["seed"]), "--no-controls",
                  "--ckpt-out", pathlib.Path(row["ckpt"]).parent.as_posix(),
                  "--out", row["out"]]


def log_of(row):
    return SCRATCH / (pathlib.Path(row["out"]).stem + ".log")


def owed():
    """The planned arms, plus any read the paper owes whose weights are already on this disk.

    A read needs a checkpoint, and `.gitignore` keeps checkpoints out of the repository, so which
    reads this machine can do is a question about the disk and not about the plan. Asking it here
    is what lets one driver finish a depth: the arms land, their paired base reads become owed, and
    the base weights those reads need are the ones the arms just saved.
    """
    rows = {r["out"]: r for r in rp.arm_rows()}
    for r in rp.pending():
        if r["out"] not in rows and rp.exists(r["ckpt"]) and rp.exists(r["arm_json"]):
            rows[r["out"]] = r
    return list(rows.values())


def progress(log, started):
    """What the transcript says so far, read without touching the process that writes it."""
    out = {"elapsed_s": round(time.time() - started, 1), "bytes": 0}
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    out["bytes"] = len(text)
    rounds = ROUND.findall(text)
    if rounds:
        it, w, b, d, drawn, n, viol, agg, mis = rounds[-1]
        out.update(round=int(it), worst_draw_rate=int(w), best_draw_rate=int(b),
                   random_draw_rate_exact=float(d), drawn=int(drawn), roots=int(n),
                   violations=int(viol), aggregate=int(agg), replay_mistakes=int(mis),
                   rounds_logged=len(rounds))
    fits = FIT.findall(text)
    if fits:
        out["last_fit_s"] = int(fits[-1][2])
        out["fits"] = len(fits)
    for line in reversed(text.splitlines()):
        if line.strip():
            out["last_line"] = line.strip()[:160]
            break
    return out


def snapshot(running, done, par, threads):
    """A file derived from the logs, so the state of the queue survives this session."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    jobs = []
    for out, (proc, row, started) in sorted(running.items()):
        jobs.append(dict(out=out, arm=row["arm"], kind=row["kind"], suite=row["suite"],
                         pid=proc.pid, alive=proc.poll() is None,
                         log=str(log_of(row).relative_to(REPO)).replace("\\", "/"),
                         **progress(log_of(row), started)))
    state = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "par": par, "threads": threads,
             "free_gb": None if free_gb() is None else round(free_gb(), 2),
             "running": jobs, "finished": [r["out"] for r in done],
             "owed": [r["out"] for r in owed()]}
    (SCRATCH / "status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    lines = ["%s  par=%d threads=%d  RAM libre %s GB  corriendo=%d hechos=%d por hacer=%d"
             % (state["written"], par, threads, state["free_gb"], len(jobs), len(done),
                len(state["owed"]))]
    for j in jobs:
        lines.append("  %-16s %-14s on %s  pid=%-6d %6.0f s  %s"
                     % (j["arm"], j["kind"], j["suite"], j["pid"], j["elapsed_s"],
                        j.get("last_line", "(sin salida todavia)")))
    for r in done:
        lines.append("  HECHO %s" % r["out"])
    (SCRATCH / "status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return state


def validate(row, rc):
    """The notebook's promises, re-run here: a job that fails its check leaves nothing behind."""
    log = log_of(row)
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    if rc != 0 or not rp.exists(row["out"]):
        raise SystemExit("%s: rc=%d y %s no esta en data/\n%s"
                         % (row["out"], rc, row["out"],
                            "".join("   | " + l + "\n" for l in text.splitlines()[-40:])))
    if row["kind"] == "arm-repair":
        d = json.loads((REPO / row["out"]).read_text(encoding="utf-8"))
        if not rp.exists(row["ckpt"]):
            raise SystemExit("%s: el brazo no guardo %s, y sus dos lecturas lo necesitan"
                             % (row["out"], row["ckpt"]))
        fin = d["repair"]["final"]
        log = d["repair"]["log"]
        # the published rule, the one runs/_scaling_table.py asserts: a log row carries the
        # certificate before its refit, so an arm that broke on a certificate refitted one less
        broke = log[-1]["worst_draw_rate"] == 1.0
        print("   repair W %.3f B %.3f D %.4f | rondas %d | historias %d | %.0f s"
              % (fin["worst_draw_rate"], fin["best_draw_rate"], fin["random_draw_rate_exact"],
                 len(log) - 1 if broke else len(log), len(d["repair"]["aggregate"]),
                 d["repair"]["elapsed_s"]), flush=True)
        return
    lines, fail = check_reads({}, [row])
    for l in lines:
        print(l, flush=True)
    if fail:
        (REPO / row["out"]).unlink()
        raise SystemExit("%s: %s -> la lectura se elimino, no se guardo"
                         % (row["out"], "; ".join(fail)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--par", type=int, default=0, help="jobs at once (default: the notebook's rule)")
    args = ap.parse_args()
    control = SCRATCH / "par"
    if control.is_file():
        args.par = int(control.read_text(encoding="utf-8").strip())
    par, threads = parallelism(args.par)

    queue = owed()
    if args.check:
        print("%d jobs por hacer, %d a la vez, --threads %d" % (len(queue), par, threads))
        for row in queue:
            print("  %-14s %-16s on %s -> %s" % (row["kind"], row["arm"], row["suite"], row["out"]))
            print("     " + " ".join(cmd_for(row, threads)))
        return 0
    if not queue:
        print("data/ no debe ningun brazo: la cola del notebook esta vacia")
        return 0

    SCRATCH.mkdir(parents=True, exist_ok=True)
    print("%d jobs, %d a la vez, --threads %d, transcripciones en %s"
          % (len(queue), par, threads, SCRATCH), flush=True)
    running, done = {}, []
    while True:
        if control.is_file():
            par, threads = parallelism(int(control.read_text(encoding="utf-8").strip()))
        # longest first: a repair is hours and a read is minutes, so letting a read take the
        # last free slot would push the next repair back by the length of that read
        ready = sorted(owed(), key=lambda r: r["kind"] != "arm-repair")
        for row in ready:
            if len(running) >= par:
                break
            if row["out"] in running or not all(rp.exists(n) for n in row.get("needs", [])):
                continue
            avail = free_gb()
            if avail is not None and avail < MIN_FREE_GB:
                print("esperando RAM: %.1f GB libres, minimo %.1f" % (avail, MIN_FREE_GB),
                      flush=True)
                break
            log = log_of(row)
            fh = open(log, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd_for(row, threads), cwd=str(REPO), stdout=fh,
                                    stderr=subprocess.STDOUT)
            fh.close()
            running[row["out"]] = (proc, row, time.time())
            print("arranca %-14s %-16s on %s pid=%d --threads %d -> %s"
                  % (row["kind"], row["arm"], row["suite"], proc.pid, threads,
                     log.relative_to(REPO).as_posix()), flush=True)
        if not running:
            break
        state = snapshot(running, done, par, threads)
        for j in state["running"]:
            print("   %-16s %-14s %6.0f s  %s"
                  % (j["arm"], j["kind"], j["elapsed_s"], j.get("last_line", "")), flush=True)
        time.sleep(60)
        for out, (proc, row, _started) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[out]
            rc = proc.returncode
            print("termina %-14s %-16s rc=%d" % (row["kind"], row["arm"], rc), flush=True)
            validate(row, rc)
            done.append(row)
    snapshot(running, done, par, threads)
    print("%d jobs resueltos; data/ debe ahora %d" % (len(done), len(owed())), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())