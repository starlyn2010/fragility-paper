"""Contract tests for the seed-ladder Colab notebook.

`colab/build_followup_notebooks.py` writes `c4_ladder_finish.ipynb` out of the cells
`tests/test_colab_notebook.py` already pins down, so the shared machinery - the bundle, the
boot, the capacity cell, the flag spellings - is checked there over every document. The
fourteen-empty arms this notebook used to plan now live in `colab/c4_finish_all.ipynb`, with
their own contract in `tests/test_reads_notebooks.py`. What is specific to this document is
the part that was never tested before and cannot fail until a runtime has spent an hour on it:

1. the blind suite. Suite B is only a holdout because of what it is not: not the depth, not
   finished, not a loss, and not in suite A. The sampling cell checks all four, so a
   copy-and-paste edit that reuses suite A's file stops the run instead of quietly certifying
   the same forty positions twice.
2. the arm order. A verification reads a checkpoint and a repair JSON, so it must never start
   before the repair that made them has finished; getting that backwards re-derives a policy
   that does not exist yet.
3. the budget. A new seed is comparable with the published one only under the same epochs,
   round cap, network and suite, so those are read out of the archived arm JSONs rather than
   restated here.
4. the coverage. These documents exist to widen error bars, so the plan is asserted to reach
   ten seeds at twelve empty cells and six at ten and eight - and every planned arm gets a
   verification on both suites.
5. the bundle. The suites are inputs to the experiment: a bare runtime that lacks the holdout
   samples its own, and then two accounts are not running the same test.
6. when it downloads. Packaging an arm costs the arm wall time, and the first ladder run paid
   for it on every finished arm. So the finisher writes nothing while a depth is incomplete,
   ships a depth the moment its last arm lands, and keeps a sweep cell for the session that
   is ending - three promises, each checked against a mutant that ships anyway.

Each promise is also checked against a mutant that breaks it the way an accident would, so a
green run means the cells really refuse.

Run with pytest, or directly: python tests/test_followup_notebooks.py
"""
import atexit
import contextlib
import io
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
import zipfile

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO / "tests"))
import test_colab_notebook as base                                    # noqa: E402

LADDER = REPO / "colab" / "c4_ladder_finish.ipynb"
NB2 = [LADDER]      # the fourteen-cell arms moved to colab/c4_finish_all.ipynb, tested there
FINISH = REPO / "colab" / "c4_finish_all.ipynb"

PLAN_TAG = "PLAN = "
SUITES_TAG = "def _grow(path, E, need, exclude"
QUEUE_TAG = "def build(empties,"
TICK_TAG = base.TICK_TAG
RESULTS_TAG = "for a in snapshot()"
SHIP_TAG = "suelta = ship()"      # the one cell that ships a depth that is not whole

CHEAP = ((8, (3,)),)          # a plan small enough to run through on the laptop
NEEDS = "needs=[src, ckpt]"


def plan_literal(nb):
    """The notebook's own PLAN line, verbatim, so a test can replace it exactly."""
    src = base._cell(PLAN_TAG, nb)
    return re.search(r"^PLAN = (.*)$", src, re.M).group(1)


def plan_of(nb):
    return [tuple(t) for t in eval(plan_literal(nb))]


def suite_rel(which, E):
    """Repo-relative path of a committed suite, named the way a runtime names it."""
    suffix = "s0" if which == "A" else "holdoutB"
    return "data/c4_endgame_roots_%d_40_%s.json" % (E, suffix)


def _posix(p):
    return str(p).replace(os.sep, "/")


def published():
    """{(empty cells, seed): (file, args)} for the arms already archived in the repository.

    Only a repair arm counts, so the name has to be the one the queue writes: the root
    suites share this glob and are JSON lists, and a control arm
    (`c4_end12_s0_all.json`) carries a different question in its args.
    """
    out = {}
    for path in sorted((REPO / "data").glob("c4_end*_s*.json")):
        m = re.fullmatch(r"c4_end(?:(\d+)|_ctrl)?_s(\d+)\.json", path.name)
        if not m:
            continue
        try:
            arm = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        args = arm.get("args") or {}
        if not args.get("empties") or args.get("seed") is None:
            continue
        if m.group(1) and str(args["empties"]) != m.group(1):
            continue
        out[(args["empties"], args["seed"])] = (path, args)
    return out


# --------------------------------------------------------------------- run one document --
KEEP = []


@atexit.register
def _clean_up():
    """The boot directories outlive a run so the caller can assert on what the cells left
    behind; they are removed once the interpreter stops, not before the first assertion."""
    for d in KEEP:
        shutil.rmtree(d, ignore_errors=True)


class _StubTime:
    """A clock that jumps an hour per reading, so the pump guard is reached at once."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = 0

    def time(self):
        self.now += 3600.0
        return self.now

    def sleep(self, _seconds):
        self.sleeps += 1
        if self.sleeps > 20:
            raise AssertionError("the pump never reached its guard; it would hold the "
                                 "runtime for the rest of the session")

    def __getattr__(self, name):
        return getattr(time, name)


def _plant(repo, pairs):
    for src_rel, dst_rel in pairs:
        dst = repo / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / src_rel, dst)


def run_all(nb, plan=CHEAP, suites=None, arms=None, until=None, mutate=None):
    """Boot the bundle into a fake /content and run every other cell, in document order.

    `plan` replaces the notebook's own plan, because the notebooks queue twelve- and
    fourteen-empty arms and no laptop solves those during a test suite; `suites` plants the
    suite files the sampling cell looks for; `arms` plants archived results so the table cell
    has something real to read; `until` stops before the first cell holding that text, which
    is how the download-timing promise is read without running the sweep cell; `mutate`
    rewrites every code cell, for the mutants below. Returns text, launch records, the
    namespace the cells shared, the booted repo and the names of the zips handed to the
    browser.
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb2_"))
    KEEP.append(tmp)
    cwd, path, sub = os.getcwd(), list(sys.path), sys.modules["subprocess"]
    old_env = os.environ.get("COLAB_CONTENT")
    out, stub = io.StringIO(), base._StubSubprocess()
    try:
        os.environ["COLAB_CONTENT"] = str(tmp)
        os.chdir(tmp)
        ns = base._boot(tmp, nb=nb)
        repo = ns["REPO"]
        os.chdir(repo)          # as bootstrap() does on a runtime: the cells read relative
        _plant(repo, suites if suites is not None else _suites_for(plan))
        for i, src in base._code_cells(nb):
            if any(t in src for t in (base.BUNDLE_TAG, base.SETUP_TAG)):
                continue
            if until and until in src:
                break
            src = src.replace(plan_literal(nb), str(plan))
            if mutate:
                src = mutate(src)
            if TICK_TAG in src:
                sys.modules["subprocess"] = stub
            if base.PUMP_TAG in src:
                ns["time"] = _StubTime()
            if RESULTS_TAG in src and arms is not None:
                _plant(repo, arms)
            with contextlib.redirect_stdout(out):
                exec(base._runnable(src), ns)
        zips = sorted(p.name for p in tmp.glob("c4_*_results.zip"))
        return out.getvalue(), stub, ns, repo, zips
    finally:
        sys.modules["subprocess"] = sub
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None)
            if f and base._under(pathlib.Path(f).resolve(), tmp):
                del sys.modules[name]
        sys.path[:] = path
        os.chdir(cwd)
        if old_env is None:
            os.environ.pop("COLAB_CONTENT", None)
        else:
            os.environ["COLAB_CONTENT"] = old_env
        # the boot directory stays until the interpreter stops: the caller asserts on
        # what the cells left in it, and rmtree here would delete it first


def _suites_for(plan):
    """Plant the committed suites of every depth in the plan, A and B alike."""
    pairs = []
    for E, _seeds in plan:
        for which in ("A", "B"):
            rel = suite_rel(which, E)
            assert (REPO / rel).exists(), "%s is not committed yet" % rel
            pairs.append((rel, rel))
    return pairs


def _arms_for(plan, src_E=12, src_s=0):
    """One planted repair arm with its two verifications, at the plan's first depth."""
    E, seeds = plan[0]
    s = seeds[0]
    return [("data/c4_end%d_s%d.json" % (src_E, src_s), "data/c4_end%d_s%d.json" % (E, s)),
            ("data/c4_end%d_verify_s%d.json" % (src_E, src_s),
             "data/c4_end%d_verify_s%d.json" % (E, s)),
            ("data/c4_end%d_verify_s%d.json" % (src_E, src_s),
             "data/c4_end%d_holdoutB_s%d.json" % (E, s))]


# ------------------------------------------------------------------------- the holdout --
def test_a_bare_runtime_runs_a_queue_end_to_end():
    for nb in NB2:
        text, stub, ns, repo, zips = run_all(nb, arms=_arms_for(CHEAP))
        assert "posiciones compartidas 0" in text, "%s: %s" % (nb.name, text[:600])
        assert _posix(ns["SUITE_A"][8]) == suite_rel("A", 8), ns.get("SUITE_A")
        assert _posix(ns["SUITE_B"][8]) == suite_rel("B", 8), ns.get("SUITE_B")
        started = [c[c.index("--out") + 1] for c, _ in stub.launches]
        assert started, "%s launched nothing" % nb.name
        assert all("holdoutB" not in o and "_verify_" not in o for o in started), (
            "%s started a verification in the first wave: %r" % (nb.name, started))
        assert (repo / "runs").is_dir(), "%s never made its log folder" % nb.name
        assert "run this cell again" in text or "queue empty" in text, text[-600:]
        assert "seed 3: W" in text, "%s: no arm line in the table: %s" % (nb.name, text[-700:])
        assert "verify A: W" in text and "verify B: W" in text, text[-900:]
        assert "CIEGA" in text and re.search(r"\[\d\.\d+, \d\.\d+\]", text), (
            "no Wilson interval was printed: " + text[-700:])
        assert "None" not in text, text[-700:]
        assert zips, "%s handed nothing to the browser" % nb.name


def test_the_ladder_boots_its_whole_plan_without_asking_anything():
    """The cheap plan above proves the cells work; this proves the document does, as written.

    Nothing runs for real, so the price is reading the six committed suites and doing the
    queue's arithmetic: nine seeds, three arms each, and a results cell that prints "nothing
    yet" per depth instead of failing on an empty table. The plan is the one in the file, so
    this is also where a plan that lost a seed or a depth would be noticed. Nothing reaches
    the browser while the arms are in flight - the zips this sees come from the sweep cell at
    the end of the document, which is the cell the next test below stops before.
    """
    plan = plan_of(LADDER)
    seeds = sum(len(s) for _E, s in plan)
    text, stub, ns, repo, zips = run_all(LADDER, plan=plan, arms=[])
    assert set(ns["SUITE_A"]) == set(ns["SUITE_B"]) == {E for E, _s in plan}, ns.get("SUITE_A")
    assert "12 vacias | semillas (7, 8, 9)" in text, text[:800]
    assert "%d arms to go after 600 min" % (3 * seeds) in text, text[-700:]
    started = [c[c.index("--out") + 1] for c, _ in stub.launches]
    assert started and all(s.startswith("data/c4_end12_s") for s in started), (
        "the first arms a bare runtime launches are not the twelve-empty repairs: %r"
        % sorted(set(started)))
    assert sorted(zips) == sorted("c4_finish_end%d_results.zip" % E for E, _s in plan), zips
    with zipfile.ZipFile(repo.parent / "c4_finish_end8_results.zip") as z:
        inside = set(z.namelist())
    assert {"data/c4_endgame_roots_8_40_s0.json",
            "data/c4_endgame_roots_8_40_holdoutB.json"} <= inside, sorted(inside)
    assert "0 de 3 semillas con brazo en disco" in text, text[-900:]


def test_a_holdout_that_is_not_a_holdout_stops_the_run():
    """Planting suite A under suite B's name is the easy mistake, and the hard failure."""
    pairs = [(suite_rel("A", 8), suite_rel("A", 8)), (suite_rel("A", 8), suite_rel("B", 8))]
    try:
        run_all(LADDER, suites=pairs)
    except AssertionError as exc:
        assert "holdout" in str(exc), exc
    else:
        raise AssertionError("a suite B identical to suite A passed the holdout check")


def test_a_suite_of_the_wrong_depth_stops_the_run():
    pairs = [(suite_rel("A", 10), suite_rel("A", 10)),
             (suite_rel("A", 8), suite_rel("B", 10))]
    try:
        run_all(LADDER, plan=((10, (3,)),), suites=pairs)
    except AssertionError as exc:
        assert "empty cells" in str(exc), exc
    else:
        raise AssertionError("an eight-empty suite passed as the ten-empty holdout")


def test_a_losing_root_is_not_admitted_as_a_draw_suite():
    """The published suites are draw-valued; a suite containing a loss certifies nothing."""
    roots = json.loads((REPO / suite_rel("B", 8)).read_text(encoding="utf-8"))
    roots[0]["value"] = 1
    hold = pathlib.Path(tempfile.mkdtemp(prefix="nb2_value_"))
    try:
        tampered = hold / "b.json"
        tampered.write_text(json.dumps(roots), encoding="utf-8")
        pairs = [(suite_rel("A", 8), suite_rel("A", 8)),
                 (str(tampered), suite_rel("B", 8))]
        try:
            run_all(LADDER, suites=pairs)
        except AssertionError as exc:
            assert "draw" in str(exc), exc
        else:
            raise AssertionError("a holdout with a winning-for-the-teacher root was accepted")
    finally:
        shutil.rmtree(hold, ignore_errors=True)


def test_a_finished_position_is_not_admitted_as_a_root():
    roots = json.loads((REPO / suite_rel("B", 8)).read_text(encoding="utf-8"))
    assert all(not _terminal(r["board"]) for r in roots), "the committed suite has a winner"
    done = json.loads(json.dumps(roots))
    done[0]["board"] = [1, 2, 1, 2, 1, 2, 1] * 6            # 42 cells, no empties at all
    hold = pathlib.Path(tempfile.mkdtemp(prefix="nb2_term_"))
    try:
        tampered = hold / "b.json"
        tampered.write_text(json.dumps(done), encoding="utf-8")
        pairs = [(suite_rel("A", 8), suite_rel("A", 8)),
                 (str(tampered), suite_rel("B", 8))]
        try:
            run_all(LADDER, suites=pairs)
        except AssertionError as exc:
            assert "empty cells" in str(exc) or "finished" in str(exc), exc
        else:
            raise AssertionError("a board with no empty cells passed as an endgame root")
    finally:
        shutil.rmtree(hold, ignore_errors=True)


def _terminal(board):
    """Cheap terminal test, so the test does not depend on the game module's state."""
    b, cols, rows = list(board), 7, 6
    for r in range(rows):
        for c in range(cols):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                vals = []
                for k in range(4):
                    rr, cc = r + dr * k, c + dc * k
                    if not (0 <= rr < rows and 0 <= cc < cols):
                        break
                    vals.append(b[rr * cols + cc])
                if len(vals) == 4 and vals[0] and len(set(vals)) == 1:
                    return True
    return False


# ------------------------------------------------------------------------- the queue ----
def queue_env(nb, plan, mutate=None):
    """The queue cell's namespace for a plan, without booting or launching anything.

    Building a job list is pure string work, so the tests that read commands - budget, suite
    identity, coverage - do not need a repository, a suite on disk, or a runtime: only the
    ordering tests pay for a boot. `mutate` edits the cell the way an accident would.
    """
    env = {"os": os, "json": json, "pathlib": pathlib, "MAX_PAR": 2, "THREADS": 1,
           "PLAN": plan, "SUITE_A": {E: suite_rel("A", E) for E, _s in plan},
           "SUITE_B": {E: suite_rel("B", E) for E, _s in plan}}
    qsrc = base._cell(QUEUE_TAG, nb)
    queue = qsrc[qsrc.index("import json, os"):]
    queue = queue.replace(plan_literal(nb), str(plan))
    if mutate:
        before = queue
        queue = mutate(queue)
        assert queue != before, "the mutant did not change the queue cell, so the " \
                                "check that depends on it proves nothing"
    exec(base._runnable(queue), env)
    return env


def jobs_of(nb, plan=None, mutate=None):
    plan = plan_of(nb) if plan is None else plan
    return queue_env(nb, plan, mutate)["JOBS"]


# ------------------------------------------------------------------------- the ordering --
def _launch(nb, mutate=None, plan=CHEAP):
    """Boot, plant the suites, build the queue and run one tick() with no arm finished.

    Everything a caller needs is read out inside the temporary repository, because the
    directory is deleted before this returns: the job list, the names of the arms whose logs
    were opened, and the launch records.
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb2_launch_"))
    KEEP.append(tmp)
    cwd, path, sub = os.getcwd(), list(sys.path), sys.modules["subprocess"]
    old_env = os.environ.get("COLAB_CONTENT")
    try:
        os.environ["COLAB_CONTENT"] = str(tmp)
        os.chdir(tmp)
        ns = base._boot(tmp, nb=nb)
        repo = ns["REPO"]
        _plant(repo, _suites_for(plan))
        env = queue_env(nb, plan, mutate)
        env.update({"REPO": repo, "snapshot": lambda *a: None})
        tsrc = base._cell(TICK_TAG, nb)
        exec(base._runnable(tsrc[tsrc.index("def pids("):tsrc.index("\ntick()")]), env)
        env["pids"] = lambda needle: []
        env["subprocess"] = launched = base._NoLaunch()
        os.chdir(repo)
        env["tick"]()
        opened = [j["name"] for j in env["JOBS"] if (repo / j["log"]).exists()]
        return {"jobs": env["JOBS"], "max_par": env["MAX_PAR"], "opened": opened,
                "launches": launched.calls, "plan": plan}
    finally:
        sys.modules["subprocess"] = sub
        os.chdir(cwd)
        if old_env is None:
            os.environ.pop("COLAB_CONTENT", None)
        else:
            os.environ["COLAB_CONTENT"] = old_env


def test_the_verifications_wait_for_the_repair_they_read():
    for nb in NB2:
        ran = _launch(nb)
        assert ran["opened"], "%s started no arm at all" % nb.name
        assert all(n.startswith("repair") for n in ran["opened"]), (
            "%s launched %r in the first wave, before any repair JSON or checkpoint exists"
            % (nb.name, ran["opened"]))
        assert len(ran["opened"]) <= ran["max_par"], ran["opened"]


def test_a_verification_without_needs_would_launch_too_early():
    """The guard is `needs`; without it a verification starts beside the repair it reads."""
    ran = _launch(LADDER, mutate=lambda q: q.replace(NEEDS, "needs=[]"))
    assert any(n.startswith("verify") for n in ran["opened"]), (
        "removing the dependency changed nothing, so the ordering test above is not reading "
        "the queue's guard: %r" % (ran["opened"],))


def test_the_blind_suite_is_never_given_to_a_repair():
    for nb in NB2:
        for job in jobs_of(nb):
            flag = "--roots-file" if job["name"].startswith("repair") else "--roots"
            read = _posix(job["cmd"][job["cmd"].index(flag) + 1])
            if job["name"].startswith("repair"):
                assert read == suite_rel("A", job["E"]), (
                    "%s: %s repairs against %s, not the published suite"
                    % (nb.name, job["name"], read))
            else:
                assert read in (suite_rel("A", job["E"]), suite_rel("B", job["E"])), job
        reads = [_posix(j["cmd"][j["cmd"].index("--roots") + 1]) for j in jobs_of(nb)
                 if j["name"].startswith("verify")]
        assert any(r.endswith("holdoutB.json") for r in reads), (
            "%s queues no verification on the blind suite at all" % nb.name)


def test_every_arm_writes_the_file_the_next_one_waits_for():
    """A job's `needs` must name files some other job actually writes."""
    for nb in NB2:
        jobs = jobs_of(nb)
        written = {j["out"] for j in jobs} | {j["cmd"][j["cmd"].index("--ckpt-out") + 1]
                                             + "/c4_gru.pt"
                                             for j in jobs
                                             if "--ckpt-out" in j["cmd"]}
        for job in jobs:
            for need in job["needs"]:
                assert _posix(need) in {_posix(w) for w in written}, (
                    "%s: %s waits for %s, which no arm of this notebook produces"
                    % (nb.name, job["name"], need))


# ---------------------------------------------------------------------------- the budget --
def test_the_new_seeds_use_the_budget_the_depth_was_published_with():
    """The numbers come from the archived arm, so a drifting queue cell fails here."""
    arms = published()
    seen = 0
    for nb in NB2:
        for job in jobs_of(nb):
            if not job["name"].startswith("repair"):
                continue
            same = [(k, v) for k, v in sorted(arms.items()) if k[0] == job["E"]]
            if not same:
                continue                       # a depth nothing has been published at yet
            src, args = same[0][1]
            cmd = job["cmd"]
            for flag, key in (("--epochs", "epochs"), ("--max-iters", "max_iters"),
                              ("--agg-repeat", "agg_repeat"), ("--hidden", "hidden"),
                              ("--emb", "emb"), ("--branch", "branch"),
                              ("--roots", "roots"), ("--root-value", "root_value"),
                              ("--empties", "empties"), ("--stop", "stop"),
                              ("--rows", "rows"), ("--cols", "cols"), ("--win", "win")):
                got = cmd[cmd.index(flag) + 1]
                assert got == str(args[key]), (
                    "%s: a repair at %d empty cells passes %s %s, but the published arm %s "
                    "used %r" % (nb.name, job["E"], flag, got, src.name, args[key]))
                seen += 1
            assert abs(float(cmd[cmd.index("--lr") + 1]) - args["lr"]) < 1e-12, (
                "%s: --lr %s against a published %r" % (nb.name, cmd[cmd.index("--lr") + 1],
                                                         args["lr"]))
            assert _posix(cmd[cmd.index("--roots-file") + 1]) == suite_rel("A", job["E"]), cmd
    assert seen >= 30, "only %d comparisons were made, so the plan holds no published depth" \
                       % seen


def test_a_mutated_budget_would_be_caught():
    """The parity check above has to be able to fail, both on a number shared by every
    depth and on one the depths disagree about: eight was published with 20 aggregated
    repeats and the 'agg' stopping rule, twelve with 60 and 'both'."""
    args = published()[(10, 0)][1]
    assert args["epochs"] != 50, "the published arm already used 50 epochs, so mutating it " \
                                "to 50 could not prove anything"
    cmd = jobs_of(LADDER, ((10, (3,)),), mutate=lambda q: q.replace(
        "'--epochs', '200'", "'--epochs', '50'"))[0]["cmd"]
    assert cmd[cmd.index("--epochs") + 1] == "50"

    eight = published()[(8, 1)][1]
    assert (eight["agg_repeat"], eight["stop"]) != (
        published()[(12, 0)][1]["agg_repeat"], published()[(12, 0)][1]["stop"]), (
        "the depths agree on the stopping rule, so a single recipe would have been enough "
        "and the per-depth table below proves nothing")
    flat = jobs_of(LADDER, ((8, (3,)), (12, (3,))), mutate=lambda q: q.replace(
        "    8: {'agg_repeat': 20, 'stop': 'agg', 'max_iters': 12},",
        "    8: {'agg_repeat': 60, 'stop': 'both', 'max_iters': 24},"))
    repair = [j["cmd"] for j in flat if j["name"].startswith("repair")]
    assert len(repair) == 2
    c8 = repair[0]
    got = {f: c8[c8.index(f) + 1] for f in ("--agg-repeat", "--stop", "--max-iters")}
    assert got == {"--agg-repeat": "60", "--stop": "both", "--max-iters": "24"}, (
        "writing twelve's budget over eight's changed none of the three flags, so the "
        "budget table is not read per depth: %r" % (got,))


# --------------------------------------------------------------------------- the coverage --
def test_the_plan_reaches_the_seed_counts_the_error_bars_need():
    plan = {}
    for nb in NB2:
        for E, seeds in plan_of(nb):
            plan.setdefault(E, set()).update(seeds)
    have = {}
    for (E, s) in published():
        have.setdefault(E, set()).add(s)
    total = {E: have.get(E, set()) | plan.get(E, set()) for E in set(have) | set(plan)}
    assert len(total[12]) >= 10, sorted(total[12])
    assert len(total[10]) >= 6 and len(total[8]) >= 6, (sorted(total[10]), sorted(total[8]))
    for E, seeds in plan.items():
        for s in seeds:
            three = ["data/c4_end%d_s%d.json" % (E, s),
                     "data/c4_end%d_verify_s%d.json" % (E, s),
                     "data/c4_end%d_holdoutB_s%d.json" % (E, s)]
            landed = [p for p in three if (REPO / p).is_file()]
            assert len(landed) in (0, 3), (
                "seed %s at %d empties is half archived (%s): the queue would re-run its "
                "repair while a verification of the old one sits beside it" % (s, E, landed))
            if landed:
                # the queue is drained for this seed, and that is a different thing from
                # repeating work: tick() finds all three outputs and launches nothing, so a
                # reviewer who runs a finished notebook spends a session reading "queue empty"
                continue
            assert (E, s) not in published(), (
                "seed %s at %d empties is already archived; the queue would repeat it"
                % (s, E))


def test_every_planned_seed_gets_both_verifications():
    for nb in NB2:
        jobs = jobs_of(nb)
        for E, seeds in plan_of(nb):
            for s in seeds:
                mine = [j["name"] for j in jobs
                        if j["E"] == E and j["out"].endswith("_s%d.json" % s)]
                assert any(m.startswith("repair") for m in mine), (nb.name, E, s, mine)
                tags = {m.split()[1] for m in mine if m.startswith("verify")}
                assert tags == {"A", "B"}, "%s: seed %s at %d empties verifies on %r" % (
                    nb.name, s, E, sorted(tags))


def test_a_thinner_plan_would_not_reach_the_seed_count():
    """The coverage test is a promise about a number, so a smaller plan must break it."""
    literal = plan_literal(LADDER)
    thin = literal.replace("(7, 8, 9)", "(7,)")
    assert thin != literal, "the twelve-empty ladder no longer carries the seeds it was " \
                           "written for, so the coverage test above reads another document"
    at12 = [set(s) for E, s in (tuple(t) for t in eval(thin)) if E == 12][0]
    assert len(at12 | set(range(7))) < 10, sorted(at12)


# -------------------------------------------------------------------- when it downloads --
def test_the_finisher_writes_nothing_while_a_depth_is_incomplete():
    """Building a zip and opening a download window takes wall time away from the arms.

    The document is booted and every cell run against a stubbed subprocess, so no arm ever
    lands and each depth stays incomplete: the tick, the pump and the results cell all reach
    `snapshot`, and none of them may hand the browser a file - even though the two suites the
    sampling cell read are themselves artefacts, which is exactly what the older queue ships
    on its first pass. The sweep cell is the one place that ignores this, so the run stops
    before it. Two mutants close the door: a document that drops the flag ships, and a
    document that stops asking whether the depth is whole ships too.
    """
    text, _stub, ns, _repo, zips = run_all(LADDER, arms=None, until=SHIP_TAG)
    assert ns["PUSH_AT_END"] is True, "el finisher perdio su bandera"
    assert zips == [], "el finisher descargo %r con brazos todavia en vuelo" % zips
    assert "descarga final" in text, "la celda de resultados salteo su pasada de entrega"

    ignore_flag = lambda s: s.replace("PUSH_AT_END = True", "PUSH_AT_END = False")
    loud = run_all(LADDER, arms=None, until=SHIP_TAG, mutate=ignore_flag)[4]
    assert loud, "una escalera que ignora la bandera es igual a la anterior: nadie lee el flag"

    never_whole = lambda s: s.replace("if not PUSH_AT_END or _done(depth)",
                                      "if not PUSH_AT_END or True")
    loud2 = run_all(LADDER, arms=None, until=SHIP_TAG, mutate=never_whole)[4]
    assert loud2, "la comprobacion de profundidad entera no decide nada: se puede borrar"


def test_the_finisher_still_ships_every_depth_at_the_end():
    """Quiet during training is not the same as silent: nothing may be stranded on the runtime."""
    _text, _stub, _ns, _repo, zips = run_all(LADDER, arms=None)
    assert sorted(zips) == sorted("c4_finish_end%d_results.zip" % E for E, _s in CHEAP), zips
    assert not any("holdout" in z for z in zips), zips


# ------------------------------------------------------------------------------ the zip --
def _carried():
    src = base._bundle_source(LADDER)
    ns = {}
    exec(src[: src.index("_archive =")], ns)
    return set(ns["BUNDLE_FILES"])


def test_the_bundle_carries_every_suite_the_notebooks_read():
    """A suite is an input to the experiment, not a detail.

    Two accounts, or one account after a runtime restart, only answer the same question if
    they read the same forty positions, so every suite the repository holds must travel
    inside the notebook: a bare runtime that re-sampled one would no longer share it with
    the arms already in the paper.
    """
    carried = _carried()
    for nb in NB2:
        for E, _seeds in plan_of(nb):
            for which in ("A", "B"):
                rel = suite_rel(which, E)
                if not (REPO / rel).exists():
                    continue        # nothing sampled at this depth yet; the next test covers it
                assert rel in carried, (
                    "%s is not in the bundle, so a bare runtime would sample its own copy of "
                    "a suite the repository already has: %s" % (rel, nb.name))
    assert len(carried) >= 11, carried


def test_a_depth_the_repository_has_not_sampled_yet_is_resampled_identically():
    """The fourteen-empty suites exist only once a runtime or the laptop makes them.

    That is not a hole in the promise above as long as the two ways of making them agree:
    `random_games` is deterministic in its sample seed, so the notebook and
    `runs/make_holdout_suites.py` must draw the same stream - same first seed, same waves,
    same per-wave game cap. If they ever differ, the committed file and the runtime's file
    are different suites and the holdout argument is two experiments.
    """
    script = (REPO / "runs" / "make_holdout_suites.py").read_text(encoding="utf-8")
    consts = {n: int(re.search(r"^%s = (\d+)" % n, script, re.M).group(1))
              for n in ("SIZE", "B_OFFSET", "B_WAVES", "PER_GAMES")}
    for nb in NB2:
        src = base._cell(SUITES_TAG, nb)
        for name in ("B_SIZE", "B_WAVES", "A_WAVES"):
            got = int(re.search(r"^%s = (\d+)" % name, src, re.M).group(1))
            want = {"B_SIZE": consts["SIZE"], "B_WAVES": consts["B_WAVES"],
                    "A_WAVES": 1}[name]
            assert got == want, "%s: the notebook sets %s = %d, the sampler uses %d" % (
                nb.name, name, got, want)
        assert "_grow(b_path, E, B_SIZE, _boards(A), %d + E, B_WAVES, %d, 'suite B')" % (
            consts["B_OFFSET"], consts["PER_GAMES"]) in src, (
            "%s: the holdout is not sampled the way runs/make_holdout_suites.py samples it"
            % nb.name)
        assert "_grow(a_path, E, 40, set(), 0, A_WAVES," in src, (
            "%s: a fresh suite A is not drawn from the published sample seed 0" % nb.name)
        cap = int(re.search(r"_grow\(b_path, E, B_SIZE, _boards\(A\), \d+ \+ E, B_WAVES, "
                            r"(\d+),", src).group(1))
        assert cap >= consts["PER_GAMES"], (
            "%s gives a runtime fewer games per wave than the sampler that wrote the "
            "committed suites, so it would give up where the repository succeeded" % nb.name)


def test_a_runtime_sampling_a_committed_depth_gets_the_same_positions():
    """The agreement above is a claim about seeds, so it is checked against the data.

    The first forty draw roots of stream 0 at eight empty cells are the published suite, and
    the first forty unseen draw roots of stream 500 + 8 are the blind holdout: the exact
    lists a bare runtime would write if the committed files were missing. A change to
    `random_games`, to a sample seed or to the suites themselves lands here.
    """
    sys.path.insert(0, str(REPO / "src"))
    try:
        import c4_game as g
        from c4_endgame import random_games
    finally:
        sys.path.pop(0)
    g.configure(6, 7, 4)
    oracle = g.Oracle()
    a = json.loads((REPO / suite_rel("A", 8)).read_text(encoding="utf-8"))
    first_three, _games = random_games(oracle, 8, 3, 0, 0, max_games=400000)
    assert [r["board"] for r in first_three] == [r["board"] for r in a[:3]], (
        "sampling eight-empty suite A from seed 0 no longer reproduces the committed file")
    b = json.loads((REPO / suite_rel("B", 8)).read_text(encoding="utf-8"))
    seen = {tuple(r["board"]) for r in a}
    fresh, _games = random_games(oracle, 8, 40, 500 + 8, 0, max_games=400000)
    unseen = [r for r in fresh if tuple(r["board"]) not in seen]
    assert len(unseen) >= len(b), "the committed holdout has %d roots, the wave only %d" % (
        len(b), len(unseen))
    assert [r["board"] for r in unseen[:len(b)]] == [r["board"] for r in b], (
        "sampling the blind holdout from seed %d no longer reproduces the committed file"
        % (500 + 8))


def test_the_snapshot_ships_everything_a_plan_names():
    """The zip is the only way an arm leaves the runtime, so it must not miss a file type."""
    plan = ((12, (3, 4)),)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb2_snap_"))
    cwd, old = os.getcwd(), os.environ.get("COLAB_CONTENT")
    try:
        os.environ["COLAB_CONTENT"] = str(tmp)
        os.chdir(tmp)
        ns = base._boot(tmp, nb=LADDER)
        repo = ns["REPO"]
        os.chdir(repo)              # as bootstrap() does on a runtime
        src = base._cell(PLAN_TAG, LADDER).replace(plan_literal(LADDER), str(plan))
        exec(base._runnable(src), ns)
        planted = ["data/c4_end12_s3.json", "data/c4_end12_verify_s3.json",
                   "data/c4_end12_holdoutB_s3.json", "data/c4_end12_s4.json",
                   "data/c4_endgame_roots_12_40_holdoutB.json",
                   "runs/c4_end12_s3/c4_gru.pt", "runs/c4_end12_s3/c4_gru_base.pt",
                   "runs/_colab_c4_end12_s3.log", "runs/_colab_c4_end12_holdoutB_s3.log"]
        for rel in planted:
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x" if rel.endswith(".pt") else "{}", encoding="utf-8")
        got = {_posix(p) for p in ns["artefacts"](12)}
        assert set(planted) <= got, sorted(set(planted) - got)
        assert not any("s9" in g for g in got), "the plan does not name that seed"
        # The depth is deliberately left incomplete - seed 4 has no verification, and neither
        # seed has all three logs - so the queue's own pass must refuse to package it and only
        # the sweep cell may. Both halves are the promise: a zip built from half an arm would
        # read as a finished arm, and a sweep that ships nothing loses the run.
        assert ns["snapshot"]() == [], "una profundidad a medias llego al navegador"
        archives = ns["ship"]()
        assert archives, "the sweep returned nothing with files on disk"
        with zipfile.ZipFile(archives[0]) as z:
            names = set(z.namelist())
        assert set(planted) <= names, sorted(set(planted) - names)
        assert all(n.startswith(("data/", "runs/")) for n in names), sorted(names)
        assert archives[0].name == "c4_finish_end12_results.zip", archives[0].name
    finally:
        os.chdir(cwd)
        if old is None:
            os.environ.pop("COLAB_CONTENT", None)
        else:
            os.environ["COLAB_CONTENT"] = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_every_job_writes_a_file_the_zip_ships():
    """The queue names an arm's files and the artefacts cell collects them separately.

    Two spellings of the same name drift: the holdout verifications wrote
    `_colab_c4_end12_holdoutB_s3.log`, which the glob built from the repair arms did not
    match, so an arm that finished on the runtime never reached the browser. This reads the
    names off the queue's own job list instead of off a list written next to the test.
    """
    plan = ((8, (3,)), (12, (3, 4)))
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb2_ship_"))
    cwd, old = os.getcwd(), os.environ.get("COLAB_CONTENT")
    try:
        os.environ["COLAB_CONTENT"] = str(tmp)
        os.chdir(tmp)
        ns = base._boot(tmp, nb=LADDER)
        repo = ns["REPO"]
        os.chdir(repo)
        src = base._cell(PLAN_TAG, LADDER).replace(plan_literal(LADDER), str(plan))
        exec(base._runnable(src), ns)
        jobs = jobs_of(LADDER, plan)
        for job in jobs:
            for rel in (job["out"], job["log"]):
                p = repo / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("{}", encoding="utf-8")
            if job["name"].startswith("repair"):
                ck = repo / job["cmd"][job["cmd"].index("--ckpt-out") + 1]
                ck.mkdir(parents=True, exist_ok=True)
                (ck / "c4_gru.pt").write_text("x", encoding="utf-8")
        for E in sorted({j["E"] for j in jobs}):
            have = {_posix(p) for p in ns["artefacts"](E)}
            for job in (j for j in jobs if j["E"] == E):
                assert _posix(job["out"]) in have, (E, job["name"], sorted(have))
                assert _posix(job["log"]) in have, (
                    "%s at %d empty cells: its log %s is not in the download"
                    % (job["name"], E, job["log"]))
            ckpts = [g for g in have if g.endswith(".pt")]
            assert len(ckpts) == len([j for j in jobs if j["E"] == E
                                      and j["name"].startswith("repair")]), ckpts
    finally:
        os.chdir(cwd)
        if old is None:
            os.environ.pop("COLAB_CONTENT", None)
        else:
            os.environ["COLAB_CONTENT"] = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_two_colab_documents_do_not_download_over_each_other():
    """Both run at once, on two accounts, in one browser, so the zip names must differ.

    The ladder names a zip after the depth it finished and the reads document after the section
    of its queue, so what has to hold is that no filename of one can ever be a filename of the
    other: each document builds its names from its own slug, and the two literal heads - every-
    thing before the first hole in the format - must not be prefix-comparable. A document left
    with a generator placeholder would print the placeholder into a filename, so that is refused
    here too.
    """
    heads = {}
    for nb in NB2 + [FINISH]:
        src = "\n".join(cell for _i, cell in base._code_cells(nb))
        assert "@@" not in src, "%s still holds a generator placeholder" % nb.name
        slug = re.search(r"^SLUG = '([^']+)'", src, re.M).group(1)
        patterns = re.findall(r'''c4_%s([^'"]*)results\.zip''', src)
        assert patterns, "%s builds no filename out of its own slug" % nb.name
        heads[nb.name] = sorted({"c4_" + slug + part.split("%")[0] for part in patterns})
    ladder, reads = heads[LADDER.name], heads[FINISH.name]
    assert all(h.startswith("c4_") for h in ladder + reads), (ladder, reads)
    assert not any(a.startswith(b) or b.startswith(a) for a in ladder for b in reads), (
        ladder, reads)

if __name__ == "__main__":
    names = sorted(n for n in globals() if n.startswith("test_"))
    failed = []
    for n in names:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                globals()[n]()
            print("ok    ", n)
        except Exception as exc:
            failed.append(n)
            print("FAIL  ", n, type(exc).__name__, exc)
    print("%d passed, %d failed" % (len(names) - len(failed), len(failed)))
    raise SystemExit(1 if failed else 0)
