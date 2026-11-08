"""Contract tests for `runs/_local_queue.py`, the driver that runs the notebook's queue here.

The driver exists because nothing in the queue's derivation is Colab-specific: the budget is read
off archived arms, the suites are committed files, and a memo-free read is a recursion that runs
anywhere. What is specific to the driver is that it builds its own command line and decides its own
queue, so two promises could not be checked before this file existed:

1. the command. A flag the driver drops or spells differently would train an arm that is not the
   published protocol and still write a JSON with the right name - the failure is a number in the
   paper, not a crash. So every read command is compared against the one the notebook's own
   `verify()` builds from the same plan row, and every repair command against both the notebook's
   `repair()` and the archived `args` of a published arm at that depth. The archives are the anchor
   that does not drain: `reads()` is derived from `LADDER_ARMS`, so these comparisons still hold
   once every output is on disk and the notebook queues nothing.
2. the queue. `owed()` adds a paired base read only when the weights it reads are on this disk,
   because `.gitignore` keeps checkpoints out of the repository and a read of weights nobody saved
   is a job that waits forever. The test changes what is on the fake disk and requires the answer
   to change with it, so a driver that always queued the read - or never queued it - fails one of
   the three states rather than passing all of them.

Nothing trains here and no file is written: the commands are compared as lists, and the disk is a
set of paths handed to `reads_plan.exists`.

Run with pytest, or directly: python tests/test_local_queue.py
"""
import json
import os
import pathlib
import sys

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO / "colab"))
sys.path.insert(0, str(REPO / "runs"))
sys.path.insert(0, str(REPO / "tests"))
import reads_plan as rp                                               # noqa: E402
import build_reads_notebooks as gen                                   # noqa: E402
import test_colab_notebook as base                                    # noqa: E402
import _local_queue as lq                                             # noqa: E402


def notebook_jobs(reads, arms, budget, threads=1):
    """The JOBS the notebook's queue cell builds for these rows, with no runtime involved.

    The cell is the generator's own template, cut where `jobs_of` cuts it: everything after the
    JOBS list is printing, and the two functions that build commands are what is under comparison.
    """
    src = gen.QUEUE
    cut = src.index("JOBS = ")
    src = src[:cut] + "\n" + src[cut:src.index("\n_suites")]
    src = (src.replace("@@READS@@", gen.reads_literal(reads))
              .replace("@@ARMS@@", gen.arms_literal(arms))
              .replace("@@BUDGET@@", json.dumps(budget)))
    ns = {"SUITES": json.loads(gen.suites_literal(reads + arms)), "THREADS": threads}
    exec(base._runnable(src), ns)
    return ns["JOBS"]


def disagree(build, rows, jobs):
    """Outputs whose command the driver spells differently from the notebook, argv[0] aside."""
    by_out = {r["out"]: r for r in rows}
    bad = []
    for j in jobs:
        row = by_out[j["out"]]
        if build(row, 1)[1:] != j["cmd"][1:]:
            bad.append(j["out"])
    return bad


def flags_of(cmd):
    """A command line as a dict, after asserting that it is only flags and their values.

    `--no-controls` is a bare flag, so the pairing cannot be a stride over the whole list.
    """
    out, i = {}, 2
    while i < len(cmd):
        key = cmd[i]
        assert key.startswith("--"), "a positional argument would make this parse meaningless: %r" % cmd
        if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            out[key] = cmd[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def archived_repair_row(E):
    """A repair row for a published arm at this depth, rebuilt from the arm's own archive.

    The oldest arms are not usable as fixtures: eight-cell seed 0 sampled its suite in-process
    before `--roots-file` existed, and seeds 1 and 2 are kept in directories from before the
    per-depth naming, which is the naming the notebook's `repair()` spells.
    """
    for arm in rp.LADDER_ARMS[E]:
        d = json.loads((REPO / rp.arm_json(arm)).read_text(encoding="utf-8"))
        if d["args"].get("roots_file") and arm == "c4_end%d_s%d" % (E, d["args"]["seed"]):
            return dict(kind="arm-repair", arm=arm, E=E, seed=d["args"]["seed"], suite="A",
                        suite_file=d["args"]["roots_file"].replace("\\", "/"),
                        ckpt="runs/%s/c4_gru.pt" % arm, out=rp.arm_json(arm), needs=[],
                        group="arms", gate="end%d" % E, arm_json=rp.arm_json(arm),
                        key="A%d" % E), d
    raise AssertionError("no archived arm at depth %d names a roots file and a modern directory" % E)


def test_every_read_the_driver_runs_is_the_command_the_notebook_would_run():
    rows = rp.reads()
    jobs = notebook_jobs(rows, [], {})
    assert len(jobs) == len(rows) > 0, "the plan produced no read to compare"
    assert disagree(lq.cmd_for, rows, jobs) == []

    # the same comparison with one flag of the driver's changed: a check that ignored the argument
    # list would pass this too, and then the promise above would be covered by nothing
    def mutant(row, threads):
        return [c if c != "120" else "60" for c in lq.cmd_for(row, threads)]

    assert len(disagree(mutant, rows, jobs)) == len(jobs), "the comparison did not notice --orders"


def test_a_repair_the_driver_runs_is_the_published_budget_flag_for_flag():
    for E in sorted(rp.LADDER_ARMS):
        row, archive = archived_repair_row(E)
        args = archive["args"]
        cmd = lq.cmd_for(row, 1)
        got = flags_of(cmd)
        for f in rp.ARM_FLAGS:
            assert got["--" + f.replace("_", "-")] == str(args[f]), (E, f, got, args[f])
        for f in ("rows", "cols", "win", "empties", "seed"):
            assert got["--" + f.replace("_", "-")] == str(args[f]), (E, f)
        # threads is this machine's, not the budget's: the ten-cell seed 0 ran at three and the
        # Colab ladder at one, which is why ARM_FLAGS leaves it out
        assert got["--threads"] == "1"
        assert got["--roots-file"] == row["suite_file"] == rp.suite("A", E)
        assert got["--out"] == rp.arm_json(row["arm"]) == args["out"]
        assert got["--ckpt-out"] == "runs/%s" % row["arm"] == args["ckpt_out"]
        assert got["--no-controls"] is True, "a repair that also trains controls is a different job"
        jobs = notebook_jobs([], [row], rp.arm_budget(E))
        assert len(jobs) == 1 and disagree(lq.cmd_for, [row], jobs) == [], E

    row, archive = archived_repair_row(12)

    def mutant(r, threads):
        return [c if c != str(archive["args"]["agg_repeat"]) else "20"
                for c in lq.cmd_for(r, threads)]

    assert disagree(mutant, [row], notebook_jobs([], [row], rp.arm_budget(12))) == [row["out"]]


def test_a_paired_read_is_owed_only_once_the_weights_it_reads_are_on_disk(monkeypatch):
    arm = rp.LADDER_ARMS[12][0]
    blind = "data/c4_end12_holdoutB_s0.json"
    read = next(r for r in rp.reads() if r["arm"] == arm and r["kind"] == "base-on-A")
    # reads() only offers the paired base reads of an arm whose blind read is already filed
    assert rp.exists(blind), "the fixture arm has no blind read, so it has no base read to owe"

    def owed_with(disk):
        monkeypatch.setattr(rp, "exists", lambda rel: rel in disk)
        return [r["out"] for r in lq.owed()]

    with_json = {read["arm_json"], blind}
    assert read["out"] not in owed_with(with_json), "a read of weights nobody saved was queued"
    assert read["out"] in owed_with(with_json | {read["ckpt"]}), "the weights are here and it waits"
    assert read["out"] not in owed_with(with_json | {read["ckpt"], read["out"]}), "already measured"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))