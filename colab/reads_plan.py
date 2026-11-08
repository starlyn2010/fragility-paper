"""Which reads and which arms a free runtime still owes, as data that one file owns.

Three places want this list. `runs/_reads_audit.py` prints it, `colab/build_reads_notebooks.py`
turns it into the one queue that drains it, and `tests/test_reads_notebooks.py` checks that what
the notebook on disk queues is still exactly what this module says is pending. It lives in one
module because the failure it prevents is silent: a notebook hand-written from a remembered list
of reads keeps its old plan after the repository changes, spends hours of Colab and delivers
nothing that was missing.

A read is a call of `src/c4_endgame_verify.py`: one checkpoint, one suite, one output JSON. The
three kinds are the three questions the paper asks of them -

* `repaired-on-A` re-derives, without the loop's memo, the certificate a repair arm claimed for
  the suite it was certified over;
* `repaired-on-B` asks the same weights for the blind suite the repair never saw, which is the
  measurement behind `tab:blind`;
* `base-on-A` and `base-on-B` read the pre-repair policy of the same arm on both suites, which is
  what separates the repair's gain from the depth's difficulty;
* the `union-*` reads belong to `sec:union`, whose arms were certified over suite A followed by
  suite B, so their read has three suites: the two halves and the eighty-position whole.

Reads are grouped by the question they answer, which in the notebook is also the download gate: a
group's zip leaves for the browser when its last read lands, so a session that dies at three
quarters of the queue still delivered what it did.

The plan has a second kind of row. Three arms at fourteen empty cells were never trained, so they
are not reads - each is a repair followed by the two reads that make it a published arm - and they
queue at the tail of the same notebook, because a reviewer should not have to know which document
holds which half of what the repository owes.
"""
from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = "data"
RUNS = "runs"

# The ladder arms, per depth, in the order the paper lists them. The names are the directories
# the queues saved them under, and they are not regular: depth eight begins with the seed-0 arm
# of the first experiment and the two seeds that ran before the per-depth naming existed.
LADDER_ARMS = {
    8: ["c4_end_ctrl_s0", "c4_end_s1", "c4_end_s2",
        "c4_end8_s3", "c4_end8_s4", "c4_end8_s5"],
    10: ["c4_end10_s%d" % s for s in range(6)],
    12: ["c4_end12_s%d" % s for s in range(10)],
}
UNION_ARMS = ["c4_end8U_s%d" % s for s in range(3)]

# What no runtime has trained yet, as (empty cells, seeds). Fourteen is the only depth with no
# arm of its own, so its three arms queue at the tail of the notebook that carries the reads:
# each is a repair against suite A and then the two reads that make it a published arm.
ARMS = ((14, (0, 1, 2)),)

# The depth a planned arm inherits its budget from. Fourteen has no archived arm that declares
# one, so the numbers come off the twelve-cell archives rather than off a table someone types.
ARM_BUDGET_FROM = {14: 12}

# The flags that are a budget: `empties` and `roots_file` say which depth a run is, not what it
# was allowed to spend.
ARM_FLAGS = ("agg_repeat", "stop", "max_iters", "epochs", "lr", "hidden", "emb", "branch",
             "roots", "root_value")

# the depth each arm belongs to, read off its own archived args rather than off the table above
DEPTH_OF = {}
for _E, _names in LADDER_ARMS.items():
    for _n in _names:
        DEPTH_OF[_n] = _E
for _n in UNION_ARMS:
    DEPTH_OF[_n] = 8

GROUPS = ("union", "closure", "pairing")


def suite(which, E):
    """Repo-relative path of a committed suite: published A, blind B, or the union U."""
    name = {
        "A": "data/c4_endgame_roots_%d_40_s0.json" % E,
        "B": "data/c4_endgame_roots_%d_40_holdoutB.json" % E,
        "U": "data/c4_endgame_roots_%d_80_AplusB.json" % E,
    }[which]
    return name


def arm_json(arm):
    """The repair arm's own archived JSON, which is where a read's expectation comes from."""
    return "data/%s.json" % arm


def _out_a(arm, E, s):
    """Where this arm's memo-free read on suite A is filed.

    Two names exist because two eras of the queue wrote them: the first depth-eight experiment
    saved `c4_end_verify_s0.json`, and everything since saves `c4_end<E>_verify_s<s>.json`.
    """
    if (E, s) == (8, 0):
        return "data/c4_end_verify_s0.json"
    return "data/c4_end%d_verify_s%d.json" % (E, s)


def reads():
    """Every read the paper reports or would need, done or not, oldest depth first."""
    out = []

    def add(kind, arm, which, ckpt, dest):
        out.append(dict(kind=kind, arm=arm, E=DEPTH_OF[arm], suite=which,
                        suite_file=suite(which, DEPTH_OF[arm]), ckpt=ckpt, out=dest,
                        group=group_of(kind), gate=group_of(kind), arm_json=arm_json(arm),
                        key=which + str(DEPTH_OF[arm])))

    for E, names in LADDER_ARMS.items():
        for arm in names:
            s = int(arm.rsplit("_s", 1)[1])
            rep = "%s/%s/c4_gru.pt" % (RUNS, arm)
            base = "%s/%s/c4_gru_base.pt" % (RUNS, arm)
            add("repaired-on-A", arm, "A", rep, _out_a(arm, E, s))
            blind = "data/c4_end%d_holdoutB_s%d.json" % (E, s)
            add("repaired-on-B", arm, "B", rep, blind)
            # a paired base read only means something next to the blind read of the same arm,
            # so the pairing follows tab:blind rather than every arm that ever ran
            if exists(blind):
                add("base-on-A", arm, "A", base,
                    "data/c4_end%d_base_s%d_onA.json" % (E, s))
                add("base-on-B", arm, "B", base,
                    "data/c4_end%d_base_s%d_onB.json" % (E, s))

    for arm in UNION_ARMS:
        for which, tag in (("A", "A"), ("B", "B"), ("U", "U")):
            add("union-repaired-on-" + which, arm, which,
                "%s/%s/c4_gru.pt" % (RUNS, arm), "data/%s_%s.json" % (arm, tag))
        for which, tag in (("A", "A"), ("B", "B"), ("U", "U")):
            add("union-base-on-" + which, arm, which,
                "%s/%s/c4_gru_base.pt" % (RUNS, arm), "data/%s_base_%s.json" % (arm, tag))
    return out


def group_of(kind):
    if kind.startswith("union-"):
        return "union"
    if kind.startswith("base-on-"):
        return "pairing"
    return "closure"


def exists(rel):
    return (REPO / rel).is_file()


def pending(all_reads=None):
    """The reads whose output JSON is not in the repository."""
    return [r for r in (all_reads or reads()) if not exists(r["out"])]


def shippable(group, todo=None):
    """Pending reads of one group whose checkpoint is on this disk, so they can be carried."""
    return [r for r in (todo if todo is not None else pending())
            if r["group"] == group and exists(r["ckpt"])]


def needs_base_fit(group, todo=None):
    """Pending reads of one group whose checkpoint was never saved; --base-fit could re-derive it."""
    return [r for r in (todo if todo is not None else pending())
            if r["group"] == group and not exists(r["ckpt"])]


def checkpoints(todo):
    """The checkpoint files a set of reads loads, in the order the reads are queued."""
    seen, out = set(), []
    for r in todo:
        if r["ckpt"] not in seen:
            seen.add(r["ckpt"])
            out.append(r["ckpt"])
    return out


def expected(read, archive=None):
    """What the archived arm already claims, so a read that disagrees is a read of wrong weights.

    A repaired read is checked against the certificate the loop ended with; a base read against
    the base certificate the loop began with. Both are on the same suite as the read, except that
    a suite-A or suite-B read of a union arm has no archived number at all - the loop certified
    eighty positions - so only the identity across the three suites constrains those.

    `archive` is the arm's parsed JSON when the caller already holds it. A gate that trains an arm
    ships the arm and its two reads together, so at ingestion time the archive is in the zip and
    not yet on disk; reading it from disk there would find nothing and silently check no read.
    """
    if archive is None:
        p = REPO / read["arm_json"]
        if not p.is_file():
            return None
        archive = json.loads(p.read_text(encoding="utf-8"))
    d = archive
    repaired = read["kind"].startswith(("arm-read-on-", "repaired-on-", "union-repaired-"))
    if repaired and "repair" not in d:
        return None
    cert = d["repair"]["final"] if repaired else d["base"]["certificate"]
    roots_file = d["args"].get("roots_file", "").replace("\\", "/")
    if not roots_file:
        # The first arms sampled their own suite in-process, before --roots-file existed, so the
        # archive does not name a file. At seed 0 that sample is the published suite of its depth
        # position for position, which is what the queue points these reads at; a sample taken at
        # another seed is forty different positions and has nothing archived to be compared with.
        roots_file = suite("A", d["game"]["empties"]) if d["args"]["seed"] == 0 else ""
    archived = ("" if not roots_file else
                "U" if "AplusB" in roots_file else
                "B" if "holdoutB" in roots_file else "A")
    return {"cert": cert, "suite": archived, "comparable": read["suite"] == archived,
            "rounds": len(d["repair"]["log"]),
            "histories": len(d["repair"]["aggregate"])}


def size_of(group):
    """Bytes the notebooks of a group must carry: one copy of every checkpoint it queues."""
    todo = shippable(group)
    return sum((REPO / c).stat().st_size for c in checkpoints(todo))


# --------------------------------------------------------------------------- the arms ----
def arm_budget(E):
    """The flags a planned depth must run under, read off the archives it inherits from.

    Typed budgets rot: a table a generator restates can silently disagree with the runs it says
    it describes. So these are the archived `args` of the arms already published at the donor
    depth, and the function refuses a donor whose own arms disagree with each other, because
    then there is no single budget to inherit and a new seed is not comparable with anything.
    """
    ref = ARM_BUDGET_FROM.get(E, E)
    seen = []
    for arm in LADDER_ARMS.get(ref, []):
        p = REPO / arm_json(arm)
        if not p.is_file():
            continue
        args = json.loads(p.read_text(encoding="utf-8"))["args"]
        assert args["empties"] == ref, "%s: filed under depth %s but its own args say %s" % (
            arm_json(arm), ref, args["empties"])
        seen.append({f: args[f] for f in ARM_FLAGS})
    if not seen:
        raise SystemExit("no archived arm at depth %s, so nothing states the budget depth %s "
                         "would inherit - a planned arm would run under numbers nobody could "
                         "check it against" % (ref, E))
    if any(s != seen[0] for s in seen):
        raise SystemExit("the arms at depth %s disagree on their budget: %r, so there is no "
                         "published number for depth %s to repeat" % (ref, seen, E))
    return seen[0]


def arm_rows():
    """A job row per output a planned arm still owes, in the order the queue runs them.

    The repair row carries no `needs`, because it makes the policy; each of its two reads waits
    for the repair's own JSON and for the checkpoint that JSON saved. A row whose output is
    already in `data/` is gone from the list, which is what lets a half-landed depth - seed 0
    whole, seed 1 trained but unread - resume at the reads instead of re-running a repair.
    """
    rows = []
    for E, seeds in ARMS:
        for s in seeds:
            arm = "c4_end%d_s%d" % (E, s)
            rep = "data/%s.json" % arm
            ckpt = "%s/%s/c4_gru.pt" % (RUNS, arm)
            plan = [("arm-repair", "A", rep, [], ckpt),
                    ("arm-read-on-A", "A", "data/c4_end%d_verify_s%d.json" % (E, s),
                     [rep, ckpt], ckpt),
                    ("arm-read-on-B", "B", "data/c4_end%d_holdoutB_s%d.json" % (E, s),
                     [rep, ckpt], ckpt)]
            for kind, which, dest, needs, load in plan:
                if exists(dest):
                    continue
                rows.append(dict(kind=kind, arm=arm, E=E, seed=s, suite=which,
                                 suite_file=suite(which, E), ckpt=load, out=dest,
                                 needs=needs, group="arms", gate="end%d" % E,
                                 arm_json=rep, key=which + str(E)))
    return rows

