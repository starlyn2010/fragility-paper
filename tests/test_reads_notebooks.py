"""Contract tests for the one notebook that finishes everything the repository owes.

`colab/build_reads_notebooks.py` writes `colab/c4_finish_all.ipynb`: every memo-free read the paper
still wants, and the three fourteen-empty arms that have never run anywhere, in one queue. The shared
machinery - bundle, boot, capacity cell, monitor, pump - is `test_colab_notebook.py`'s problem and runs
over this document too. What is specific to it, and could not be tested before this file existed, is
seven promises:

1. the plan. A queue hand-written from a remembered list of work keeps its old plan after the
   repository changes and spends hours of free CPU delivering nothing that was missing. So the reads
   the document queues are compared against `colab/reads_plan.py` recomputed from the files on disk,
   and its arm rows against `reads_plan.arm_rows()` - and a read or an arm the repository already has
   is never queued again.
2. the weights. A read of somebody else's checkpoint is not a result, it is a wrong number with a real
   file name. Each embedded checkpoint has to hash to the file on this disk, have the size the
   notebook claims, and be the path the archived arm records for its own weights - and it has to
   arrive in a cell of its own, because a document missing one cell of bytes would otherwise queue
   reads that wait forever.
3. the suites. The blind suite must share no board with the published one - at fourteen empties too,
   where the arms this document trains have never been read - and the union must be exactly the
   published one followed by the blind one, or `sec:union`'s argument is a file-naming accident.
4. the budget. A fourteen-empty arm is comparable with the published curve only under the numbers the
   twelve-empty arms were published with, and those are read out of the archives rather than typed, so
   the repair command is checked flag by flag against what `reads_plan.arm_budget` says.
5. the arithmetic of the check. Every read carries the certificate the arm logged beside it - or, for
   an arm this document itself trains, the JSON the runtime just wrote - and the results cell refuses
   a read that disagrees, or a union that stops being the minimum of its two halves.
6. when it downloads. Nothing may be packaged while a job of that gate is in flight, so a gate's zip
   opens only when every one of its outputs is on disk. The three read gates therefore land in the
   browser before the first repair starts, and the only zip that can open during training is the
   fourteen-cell one - which is also the only place those weights will ever exist, because `*.pt` is
   not in git.
7. that it is the one document. The prose counts the jobs the queue actually holds, the four shared
   cells are byte-for-byte the tested ones from `colab/c4_endgame_scaling.ipynb`, no other notebook in
   `colab/` queues this same work, nothing mounts Drive or asks for credentials, and the generator's
   own `--check` refuses a cell hand-edited inside the notebook.

Each promise is also read against a mutant that breaks it the way an accident would, so a green run
means the cells really refuse rather than really print - and the refusals are checked by their message
as well as by the fact of stopping. Nothing trains here: the launch cell sees a stub `subprocess`,
because the machine running this is the researcher's laptop. The fabricated verifier JSONs are built
from the arms' own certificates, and for a union arm the two halves are made to add up to the whole -
otherwise the notebook would be refusing for a reason this file invented.

Run with pytest, or directly: python tests/test_reads_notebooks.py
"""
import atexit
import contextlib
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "colab"))
import build_followup_notebooks as bf                                 # noqa: E402
import test_colab_notebook as base                                    # noqa: E402
import test_followup_notebooks as f2                                  # noqa: E402
import reads_plan as rp                                               # noqa: E402
import build_reads_notebooks as gen                                   # noqa: E402

NB = gen.OUT_DIR / gen.NAME
# Each anchor ends in the bracket or quote its literal opens with: a bare name is also the tail of
# another cell's prose, and _cell hands back the first code cell that contains the tag.
GATES_TAG = "\nGATES = {"
WEIGHTS_TAG = "\nWEIGHTS = {"
PUT_TAG = "\n_put('"
DONE_TAG = "faltan = sorted(set(WEIGHTS)"
SUITES_TAG = "\nSUITES = {"
SUITES_E_TAG = "\nSUITES_E = {"
READS_TAG = "\nREADS = ["
ARMS_TAG = "\nARMS = ["
BUDGET_TAG = "\nBUDGET = {"
QUEUE_TAG = "def verify(src_row)"
EXPECTED_TAG = "\nEXPECTED = {"
MONITOR_TAG = "if not [j for j in JOBS if not os.path.exists(j['out'])]:"
SHIP_TAG = "suelta = ship()"
PUMP_TAG = base.PUMP_TAG

VERIFY_SCRIPT = (REPO / "src" / "c4_endgame_verify.py").read_text(encoding="utf-8")
TRAIN_SCRIPT = (REPO / "src" / "c4_endgame.py").read_text(encoding="utf-8")
KEEP = []
KEEP_DOC = []
# one arm repaired on both suites and one arm certified over the union: between them the three read
# groups, over four checkpoints.
SHAPES = ("c4_end8_s3", "c4_end8U_s1")


@atexit.register
def _clean_up():
    """The booted directories outlive a run so a caller can assert on what the cells left behind."""
    for d in KEEP:
        shutil.rmtree(d, ignore_errors=True)


def _posix(p):
    return str(p).replace(os.sep, "/")


def zip_name(gate_fragment):
    """The file a gate's zip is handed to the browser under, as the packer cell spells it."""
    return "c4_%s_%s_results.zip" % (gen.SLUG, gate_fragment)


def literal(anchor, nb=NB):
    """The JSON literal a cell assigns to `anchor`, parsed rather than eval'd."""
    src = base._cell(anchor, nb)
    return json.JSONDecoder().raw_decode(src, src.index(anchor) + len(anchor) - 1)[0]


def queued(nb=NB):
    """The read rows, as the document's own queue cell spells them."""
    return literal(READS_TAG, nb)


def armed(nb=NB):
    """The fourteen-cell rows: one repair and two reads per seed, as the cell spells them."""
    return literal(ARMS_TAG, nb)


def shape_plan():
    """The rows the shape document is built from: every carryable read of the two shape arms."""
    order = [g for g, _zip in gen.GATES]
    reads = [r for r in rp.reads() if r["arm"] in SHAPES and rp.exists(r["ckpt"])]
    reads.sort(key=lambda r: order.index(r["gate"]))
    return reads


def shapes_doc():
    """A document that queues every *shape* of read, whether or not the repository still owes it.

    The real notebook's queue shrinks every time a run is folded in, and since 5 November it holds
    only fourteen-empty arms: no reads, no weights. A union over three suites, a blind read of an
    eight-empty arm and a paired base read are still promises about this generator, so they are tested
    against a document that queues them - built here by the same `cells_for` that wrote the one on
    disk, over the two depth-8 arms that together cover all three read groups and the four checkpoints
    their reads need.
    """
    if not KEEP_DOC:
        reads = shape_plan()
        assert len(reads) == 10, "%d rows for the two shape arms, expected 10" % len(reads)
        assert {r["gate"] for r in reads} == {"union", "closure", "pairing"}, reads
        nb = bf.render(gen.cells_for(reads, rp.arm_rows(), rp.checkpoints(reads)))
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="nbshapes_"))
        KEEP.append(tmp)
        path = tmp / gen.NAME
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        gen.build_bundle.build_for(path)
        KEEP_DOC.append(path)
    return KEEP_DOC[0]


def docs():
    """The two documents a promise is checked against: the one on disk, and the one with every shape.

    The shipped notebook queues no reads at all now that the paired control closed, so a test that
    only ever looks at it watches an empty queue and passes on nothing. A read, a carried weight or a
    gate that opens over reads is still a promise about this generator, and the shape document is the
    one that holds them.
    """
    return (NB, shapes_doc())


def carryable():
    """The reads a runtime can be handed, because their checkpoint is on this disk.

    Not only the pending ones: a shape the paper rests on stops being owed once it is archived, and a
    test still has to be able to queue it in order to watch the document refuse it.
    """
    return [r for r in rp.reads() if rp.exists(r["ckpt"])]


def owed():
    """What the repository still owes: a planned read that is not archived and whose checkpoint is on
    this disk - which is exactly the list the generator queues."""
    return [r for r in rp.pending() if rp.exists(r["ckpt"])]


def plan_set(rows):
    """What identifies a job: the six fields its command is built from.

    `reads_plan` calls the suite `suite` and a notebook row calls it `which`; the rest is shared, and a
    read that differs in only one of these fields is a different read.
    """
    return {(r["arm"], r["kind"], r.get("which") or r["suite"], r["out"], r["ckpt"],
             r["suite_file"]) for r in rows}


def as_row(r):
    """A notebook's queue row back in the shape `reads_plan.reads()` produces.

    The document's rows carry what a command needs and nothing else; the archived JSON a read's
    certificate is copied from is named only in the plan, so it is looked up here by what identifies
    the read - and a queued read the plan does not know about is an assertion, not a silent miss.
    """
    want = next((p for p in carryable()
                 if (p["arm"], p["kind"], p["suite"]) == (r["arm"], r["kind"], r["which"])), None)
    assert want, "no planned read of %s on %s, so the queue and the plan disagree" % (
        r["arm"], r["which"])
    return dict(r, suite=r["which"], arm_json=want["arm_json"])


def jobs_of(nb=NB, mutate=None, suites=None):
    """The document's own JOBS list, from its queue cell, with no runtime involved.

    Building commands is pure string work, so the tests that read a command - budget, dependency, which
    suite a repair is pointed at - do not pay for a boot. `mutate` edits the cell the way an accident
    would, and the caller is told when the edit lands nowhere; `suites` replaces the table the
    validating cell published, which is how the queue's own cross-check gets exercised.
    """
    src = base._cell(QUEUE_TAG, nb)
    if mutate:
        before = src
        src = mutate(src)
        assert src != before, "the mutant did not change the queue cell, so the check that " \
                              "depends on it proves nothing"
    cut = src.index("JOBS = ")
    head, body = src[:cut], src[cut:src.index("\n_suites")]
    ns = {"SUITES": literal(SUITES_TAG, nb) if suites is None else suites, "THREADS": 1}
    exec(base._runnable(head + "\n" + body), ns)
    return ns["JOBS"]


def patched(anchor, new_src, nb=NB):
    """A copy of the notebook with one cell's source replaced, in a scratch directory.

    Editing the document on disk would leave a notebook with a truncated queue behind if this test
    crashed, and that notebook is what the researcher is about to run.
    """
    doc = json.loads(pathlib.Path(nb).read_text(encoding="utf-8"))
    hit = 0
    for c in doc["cells"]:
        if c["cell_type"] == "code" and anchor in "".join(c["source"]):
            c["source"] = new_src.splitlines(keepends=True)
            hit += 1
    assert hit == 1, "%d cells contain %r, expected 1" % (hit, anchor)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nbpatch_"))
    KEEP.append(tmp)
    out = tmp / NB.name
    out.write_text(json.dumps(doc), encoding="utf-8")
    return out


def trimmed_reads_cell(nb=NB):
    """The queue cell with its last read dropped, as source for `patched`."""
    rows = queued(nb)
    assert len(rows) > 1, "one read is nothing to drop, so the plan test would prove nothing"
    src = base._cell(READS_TAG, nb)
    start = src.index("[", src.index(READS_TAG))
    end = src.index(ARMS_TAG)
    return src[:start] + json.dumps(rows[:-1], indent=4) + src[end:]


class Refused(SystemExit):
    """A cell's SystemExit, plus what the run printed and reached before it stopped.

    The cells name what they rejected on their own line and then exit with a summary, so a test that
    only read the exit would pass on the wrong refusal. The launch records and the boot directory come
    along too: the promise being checked is usually that nothing was launched and nothing was zipped.
    Subclassing SystemExit keeps a caller that only cares that the notebook stopped from having to
    know about this file.
    """

    def __init__(self, message, text, stub=None, tmp=None):
        SystemExit.__init__(self, message)
        self.text = text
        self.stub = stub
        self.tmp = tmp


# ------------------------------------------------------------------- run one document ----
def run_all(plant=None, plant_json=None, plant_raw=None, mutate=None, until=None, nb=NB):
    """Boot the bundle into a fake /content and run every other cell, in document order.

    `plant` copies repository files into the booted runtime, `plant_json` writes fabricated JSON and
    `plant_raw` any other bytes, all before the first cell runs - that is how the results cell's two
    guards and the fourteen-cell zip are read against data. `mutate` rewrites every code cell, for the
    mutants below. `until` stops before the first cell containing it, which is how the ship cell - an
    emergency exit meant to download an unfinished gate - is kept out of a run that tests what happens
    while jobs are in flight. Returns text, launch records, the namespace the cells shared, the booted
    repo and the names of the zips handed to the browser.
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nbreads_"))
    KEEP.append(tmp)
    cwd, path, sub = os.getcwd(), list(sys.path), sys.modules["subprocess"]
    old_env = os.environ.get("COLAB_CONTENT")
    out, stub = io.StringIO(), base._StubSubprocess()
    try:
        os.environ["COLAB_CONTENT"] = str(tmp)
        os.chdir(tmp)
        ns = base._boot(tmp, nb=nb)
        repo = ns["REPO"]
        os.chdir(repo)                    # as bootstrap() does: the cells read relative paths
        for src_rel, dst_rel in (plant or []):
            dst = repo / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / src_rel, dst)
        for rel, obj in (plant_json or []):
            dst = repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        for rel, body in (plant_raw or []):
            dst = repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(body, encoding="utf-8")
        for _i, src in base._code_cells(nb):
            if any(t in src for t in (base.BUNDLE_TAG, base.SETUP_TAG)):
                continue
            if until and until in src:
                break
            src = mutate(src) if mutate else src
            if base.TICK_TAG in src:
                sys.modules["subprocess"] = stub
            if PUMP_TAG in src:
                ns["time"] = f2._StubTime()
            try:
                with contextlib.redirect_stdout(out):
                    exec(base._runnable(src), ns)
            except SystemExit as exc:
                raise Refused(str(exc), out.getvalue(), stub, tmp)
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


def verifier_cert(arm, tier, which, cert, **over):
    """A verifier-shaped JSON from a certificate, with `over` applied to break it.

    A union arm certified eighty positions, so the read of one of its halves carries the half's own
    counts: the same rates over forty roots, and a share of the drawn roots that leaves the two halves
    adding up to the whole. Anything else and the notebook would refuse for a reason this file made up.
    """
    drawn = cert["n_roots_drawn_worst"]
    if which in ("A", "B") and arm in rp.UNION_ARMS:
        drawn = min(40, drawn) if which == "A" else drawn - min(40, drawn)
    v = {"ckpt": "runs/%s/%s" % (arm, "c4_gru_base.pt" if tier == "base" else "c4_gru.pt"),
         "roots": 80 if which == "U" else 40,
         "worst_draw_rate": cert["worst_draw_rate"],
         "best_draw_rate": cert["best_draw_rate"],
         "random_draw_rate_exact": cert["random_draw_rate_exact"],
         "n_roots_drawn_worst": drawn,
         "deterministic_orders": 120, "order_mean": 1.0, "order_min": 1.0, "order_max": 1.0,
         "adaptive_games_per_root": 40, "adaptive_mean": 1.0, "adaptive_min": 1.0}
    v.update(over)
    return v


def verifier_json(arm, tier, which, **over):
    """The same, from the certificate the archived arm actually logged."""
    d = json.loads((REPO / "data" / ("%s.json" % arm)).read_text(encoding="utf-8"))
    cert = d["repair"]["final"] if tier == "repaired" else d["base"]["certificate"]
    return verifier_cert(arm, tier, which, cert, **over)


def planted_for(rows):
    """Every read in `rows` as the JSON its own arm implies, so a queue looks finished."""
    return [(r["out"], verifier_json(r["arm"], "base" if "base" in r["kind"] else "repaired",
                                     r["which"])) for r in rows]


def arm_cert():
    """A fourteen-empty arm that certified every root it was given: the interesting case."""
    return {"worst_draw_rate": 1.0, "best_draw_rate": 1.0, "random_draw_rate_exact": 1.0,
            "n_roots": 40, "n_roots_drawn_worst": 40}


def arm_archive(arm, E, seed, cert):
    """A repair arm's own JSON, as `src/c4_endgame.py` would have written it.

    An arm this notebook trains has no archived copy, so its expectation - the thing the results cell
    compares a read against - exists only once the runtime has produced it. Fabricating that file is
    the only way to test the comparison without spending hours on a fourteen-empty recursion.
    """
    args = rp.arm_budget(E)
    args.update({"rows": 6, "cols": 7, "win": 4, "empties": E, "seed": seed, "threads": 1,
                 "no_controls": True, "roots_file": rp.suite("A", E).replace("/", "\\")})
    return {"args": args, "base": {"certificate": cert},
            "repair": {"final": cert, "log": [0, 1], "aggregate": [0, 1, 2], "elapsed_s": 1234.5,
                       "checkpoint": "runs/%s/c4_gru.pt" % arm,
                       "base_checkpoint": "runs/%s/c4_gru_base.pt" % arm}}


def arm_planted(broken_outs=()):
    """The fourteen-cell outputs, consistent with each other, plus the weights they saved.

    `broken_outs` names read outputs made to disagree with the repair that produced them, which is what
    the results cell is supposed to catch on a depth this document trains itself.
    """
    jsons, raw, certs = [], [], {}
    for row in armed():
        if row["kind"] == "arm-repair":
            certs[row["arm"]] = arm_cert()
            jsons.append((row["out"], arm_archive(row["arm"], row["E"], row["seed"],
                                                  certs[row["arm"]])))
    for row in armed():
        if row["kind"] == "arm-repair":
            continue
        over = {"worst_draw_rate": 0.5, "n_roots_drawn_worst": 7} if row["out"] in broken_outs else {}
        jsons.append((row["out"], verifier_cert(row["arm"], "repaired", row["which"],
                                                certs[row["arm"]], **over)))
    for arm in sorted(certs):
        raw.append(("runs/%s/c4_gru.pt" % arm, "weights"))
        raw.append(("runs/%s/c4_gru_base.pt" % arm, "weights"))
    return jsons, raw


def redirect(rel, to):
    """A mutator that points one suite file at another, in the suite cell only."""

    def mutate(src):
        if SUITES_TAG in src:
            assert '"%s"' % rel in src, "the notebook no longer reads %s" % rel
            return src.replace('"%s"' % rel, '"%s"' % to)
        return src

    return mutate


def corrupt_chunk(src):
    """Flip one base64 character in a checkpoint cell, so the bytes stop hashing to what is claimed."""
    if PUT_TAG not in src:
        return src
    m = re.search(r"'([A-Za-z0-9+/]{60,})'", src)
    assert m, "a weights cell with no base64 to corrupt"
    bad = m.group(1)
    flip = "A" if bad[0] != "A" else "B"
    return src[:m.start(1)] + flip + bad[1:] + src[m.end(1):]


# ---------------------------------------------------------------------- the queued plan ----
def test_the_notebook_queues_exactly_the_reads_the_repository_still_owes():
    """An empty list is the correct answer once the last read is archived, so no `rows` assertion.

    What keeps this from being vacuous is the pair below: the same equality is checked against a
    document that does queue reads, and against a copy of it with one read dropped.
    """
    rows = queued()
    want, got = plan_set(owed()), plan_set(rows)
    assert got == want, "the notebook queues %r, the repository is missing %r" % (
        sorted(got - want), sorted(want - got))
    for r in rows:
        assert not (REPO / r["out"]).exists(), (
            "%s is already archived; the notebook would re-read it and overwrite a published "
            "number" % r["out"])
        assert (REPO / r["ckpt"]).is_file(), "%s is not on this disk" % r["ckpt"]
        assert r["gate"] == rp.group_of(r["kind"]), (
            "%s is a %s read but sits in the %s gate, so its zip would be filed under a section "
            "that did not produce it" % (r["out"], r["kind"], r["gate"]))


def test_the_notebook_queues_the_untrained_arms_exactly_once_each():
    rows = armed()
    assert plan_set(rows) == plan_set(rp.arm_rows()), (
        "the notebook queues %r, the plan says %r"
        % (sorted(plan_set(rows) - plan_set(rp.arm_rows())),
           sorted(plan_set(rp.arm_rows()) - plan_set(rows))))
    for E, seeds in rp.ARMS:
        mine = [r for r in rows if r["E"] == E]
        assert {r["seed"] for r in mine} == set(seeds), (
            "%d empties is planned for %r but the queue carries %r"
            % (E, sorted(seeds), sorted({r["seed"] for r in mine})))
        assert len(mine) == 3 * len(seeds), (E, [r["kind"] for r in mine])
        for s in seeds:
            kinds = sorted(r["kind"] for r in mine if r["seed"] == s)
            assert kinds == ["arm-read-on-A", "arm-read-on-B", "arm-repair"], (E, s, kinds)
        for r in mine:
            assert r["gate"] == "end%d" % E, (E, r["gate"])
            assert not (REPO / r["out"]).exists(), "%s already exists" % r["out"]


def test_nothing_is_queued_twice_and_every_job_has_a_gate():
    rows = queued() + armed()
    outs = [r["out"] for r in rows]
    assert len(outs) == len(set(outs)), [o for o in outs if outs.count(o) > 1]
    gates = {g for g, _zip in gen.GATES}
    assert {r["gate"] for r in rows} <= gates, (
        "a job sits in a gate no zip will ever be built for")
    assert {r["gate"] for r in rows} == set(literal(GATES_TAG)), (
        "a gate is declared that no job belongs to, so its zip could never open")


def test_a_dropped_read_is_noticed():
    """The plan test above has to fail on a notebook that lost a read, not pass quietly.

    So it is run against the document that does queue reads: an equality between two empty lists -
    which is what the shipped notebook holds today - cannot be broken by a deletion, and a test that
    only ever checks it would stop having teeth without anyone noticing.
    """
    doc = shapes_doc()
    rows = queued(doc)
    assert plan_set(rows) == plan_set(shape_plan()), (
        "the shape document no longer queues the rows it was built from")
    nb2 = patched(READS_TAG, trimmed_reads_cell(doc), nb=doc)
    short = literal(READS_TAG, nb2)
    assert len(short) == len(rows) - 1, "the copy kept the dropped read, so nothing was removed"
    assert plan_set(short) != plan_set(shape_plan()), (
        "a notebook with one read removed still matches the plan")


def test_the_reads_come_before_anything_that_trains():
    """Ordering is the mechanism: `tick()` is first come first served, so this is the schedule.

    A repair started while a read is still owed would delay the number the paper is waiting on, and
    the reads' zips fire only when their gate is whole - so the reads being first is also what keeps
    packaging out of training.
    """
    for doc in docs():
        jobs, rows = jobs_of(doc), queued(doc) + armed(doc)
        assert len(jobs) == len(rows), (len(jobs), len(rows))
        gate = {r["out"]: r["gate"] for r in rows}
        order = [gate[j["out"]] for j in jobs]
        read_gates = [r["gate"] for r in queued(doc)]
        assert order[:len(read_gates)] == read_gates, (
            "the first %d jobs are not the reads in gate order: a repair would start while the paper "
            "still waits" % len(read_gates))
        assert set(order[len(read_gates):]) == {r["gate"] for r in armed(doc)}, order
        for j in jobs[:len(read_gates)]:
            assert j["cmd"][1] == "src/c4_endgame_verify.py", j["cmd"]
        for j in jobs[len(read_gates):]:
            assert j["cmd"][1] in ("src/c4_endgame.py", "src/c4_endgame_verify.py"), j["cmd"]
        if doc is NB:
            assert read_gates == [], "the shipped document has reads again; check its prose too"
        else:
            assert read_gates, "the shape document lost the reads it exists to test"


def test_an_arm_that_sampled_its_own_suite_is_still_a_read_the_plan_can_answer():
    """The first 7x6 arms trained before `--roots-file` existed, so their archived `args` name no
    suite at all - and `expected()` used to index that key, which is a crash rather than a refusal
    in the module the notebooks, the audit and the ingestion all import.

    So the function has to say something about those arms instead of raising. At seed 0 the
    in-process sample is the published suite of its depth, so a suite-A read does have the arm's own
    certificate to be compared with, and the suite-B read of the same arm has no archived number and
    says so rather than borrowing one.
    """
    sampled = []
    for p in sorted((REPO / "data").glob("c4_end*_s*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(d, dict) and isinstance(d.get("args"), dict) \
                and "empties" in d["args"] and "roots_file" not in d["args"]:
            sampled.append((p, d))
    assert sampled, ("ningun brazo archivado precede a --roots-file: la rama de reads_plan.expected "
                     "que los resuelve no protege nada y puede borrarse con esta prueba")
    for p, d in sampled:
        for which in ("A", "B"):
            row = {"arm_json": "data/" + p.name, "suite": which, "kind": "base-on-" + which}
            want = rp.expected(row)
            assert want, "%s on %s: ni expectation ni negativa" % (p.name, which)
            assert want["suite"] == ("A" if d["args"]["seed"] == 0 else ""), (p.name, which)
            assert want["comparable"] == (which == "A" and d["args"]["seed"] == 0), (p.name, which)


# ------------------------------------------------------------------------- the weights ----
def test_one_cell_per_checkpoint_and_none_of_them_is_a_megabyte_block():
    for doc in docs():
        weights = literal(WEIGHTS_TAG, doc)
        chunks = [s for _i, s in base._code_cells(doc) if PUT_TAG in s]
        assert len(chunks) == len(weights), "%d checkpoints listed, %d cells of bytes" % (
            len(weights), len(chunks))
        assert sum(1 for _i, s in base._code_cells(doc) if "\ndef _put(" in s) == 1, (
            "the writer is not one definition the chunks call")
        for rel in weights:
            assert sum(1 for s in chunks if "_put('%s'" % rel in s) == 1, (
                "%s does not arrive in exactly one cell" % rel)
        if not chunks:
            assert not weights, "a document with weights wrote no cell of bytes for them"
            continue
        biggest = max(len(s) for s in chunks)
        assert biggest < 512 * 1024, (
            "a %d KB cell is what makes the browser editor unusable; split it" % (biggest // 1024))


def test_a_carried_weight_hashes_to_the_file_on_this_disk():
    for doc in docs():
        weights = literal(WEIGHTS_TAG, doc)
        assert set(weights) == {r["ckpt"] for r in queued(doc)}, (
            "the reads load %r but the notebook carries %r"
            % (sorted({r["ckpt"] for r in queued(doc)}), sorted(weights)))
        for rel, meta in weights.items():
            on_disk = (REPO / rel).read_bytes()
            assert hashlib.sha256(on_disk).hexdigest() == meta["sha256"], (
                "%s is not the file this notebook says it carries" % rel)
            assert len(on_disk) == meta["bytes"], "%s changed size" % rel
            assert meta["tier"] == ("base" if rel.endswith("c4_gru_base.pt") else "repaired")
            # the arm's own JSON names the path it saved its weights to: the only record in this
            # repository that ties a .pt on disk to a run the paper cites
            rec = json.loads((REPO / meta["arm_json"]).read_text(encoding="utf-8"))["repair"]
            named = rec["base_checkpoint" if meta["tier"] == "base" else "checkpoint"]
            assert _posix(named) == rel, "%s: arm %s records its %s weights at %s, not %s" % (
                rel, meta["arm"], meta["tier"], _posix(named), rel)


# The five refusals below are promises of the weights machinery, and the shipped document no longer
# exercises any of them: it carries no bytes, because every read that needed one is archived. So they
# run against the shape document, where a checkpoint has to arrive, hash and be asked for.
def test_a_weight_that_does_not_hash_is_refused():
    doc = shapes_doc()
    first = base._cell(PUT_TAG, doc)
    rel = re.search(r"\n_put\('([^']+)'", first).group(1)
    mutated = corrupt_chunk(first)
    assert mutated != first, "flipping a character changed nothing"
    try:
        run_all(nb=doc, mutate=lambda s: mutated if s == first else s)
        raise AssertionError("the notebook accepted weights that do not hash to what it claims")
    except Refused as exc:
        assert "no hash a lo que el notebook anuncia" in str(exc), str(exc)
        assert "%s:" % rel in str(exc), str(exc)
        assert "no es el checkpoint del brazo %s" % literal(WEIGHTS_TAG, doc)[rel]["arm"] in str(exc), \
            str(exc)
        assert not exc.stub.launches, "reads launched before the refusal: %r" % exc.stub.launches


def test_a_weight_of_the_wrong_size_is_refused():
    doc = shapes_doc()
    src = base._cell(WEIGHTS_TAG, doc)
    m = re.search(r'"bytes": (\d+)', src)
    assert m, "the weights table declares no size to corrupt"
    mutated = src[:m.start(1)] + str(int(m.group(1)) + 1) + src[m.end(1):]
    try:
        run_all(nb=doc, mutate=lambda s: mutated if s == src else s)
        raise AssertionError("the notebook wrote a checkpoint whose declared size did not match")
    except Refused as exc:
        assert "bytes esperados" in str(exc), str(exc)


def test_a_checkpoint_the_queue_never_asks_for_is_not_written():
    doc = shapes_doc()
    src = base._cell(PUT_TAG, doc)
    m = re.search(r"_put\('([^']+)'", src)
    assert m, "a weights cell that names no checkpoint"
    mutated = src[:m.start(1)] + "runs/c4_end99_s7/c4_gru.pt" + src[m.end(1):]
    try:
        run_all(nb=doc, mutate=lambda s: mutated if s == src else s)
        raise AssertionError("the notebook wrote a checkpoint no read in it is pointed at")
    except Refused as exc:
        assert "no cola ninguna lectura de ese checkpoint" in str(exc), str(exc)


def test_a_document_without_its_weights_stops_before_anything_launches():
    """Skipping the cells of bytes is the accident a reviewer will make, and it must not hang."""
    try:
        run_all(nb=shapes_doc(), mutate=lambda s: "pass" if PUT_TAG in s else s)
        raise AssertionError("a document with no weights at all ran its queue")
    except Refused as exc:
        assert "faltan las celdas de bytes" in str(exc), str(exc)
        assert not exc.stub.launches, "jobs launched with no checkpoint on disk: %r" % (
            exc.stub.launches)
        assert not list(exc.tmp.glob("c4_*_results.zip")), "a zip of nothing reached the browser"


def test_a_document_that_carries_nothing_does_not_wait_for_bytes():
    """The other half of that refusal: with no checkpoint listed, the queue has to run.

    The shipped document is exactly this case, and a weights cell that stopped a training-only queue
    would turn the closing of a read into a notebook that launches nothing at all.
    """
    assert not literal(WEIGHTS_TAG, NB), "the shipped document carries weights again"
    _text, stub, _ns, _repo, _zips = run_all(nb=NB, until=SHIP_TAG)
    assert stub.launches, "a queue with no bytes to lay down launched nothing: %s" % _text[-600:]


def test_one_skipped_weight_cell_is_named_by_the_refusal():
    doc = shapes_doc()
    src = base._cell(PUT_TAG, doc)
    missing = re.search(r"_put\('([^']+)'", src).group(1)
    try:
        run_all(nb=doc, mutate=lambda s: "pass" if s == src else s)
        raise AssertionError("the document accepted a checkpoint that never arrived")
    except Refused as exc:
        assert missing in str(exc), "%s was skipped and the refusal did not name it: %s" % (
            missing, str(exc))


def test_every_read_loads_a_checkpoint_this_document_carried():
    """A read of somebody else's weights is the failure this document must not be able to make.

    Checked on both documents: the shape one reads weights it carries, the shipped one reads only the
    weights its own repairs write, and in both cases the `--ckpt` of a launched read has to be one of
    the files this very document put on disk.
    """
    for doc in docs():
        _text, stub, _ns, _repo, _zips = run_all(nb=doc, until=SHIP_TAG)
        assert stub.launches, "a bare runtime launched nothing"
        weights = set(literal(WEIGHTS_TAG, doc))
        produced = {r["ckpt"] for r in armed(doc)}
        read_launches = [c for c, _kw in stub.launches if c[1].endswith("c4_endgame_verify.py")]
        if doc is NB:
            assert not read_launches, (
                "the shipped document launched a read of weights nothing trained yet: %r"
                % read_launches)
        else:
            assert read_launches, "the first wave launched no read at all"
        for cmd in read_launches:
            assert _posix(cmd[cmd.index("--ckpt") + 1]) in weights | produced, cmd


# --------------------------------------------------------------------------- the suites ----
def test_the_suites_the_queue_points_at_are_committed_and_in_the_bundle():
    import build_bundle
    seen_union = False
    for doc in (NB, shapes_doc()):
        suites = literal(SUITES_TAG, doc)
        for r in literal(READS_TAG, doc) + literal(ARMS_TAG, doc):
            assert r["suite_file"] in suites.values(), (
                "%s reads %s, which the suite cell does not know" % (r["out"], r["suite_file"]))
            assert (REPO / r["suite_file"]).exists(), "%s is not committed" % r["suite_file"]
            assert r["suite_file"] in build_bundle.BUNDLE_FILES, (
                "%s is read by this notebook but does not travel in the bundle, so a bare runtime "
                "would stop or sample its own positions" % r["suite_file"])
        depths = set(literal(SUITES_E_TAG, doc).values())
        assert 14 in depths, (
            "the fourteen-cell depth is not validated at all, so the arms would read it unopened")
        for key, path in suites.items():
            assert path == rp.suite(key[0], int(key[1:])), (key, path)
            if key.startswith("U"):
                seen_union = True
    assert seen_union, "no document exercised the union key of the loop above"


def test_the_suite_cell_opens_every_file_and_validates_the_fourteen_cell_depth_too():
    texts = {}
    for doc in (NB, shapes_doc()):
        text = texts[doc] = run_all(nb=doc, until=QUEUE_TAG)[0]
        for E in sorted(set(literal(SUITES_E_TAG, doc).values())):
            assert "A%d y B%d comparten 0 posiciones" % (E, E) in text, (E, text[:2500])
        assert "las suites están validadas" in text, text[:2500]
    assert "U8 es exactamente A seguida de B: 40 + 40 = 80" in texts[shapes_doc()], \
        texts[shapes_doc()][:2500]


def test_a_blind_suite_that_shares_boards_stops_the_run():
    try:
        run_all(nb=shapes_doc(), until=QUEUE_TAG, mutate=redirect(rp.suite("B", 8), rp.suite("A", 8)))
        raise AssertionError("a blind suite that is the published one passed the holdout check")
    except AssertionError as exc:
        assert "not a holdout" in str(exc), str(exc)


def test_the_fourteen_cell_blind_suite_is_a_holdout_too():
    """The arms trained here are certified against A and read blind on B: if B overlapped A, the
    fourteen-empty row of `tab:blind` would measure nothing at all."""
    try:
        run_all(until=QUEUE_TAG, mutate=redirect(rp.suite("B", 14), rp.suite("A", 14)))
        raise AssertionError("a fourteen-empty suite B identical to suite A passed the holdout check")
    except AssertionError as exc:
        assert "is not a holdout" in str(exc) and "14" in str(exc), str(exc)


def test_a_union_suite_that_is_not_a_followed_by_b_stops_the_run():
    a = json.loads((REPO / rp.suite("A", 8)).read_text(encoding="utf-8"))
    try:
        run_all(nb=shapes_doc(), until=QUEUE_TAG, plant_json=[("data/not_a_union.json", a + a)],
                mutate=redirect(rp.suite("U", 8), "data/not_a_union.json"))
        raise AssertionError("a union that is suite A twice was accepted")
    except AssertionError as exc:
        assert "not suite A followed by suite B" in str(exc), str(exc)


def test_a_suite_of_the_wrong_depth_stops_the_run():
    eight = json.loads((REPO / rp.suite("A", 8)).read_text(encoding="utf-8"))
    ten = json.loads((REPO / rp.suite("A", 10)).read_text(encoding="utf-8"))
    try:
        run_all(nb=shapes_doc(), until=QUEUE_TAG, plant_json=[("data/mixed_suite.json", eight[:39] + [ten[0]])],
                mutate=redirect(rp.suite("A", 8), "data/mixed_suite.json"))
        raise AssertionError("a suite that is not all one depth was read")
    except AssertionError as exc:
        assert "not a 8-empty endgame" in str(exc), str(exc)


def test_a_value_one_root_is_not_admitted_as_a_draw_suite():
    eight = json.loads((REPO / rp.suite("A", 8)).read_text(encoding="utf-8"))
    flipped = eight[:39] + [dict(eight[39], value=1)]
    try:
        run_all(nb=shapes_doc(), until=QUEUE_TAG, plant_json=[("data/winner_suite.json", flipped)],
                mutate=redirect(rp.suite("A", 8), "data/winner_suite.json"))
        raise AssertionError("a suite with a root the learner cannot draw was read")
    except AssertionError as exc:
        assert "cannot draw" in str(exc), str(exc)


def test_a_finished_position_is_not_admitted_as_a_root():
    eight = json.loads((REPO / rp.suite("A", 8)).read_text(encoding="utf-8"))
    board = ([1, 1, 1, 1, 2, 2, 2] + [2, 2, 2, 1, 1, 1, 2] + [1, 1, 2, 2, 1, 2, 1]
             + [2, 1, 2, 1, 2, 1, 2] + [1, 2, 1, 2, 1, 2, 0] + [0] * 7)
    assert len(board) == 42 and board.count(0) == 8, (
        "the fabricated board is not an eight-empty position either way")
    done = json.loads(json.dumps(eight))
    done[0]["board"] = board
    try:
        run_all(nb=shapes_doc(), until=QUEUE_TAG, plant_json=[("data/finished_suite.json", done)],
                mutate=redirect(rp.suite("A", 8), "data/finished_suite.json"))
        raise AssertionError("a position that is already a win was read as a draw root")
    except AssertionError as exc:
        assert "finished position" in str(exc), str(exc)


def test_a_missing_suite_stops_the_run():
    try:
        run_all(nb=shapes_doc(), until=QUEUE_TAG, mutate=redirect(rp.suite("A", 8), "data/never_committed.json"))
        raise AssertionError("a suite absent from the runtime was replaced by sampling")
    except Refused as exc:
        assert "is not in the runtime" in str(exc), str(exc)


def test_a_queue_pointing_at_a_suite_the_validation_did_not_keep_is_refused():
    """The suite cell and the queue could disagree: then a read would run over positions nobody
    checked, and its number would look like a certificate."""
    validated = literal(SUITES_TAG)
    suites = dict(validated, A14=rp.suite("A", 10))
    assert suites["A14"] != validated["A14"], "the document does not name A14 as expected"
    try:
        jobs_of(suites=suites)
    except AssertionError as exc:
        assert "la celda de suites valido" in str(exc), str(exc)
    else:
        raise AssertionError("the queue built a command over a suite the validating cell replaced")


# ------------------------------------------------------------------------ what it runs ----
def test_a_read_verifies_and_only_the_new_arms_train():
    jobs = jobs_of()
    trained = [j for j in jobs if j["cmd"][1] == "src/c4_endgame.py"]
    assert len(trained) == len([r for r in armed() if r["kind"] == "arm-repair"]), (
        "%d training commands for %d planned repairs"
        % (len(trained), len([r for r in armed() if r["kind"] == "arm-repair"])))
    for j in jobs:
        script, flags = j["cmd"][1], [a for a in j["cmd"] if a.startswith("--")]
        source = VERIFY_SCRIPT if script.endswith("verify.py") else TRAIN_SCRIPT
        unknown = sorted(f for f in flags if ("'%s'" % f) not in source and ('"%s"' % f) not in source)
        assert not unknown, "%s: %s has no %s flag" % (script, j["name"], unknown)
        if script.endswith("verify.py"):
            assert {"--ckpt", "--roots", "--out", "--orders", "--games"} <= set(flags), j["name"]
            assert j["cmd"][j["cmd"].index("--orders") + 1] == "120", j["name"]
            assert j["cmd"][j["cmd"].index("--games") + 1] == "40", j["name"]
            assert "--no-controls" not in flags, "a verifier call is not a training run"
        else:
            assert {"--empties", "--roots-file", "--seed", "--ckpt-out", "--out"} <= set(flags)
            assert "--no-controls" in flags, (
                "a fourteen-empty arm with controls would spend the session on a question the "
                "published ladder already answered")


def test_a_read_waits_for_its_suite_as_well_as_its_weights():
    for j in jobs_of():
        if j["cmd"][1] == "src/c4_endgame.py":
            continue
        suite = [p for p in j["needs"] if "roots_" in _posix(p)]
        assert len(suite) == 1, (j["name"], j["needs"])
        assert (REPO / suite[0]).exists(), suite[0]
        assert _posix(suite[0]) == j["cmd"][j["cmd"].index("--roots") + 1], j["name"]


def test_an_arm_read_waits_for_the_repair_json_and_the_checkpoint_it_saves():
    """A read of weights still being written, or of an arm whose certificate does not exist yet, is
    not a read: the dependency is the whole claim that the number belongs to that run."""
    by_out = {j["out"]: j for j in jobs_of()}
    for r in armed():
        j = by_out[r["out"]]
        if r["kind"] == "arm-repair":
            assert j["needs"] == [rp.suite("A", r["E"])], (
                "a repair is handed %r; it must be handed the published suite only" % j["needs"])
            assert r["E"] == 14, r["E"]
            continue
        want = {rp.suite(r["which"], r["E"]), r["arm_json"], r["ckpt"]}
        assert set(j["needs"]) == want, (j["name"], sorted(j["needs"]), sorted(want))
        assert j["cmd"][j["cmd"].index("--ckpt") + 1] == r["ckpt"], j["name"]


def bundled(nb=NB):
    """Every path the document's own bundle unpacks, so a `needs` entry can be traced to a producer.

    The bundle cell is generated after the queue cells and holds a base64 zip with a sha256 beside
    each name; the names are declared in `BUNDLE_FILES` before the archive itself, which is what this
    reads.
    """
    src = base._bundle_source(nb)
    ns = {}
    exec(src[: src.index("_archive =")], ns)
    return {_posix(rel) for rel in ns["BUNDLE_FILES"]}


def test_the_repair_writes_the_files_its_own_reads_wait_for():
    """`needs` is only a schedule if something produces the file; otherwise the queue waits forever.

    Three things do: a job's own output, the weights a repair saves and the checkpoints the document
    carries, and - for the suites - the bundle cell that unpacks them before the queue is built.
    """
    jobs, written = jobs_of(), set()
    for j in jobs:
        written.add(_posix(j["out"]))
        if "--ckpt-out" in j["cmd"]:
            d = j["cmd"][j["cmd"].index("--ckpt-out") + 1]
            written |= {_posix(d + "/c4_gru.pt"), _posix(d + "/c4_gru_base.pt")}
    written |= set(literal(WEIGHTS_TAG)) | bundled()
    for j in jobs:
        for need in j["needs"]:
            assert _posix(need) in written, (
                "%s waits for %s, which nothing in this notebook writes" % (j["name"], need))
    suites = set(literal(SUITES_TAG).values())
    assert suites <= bundled(), sorted(suites - bundled())


def test_the_blind_suite_is_never_given_to_a_repair():
    rows = {r["out"]: r for r in queued() + armed()}
    for j in jobs_of():
        r = rows[j["out"]]
        flag = "--roots-file" if j["cmd"][1] == "src/c4_endgame.py" else "--roots"
        read = _posix(j["cmd"][j["cmd"].index(flag) + 1])
        assert read == rp.suite(r["which"], r["E"]), (j["name"], read, r["which"], r["E"])
        if flag == "--roots-file":
            assert r["which"] == "A", (
                "%s would be repaired against the blind suite it is supposed to be tested on"
                % j["name"])
    assert any(rp.suite("B", r["E"]) == _posix(j["cmd"][j["cmd"].index("--roots") + 1])
               for j in jobs_of() if j["cmd"][1].endswith("verify.py")), (
        "this queue never reads a blind suite, so `tab:blind` is not what it is closing")


def test_the_fourteen_empty_budget_is_the_twelve_empty_published_one():
    """Typed budgets rot, so this is the archived `args` of the donor depth, flag by flag."""
    budget = literal(BUDGET_TAG)
    assert budget == rp.arm_budget(14), (budget, rp.arm_budget(14))
    donor = rp.ARM_BUDGET_FROM[14]
    for arm in rp.LADDER_ARMS[donor]:
        args = json.loads((REPO / rp.arm_json(arm)).read_text(encoding="utf-8"))["args"]
        assert args["empties"] == donor, (arm, args["empties"])
        for k, v in sorted(budget.items()):
            assert args[k] == v, (
                "a fourteen-empty arm would run %s=%r where depth %d was published with %r"
                % (k, v, donor, args[k]))
    assert (budget["stop"], budget["agg_repeat"]) == ("both", 60), budget


def test_a_repair_command_carries_the_budget_it_claims():
    budget = literal(BUDGET_TAG)
    pairs = (("--epochs", "epochs"), ("--max-iters", "max_iters"), ("--agg-repeat", "agg_repeat"),
             ("--stop", "stop"), ("--hidden", "hidden"), ("--emb", "emb"), ("--branch", "branch"),
             ("--roots", "roots"), ("--root-value", "root_value"), ("--lr", "lr"))
    repairs = [j for j in jobs_of() if j["cmd"][1] == "src/c4_endgame.py"]
    assert repairs, "the queue holds no repair, so the budget is never used"
    for j in repairs:
        for flag, key in pairs:
            got = j["cmd"][j["cmd"].index(flag) + 1]
            assert got == str(budget[key]), (j["name"], flag, got, budget[key])
        assert j["cmd"][j["cmd"].index("--empties") + 1] == "14", j["name"]
        assert _posix(j["cmd"][j["cmd"].index("--roots-file") + 1]) == rp.suite("A", 14), j["name"]


def test_a_mutated_budget_is_read_off_the_command_and_not_only_off_the_table():
    """The check above compares the command to the budget, so a command that stops carrying it has
    to disagree - which is what proves this file is not matching a literal against itself."""
    def mutate(src):
        return src.replace("'--agg-repeat', str(BUDGET['agg_repeat'])", "'--agg-repeat', '1'")

    repairs = [j for j in jobs_of(mutate=mutate) if j["cmd"][1] == "src/c4_endgame.py"]
    got = {j["cmd"][j["cmd"].index("--agg-repeat") + 1] for j in repairs}
    assert got == {"1"}, got
    assert got != {str(literal(BUDGET_TAG)["agg_repeat"])}, (
        "writing a number over the budget changed nothing, so no test reads the command")


# ----------------------------------------------------------------------- the result cell ----
def test_the_expectation_is_the_arms_own_certificate():
    for doc in (NB, shapes_doc()):
        exp = literal(EXPECTED_TAG, doc)
        rows = literal(READS_TAG, doc)
        checked = 0
        for r in rows:
            want = rp.expected(as_row(r))
            if not want:
                continue
            tier = "repaired" if "repaired" in r["kind"] else "base"
            assert tier in exp.get(r["arm"], {}), "no %s expectation for %s" % (tier, r["arm"])
            got = exp[r["arm"]][tier]
            assert got["suite"] == want["suite"], (r["arm"], got["suite"], want["suite"])
            for k, v in got["cert"].items():
                assert abs(v - want["cert"][k]) < 1e-12, (r["arm"], tier, k, v)
            checked += 1
        assert checked == len(rows), (
            "%d of the %d reads in this document carry the certificate their arm logged, so a read "
            "without one was queued and shipped without ever being compared" % (checked, len(rows)))
        for r in armed():
            assert r["arm"] not in exp, (
                "%s has not run yet, so an expectation copied at build time would be an invented "
                "number" % r["arm"])


def test_a_read_is_checked_against_the_certificate_the_arm_logged():
    doc = shapes_doc()
    row = next((r for r in queued(doc) if r["kind"].startswith("repaired-on-")
                and rp.expected(as_row(r))["comparable"]), None)
    assert row, "no queued read is on the suite its arm certified over, so the comparison never runs"
    good = verifier_json(row["arm"], "repaired", row["which"])
    text = run_all(nb=doc, plant_json=[(row["out"], good)])[0]
    assert "COINCIDE" in text, text[-900:]
    bad = verifier_json(row["arm"], "repaired", row["which"], worst_draw_rate=0.0,
                        n_roots_drawn_worst=1)
    try:
        run_all(nb=doc, plant_json=[(row["out"], bad)])
        raise AssertionError("a read that does not reproduce its own arm was printed")
    except Refused as exc:
        assert "no reproduce" in str(exc), str(exc)
        assert "no son los del brazo %s" % row["arm"] in exc.text, exc.text[-1200:]


def test_the_union_identity_is_enforced_and_a_broken_union_is_refused():
    """The claim `sec:union` rests on: a certificate over eighty positions is the minimum of its two
    disjoint halves - so a document that queues all three reads must refuse when they stop adding up."""
    doc = shapes_doc()
    rows = {}
    for arm in rp.UNION_ARMS:
        got = {r["which"]: r for r in queued(doc)
               if r["arm"] == arm and r["kind"].startswith("union-repaired-")}
        if set(got) == {"A", "B", "U"}:
            rows = got
            break
    assert rows, "no union arm queues all three suites, so the identity is never exercised"
    plant = planted_for(rows.values())
    text = run_all(nb=doc, plant_json=plant)[0]
    assert "identidad W_U = min(W_A, W_B): OK" in text, text[-1400:]
    assert "COINCIDE" in text, text[-1400:]
    whole = verifier_json(rows["U"]["arm"], "repaired", "U")["n_roots_drawn_worst"]
    try:
        run_all(nb=doc, plant_json=plant + [(rows["U"]["out"],
                                     verifier_json(rows["U"]["arm"], "repaired", "U",
                                                   n_roots_drawn_worst=whole - 1))])
        raise AssertionError("a union whose drawn roots did not add up was accepted")
    except Refused as exc:
        assert "no es el minimo de sus dos mitades" in exc.text, exc.text[-1200:]


def test_a_new_arm_is_checked_against_the_repair_json_the_runtime_wrote():
    """An arm this document trains has no build-time copy of its own certificate, so the comparison
    only exists at runtime - and it is the one thing that tells a read of the right weights from a
    read of somebody else's."""
    jsons, raw = arm_planted()
    text = run_all(plant_json=jsons, plant_raw=raw)[0]
    assert "repair: W" in text, text[-1500:]
    compared = [r for r in armed() if r["kind"] == "arm-read-on-A"]
    for r in compared:
        assert re.search(r"%s\s+arm-read-on-A\s+on A .*COINCIDE" % r["arm"], text), text[-3000:]
    assert text.count("COINCIDE") == len(compared), (
        "the new arms' reads were printed without being compared to their own repair, or something "
        "else was compared: " + text[-1500:])
    for r in [x for x in armed() if x["kind"] == "arm-read-on-B"]:
        line = next(l for l in text.splitlines() if r["arm"] in l and "arm-read-on-B" in l)
        assert "el loop decia" not in line, (
            "the blind read was checked against a certificate the arm never made on suite B - the "
            "number the paper goes on to measure, not one it already owns")

    broken = [r for r in armed() if r["kind"] == "arm-read-on-A"][:1]
    jsons, raw = arm_planted(broken_outs=[r["out"] for r in broken])
    try:
        run_all(plant_json=jsons, plant_raw=raw)
        raise AssertionError("a fourteen-empty read that disagreed with its own arm was accepted")
    except Refused as exc:
        assert "no son los del brazo %s" % broken[0]["arm"] in exc.text, exc.text[-1500:]
        assert "arm-read-on-A" in exc.text, exc.text[-1500:]


def test_the_blind_tally_keeps_the_three_populations_apart():
    """Folding the union arms and the new depth into one number over `tab:blind` would be a claim the
    paper does not make, so the three populations are counted on their own lines."""
    doc = shapes_doc()
    jsons, raw = arm_planted()
    text = run_all(nb=doc, plant_json=planted_for(queued(doc)) + jsons, plant_raw=raw)[0]
    for label in ("tab:blind: reparados contra la suite A",
                  "sec:union: reparados contra la union A+B",
                  "nuevos a catorce vacias: reparados contra A"):
        assert re.search(re.escape(label) + r": \d+ de \d+ llegan a W=1 ciego \[\d\.\d+, \d\.\d+\]",
                         text), (label, text[-2000:])
    src = base._cell(EXPECTED_TAG, doc)
    needle = "ALL = READS + [a for a in ARMS if a['kind'] != 'arm-repair']"
    assert needle in src, "the results cell no longer folds the arm reads into its own list"
    nb2 = patched(EXPECTED_TAG, src.replace(needle, "ALL = READS"), nb=doc)
    thinner = run_all(nb=nb2, plant_json=planted_for(queued(doc)) + jsons, plant_raw=raw)[0]
    assert "nuevos a catorce vacias" not in thinner, (
        "the third population is printed even with the arm rows dropped from the tally, so the "
        "test above does not read them")
    assert "tab:blind" in thinner, thinner[-1500:]


# -------------------------------------------------------------------------- the download ----
def test_the_gates_are_the_read_groups_and_the_new_depth():
    """A gate exists exactly for the group of work the document queues, in the order it empties.

    The real document loses a gate every time a run is folded in, so the four-gate shape - three read
    groups plus the depth that has to be trained - is checked on the document that queues all of them.
    """
    for doc in (NB, shapes_doc()):
        gates = literal(GATES_TAG, doc)
        rows = literal(READS_TAG, doc) + literal(ARMS_TAG, doc)
        owed = [g for g, _zip in gen.GATES if any(r["gate"] == g for r in rows)]
        assert list(gates) == owed, (list(gates), owed)
        for gate, g in gates.items():
            want = sorted(r["out"] for r in rows if r["gate"] == gate)
            assert g["outs"] == want, (gate, sorted(set(g["outs"]) ^ set(want)))
            assert want, gate
        names = [zip_name(g["zip"]) for g in gates.values()]
        assert len(set(names)) == len(names), names
        for g in gates.values():
            assert g["zip"] in base._cell(GATES_TAG, doc), g["zip"]
        read_gates = [g for k, g in gates.items() if not k.startswith("end")]
        assert all(g["extra"] == [] and g["globs"] == [] for g in read_gates), (
            "a read gate collects files it was never owed: a read zip must carry results, not weights")
        end = [g for k, g in gates.items() if k.startswith("end")][0]
        assert end["globs"] == ["runs/c4_end14_s*/c4_gru*.pt"], end["globs"]
        assert set(end["extra"]) == {rp.suite("A", 14), rp.suite("B", 14)}, end["extra"]
    assert len(literal(GATES_TAG, shapes_doc())) == 4, (
        "the shape document no longer queues all four groups, so a gate could disappear from the "
        "generator and nothing here would notice")


def test_nothing_is_downloaded_while_a_gate_is_unfinished():
    """The gate that is short one job is exactly the one that stays shut; the whole ones may open.

    `tick()` calls `snapshot()` after every launch, so a section whose last read landed is handed to
    the browser on the spot - which is what keeps the three read zips out of the training run. What
    may not happen is a partial section shipping, here on both sides of it: a read gate missing its
    first union row, and the depth gate missing one of its six reads while both weights are on disk.
    """
    doc = shapes_doc()
    _text, _stub, _ns, _repo, zips = run_all(nb=doc, until=SHIP_TAG)
    assert not zips, "a zip reached the browser with jobs still running: %r" % zips

    rows = queued(doc)
    _t, _s, _n, _r, zips = run_all(nb=doc, until=SHIP_TAG, plant_json=planted_for(rows[1:]))
    assert zip_name("reads_union") not in zips, (
        "one union read short and the section shipped anyway: %r" % zips)
    assert sorted(zips) == sorted(zip_name(g) for g in ("reads_closure", "reads_pairing")), zips

    jsons, raw = arm_planted()
    _t, _s, _n, _r, zips = run_all(nb=doc, until=SHIP_TAG,
                                   plant_json=planted_for(rows) + jsons[:-1], plant_raw=raw)
    assert not any("end14" in z for z in zips), (
        "the fourteen-cell gate ships with a read still in flight, and the arm's weights leave the "
        "runtime before the arm is measured: %r" % zips)
    assert len(zips) == 3, zips


def test_the_read_gates_open_exactly_when_the_training_starts():
    """Every read on disk and nothing from the new depth: the three read zips fire, and the only jobs
    left to launch are the repairs - which is the promise that nothing packages during training."""
    doc = shapes_doc()
    text, stub, _ns, repo, zips = run_all(nb=doc, until=SHIP_TAG,
                                          plant_json=planted_for(queued(doc)))
    assert sorted(zips) == sorted(zip_name(g) for g in
                                  ("reads_union", "reads_closure", "reads_pairing")), zips
    assert not any("end14" in z for z in zips), zips
    started = {cmd[1] for cmd, _kw in stub.launches}
    assert started == {"src/c4_endgame.py"}, sorted(started)
    with zipfile.ZipFile(repo.parent / zips[0]) as z:
        names = set(z.namelist())
    assert any(n.startswith("data/") for n in names), sorted(names)
    assert not any("c4_gru" in n for n in names), (
        "a read zip shipped a checkpoint back: %r" % [n for n in names if "c4_gru" in n])
    assert "faltan %d jobs" % len([a for a in armed() if a["kind"] != "arm-repair"]) in text, \
        text[-1200:]


def test_the_fourteen_gate_zip_carries_the_only_copy_of_the_new_weights():
    """`*.gitignore` keeps `.pt` out of the repository, so that zip is where the fourteen-cell weights
    will exist or nowhere: the arms, their logs, both fourteen suites and the weights themselves."""
    jsons, raw = arm_planted()
    _text, _stub, _ns, repo, zips = run_all(until=SHIP_TAG, plant_json=jsons, plant_raw=raw)
    assert len(zips) == 1 and "end14" in zips[0], zips
    with zipfile.ZipFile(repo.parent / zips[0]) as z:
        names = set(z.namelist())
    assert {rp.suite("A", 14), rp.suite("B", 14)} <= names, sorted(names)
    pts = sorted(n for n in names if n.endswith(".pt"))
    assert len(pts) == 2 * len({r["arm"] for r in armed()}), pts
    assert all(n.startswith(("data/", "runs/")) for n in names), sorted(names)
    assert sorted(r["out"] for r in armed()) <= sorted(names), (
        "the gate opened before all nine of its outputs were in it")


def test_every_gate_lands_in_the_browser_when_the_queue_empties():
    doc = shapes_doc()
    jsons, raw = arm_planted()
    text, _stub, _ns, _repo, zips = run_all(nb=doc, plant_json=planted_for(queued(doc)) + jsons,
                                            plant_raw=raw)
    assert len(zips) == 4, zips
    assert len(zips) == len(literal(GATES_TAG, doc)), zips
    assert "queue empty - every arm is on disk and in the browser" in text, text[-800:]
    assert "descarga final" in text, text[-1200:]


def test_the_ship_cell_downloads_a_gate_that_is_only_half_finished():
    """A session that is ending is the one case where a partial gate beats a lost run.

    `tick()` and the results cell only ever call `snapshot()`, which refuses an unfinished section;
    the document's last cell calls `ship()`. The same disk state - four of the six union reads a
    section holds - has to produce no download through the automatic path and one through the
    emergency exit, which is why both halves run here rather than only the second.
    """
    doc = shapes_doc()
    rows = [r for r in queued(doc) if r["gate"] == "union"][:4]
    assert len(rows) < len([r for r in queued(doc) if r["gate"] == "union"]), (
        "the union section has no unfinished state to plant")
    planted = planted_for(rows)
    owed = len(queued(doc)) + len([a for a in armed(doc) if a["kind"] != "arm-repair"])
    text, _stub, _ns, _repo, zips = run_all(nb=doc, until=SHIP_TAG, plant_json=planted)
    assert not zips, "the automatic path handed down an unfinished section: %r" % zips
    assert "faltan %d jobs" % (owed - len(rows)) in text, text[-900:]

    _text, _stub, _ns, repo, zips = run_all(nb=doc, plant_json=planted)
    assert zip_name("reads_union") in zips, (
        "the ship cell was the only way these four reads leave the runtime: %r" % zips)
    with zipfile.ZipFile(repo.parent / zip_name("reads_union")) as z:
        names = sorted(n for n in z.namelist() if n.startswith("data/"))
    assert names == sorted(r["out"] for r in rows), names
    assert "zip al navegador" in _text or "zips al navegador" in _text, _text[-800:]


# ------------------------------------------------------------------ prose and provenance ----
RESOURCE_TAG = "def quota_cores()"


def markdown(nb=NB):
    """Every markdown cell of a document, line-wraps folded into single spaces."""
    doc = json.loads(nb.read_text(encoding="utf-8"))
    return " ".join(" ".join("".join(c["source"]).split()) for c in doc["cells"]
                   if c["cell_type"] == "markdown")


# Each number the prose quotes, with the sentence the generator writes in its place when that number
# is zero. The zero shape is matched by its own words, not by the absence of a digit: a reworded
# sentence that quotes nothing must fail here rather than be read as "no reads, of course".
ARM_ITEMS = r"The (?:only|other) (\d+) items are the arms at fourteen empty cells"
MINUTES_CLAIM = r"returns by itself after (\d+) minutes"
READ_FORMS = ((r"(\d+) of the list's items (?:are reads|is a read)", True),
              (r"Nothing on it re-reads weights that already exist", False))
CKPT_FORMS = ((r"(\d+) trained checkpoints ride inside", True),
              (r"No checkpoint has to travel inside", False))
QUEUE_WITH_READS = r"(\d+) reads, then (\d+) repairs each with its two reads"
QUEUE_ARMS_ONLY = r"(\d+) jobs, two at a time: (\d+) repairs at fourteen empty cells"


def _once(rx, text, what):
    hits = re.findall(rx, text)
    assert len(hits) == 1, "the prose no longer states %s exactly once: %r" % (what, hits)
    return int(hits[0])


def _either(forms, text, what):
    """The number one of two sentence shapes quotes, or 0 when the document uses the shape that says
    there are none."""
    for rx, numbered in forms:
        assert len(re.findall(rx, text)) <= 1, "the prose states %s twice" % what
    for rx, numbered in forms:
        hits = re.findall(rx, text)
        if hits:
            return int(hits[0]) if numbered else 0
    raise AssertionError("the prose states %s in neither of its two shapes: %r"
                         % (what, [rx for rx, _n in forms]))


def claims(nb=NB):
    """The numbers the prose tells the researcher, read off the document rather than the generator.

    Each one has to appear exactly once: a sentence reworded so the count no longer shows up would
    otherwise make this whole section silently true.
    """
    text = markdown(nb)
    got = {"reads": _either(READ_FORMS, text, "how many items are reads"),
           "checkpoints": _either(CKPT_FORMS, text, "how many checkpoints travel inside"),
           "arm_items": _once(ARM_ITEMS, text, "how many items are arms"),
           "minutes": _once(MINUTES_CLAIM, text, "the pump's budget")}
    hits = re.findall(QUEUE_WITH_READS, text)
    assert len(hits) <= 1, hits
    if hits:
        got["queued_reads"], got["repairs"] = int(hits[0][0]), int(hits[0][1])
    else:
        hits = re.findall(QUEUE_ARMS_ONLY, text)
        assert len(hits) == 1, "the prose states the queue order in neither of its two shapes"
        got["jobs"], got["repairs"] = int(hits[0][0]), int(hits[0][1])
        got["queued_reads"] = 0
    return got


def test_the_prose_counts_the_jobs_this_document_actually_queues():
    for doc in docs():
        got, arms, rows = claims(doc), armed(doc), queued(doc)
        want = {"reads": len(rows), "arm_items": len(arms),
                "checkpoints": len(literal(WEIGHTS_TAG, doc)), "queued_reads": len(rows),
                "repairs": len([a for a in arms if a["kind"] == "arm-repair"]),
                "minutes": gen.MINUTES}
        if "jobs" in got:
            want["jobs"] = len(arms)
        assert got == want, (doc, got, want)


def test_the_prose_is_a_claim_about_this_document_and_not_about_the_generator():
    """A queue trimmed without regenerating the prose is the drift a hand edit produces.

    The shipped document has no reads to drop today, so the mutant runs against the shape document,
    whose prose quotes a read count.
    """
    doc = shapes_doc()
    nb2 = patched(READS_TAG, trimmed_reads_cell(doc), nb=doc)
    assert len(literal(READS_TAG, nb2)) == len(queued(doc)) - 1
    assert claims(nb2)["reads"] != len(literal(READS_TAG, nb2)), (
        "the read count is quoted from the generator, so a document whose queue no longer matches "
        "its own prose would pass")
    assert claims(nb2)["queued_reads"] != len(literal(READS_TAG, nb2)), (
        "section 4 and section 0 would disagree about the same queue and nobody would notice")


def test_the_boot_capacity_monitor_and_pump_are_the_cells_the_scaling_notebook_is_built_from():
    """The four shared cells are copied, not rewritten, so a fix there has to arrive untouched.

    `bf.render` strips a trailing newline off every cell, so a cell is compared as its own body and
    nothing else - a rewrite that also dropped a line would still differ.
    """
    def same(a, b, why="this document keeps its own copy of a shared cell"):
        x, y = a.rstrip("\n"), b.rstrip("\n")
        if x != y:
            i = next(k for k in range(max(len(x), len(y)) + 1) if x[k:k + 1] != y[k:k + 1])
            raise AssertionError("%s\n  copied: %r\n  shared: %r" % (why, x[i:i + 80], y[i:i + 80]))

    same(base._cell(base.SETUP_TAG, NB),
         bf.scaling_cell("setup").replace("c4_endgame_scaling.ipynb", gen.NAME),
         "the boot cell is no longer the tested one, with only the document's name changed")
    same(base._cell(RESOURCE_TAG, NB), bf.scaling_cell("resource"),
         "the capacity cell is no longer the tested one")
    same(base._cell(MONITOR_TAG, NB), bf.scaling_cell("monitor"),
         "the monitor cell is no longer the tested one")
    same(base._cell(base.TICK_TAG, NB),
         bf.scaling_cell("tick").replace("    snapshot(E)\n", "    snapshot()\n"),
         "the queue cell is no longer the tested one, with only the per-depth gate folded away")

    def without_budget(src):
        return "\n".join(l for l in src.split("\n") if not l.strip().startswith("MAX_MINUTES"))

    pump = base._cell(PUMP_TAG, NB)
    assert "MAX_MINUTES = %d\n" % gen.MINUTES in pump, pump[:400]
    assert gen.PROGRESS in pump, "the pump cell no longer holds the progress block tested below"
    same(without_budget(pump).replace(gen.PROGRESS, "", 1),
         without_budget(bf.scaling_cell("pump")),
         "this document keeps its own pump, so a runtime-death fix would not reach it")
    forked = base._cell(MONITOR_TAG, NB).replace("  wait  ", "  waiting ")
    assert forked.rstrip("\n") != bf.scaling_cell("monitor").rstrip("\n"), (
        "the monitor cell has no `wait` line, so the mutation this test is meant to catch would "
        "pass it")


def test_the_pump_prints_the_round_of_the_arm_that_is_still_running():
    """A frozen `39/48` is not a deadlock: an arm writes its JSON on the final round, so the count of
    finished jobs cannot move while a fourteen-empty repair works.

    So the pump has to name what the live job is doing - and only the live jobs. A job whose log does
    not exist yet would raise inside the pump, and a pump that dies in the middle of a session is the
    failure this cell exists to prevent, which is why the block reads the file it just checked.
    """
    assert gen.PROGRESS in base._cell(PUMP_TAG, NB), (
        "the pump cell no longer carries the progress block, so a long session has no signal at all")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pumplog_"))
    KEEP.append(tmp)
    log = tmp / "one.log"
    log.write_text("\n".join(["base fit ... [467s]",
                              "[repair it   7] W 0 B 1 D 0.5123 drawn 3/40 viol 266",
                              "[repair it   8] W 0 B 1 D 0.6041 drawn 4/40 viol 266"]),
                   encoding="utf-8")
    jobs = [{"name": "repair 14 vacias s0", "out": str(tmp / "arm.json"), "log": str(log)},
            {"name": "a finished read", "out": str(log), "log": str(log)},
            {"name": "a read with no log yet", "out": str(tmp / "nobody.json"),
             "log": str(tmp / "absent.log")}]

    def pids(pat):
        return ("arm.json" in pat) or ("nobody.json" in pat)

    def report(block):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(textwrap.dedent(block), {"JOBS": jobs, "os": os, "pids": pids})
        return out.getvalue()

    # the block as the document on disk holds it, not as the generator remembers it
    cell = base._cell(PUMP_TAG, NB)
    block = cell[cell.index("    for j in JOBS:"):cell.index("    left = [j['name']")]
    text = report(block)
    assert "RUN   repair 14 vacias s0 | [repair it   8]" in text, text
    assert text.count("RUN") == 1, (
        "%r: a job that is not running, or has no log to read, was printed anyway" % text)

    oldest = report(block.replace("_l[-1]", "_l[0]"))
    assert "RUN   repair 14 vacias s0 | [repair it   8]" not in oldest, oldest
    assert "base fit" in oldest, oldest
    unguarded = block.replace(" and os.path.exists(j['log'])", "")
    assert unguarded != block, "the block does not check for a log, so this mutant tests nothing"
    try:
        report(unguarded)
    except OSError:
        pass
    else:
        raise AssertionError("a pump that opens a log which does not exist yet survived the run")


def test_this_is_the_only_document_that_owes_these_jobs():
    """One notebook was asked for, so no other document may queue the same work a second time."""
    outs = {r["out"] for r in queued()} | {r["out"] for r in armed()}
    assert len(outs) == len(queued()) + len(armed()), "two jobs write the same output file"
    for p in sorted((REPO / "colab").glob("*.ipynb")):
        if p == NB:
            continue
        src = "".join("".join(c["source"])
                      for c in json.loads(p.read_text(encoding="utf-8"))["cells"])
        shared = sorted(o for o in outs if o in src)
        assert not shared, "%s also queues %s: one runtime would pay for that read twice" % (
            p.name, shared)
    assert not (REPO / "colab" / "c4_endgame_14_holdout.ipynb").exists(), (
        "the fourteen-cell document this one replaced is still in colab/, where a reviewer would "
        "run both")
    assert (REPO / "runs" / "_superseded_c4_endgame_14_holdout.ipynb").exists(), (
        "the retired document was deleted instead of kept out of the way, so its last run has no "
        "source to check these numbers against")


def test_the_document_mounts_nothing_and_never_asks_the_researcher_to_sign_in():
    """No Drive, no upload, no credentials: the weights and the results cross on their own."""
    code = "".join(src for _i, src in base._code_cells(NB)).lower()
    for needle in ("drive.mount", "google.colab.drive", "getpass", "input(", "password"):
        assert needle not in code, needle
    assert "no Drive mount" in markdown(), markdown()[:600]


# ------------------------------------------------------------------------ the generator ----
def test_the_document_on_disk_is_exactly_what_its_generator_writes():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = gen.write(check_only=True)
    assert rc == 0, out.getvalue()
    assert "matches its generator" in out.getvalue(), out.getvalue()


def test_a_cell_edited_inside_the_notebook_is_caught_by_the_check():
    """`--check` is the gate that keeps this document honest; a bent budget line has to trip it."""
    pump = base._cell(PUMP_TAG, NB)
    bent = pump.replace("MAX_MINUTES = %d" % gen.MINUTES, "MAX_MINUTES = 599")
    assert bent != pump, "the pump cell has no budget line to bend, so this test checks nothing"
    nb2 = patched(PUMP_TAG, bent)
    old, out = gen.OUT_DIR, io.StringIO()
    gen.OUT_DIR = nb2.parent
    try:
        with contextlib.redirect_stdout(out):
            rc = gen.write(check_only=True)
    finally:
        gen.OUT_DIR = old
    assert rc == 1, out.getvalue()
    assert "DIFFERS" in out.getvalue(), out.getvalue()


def test_the_generator_refuses_to_write_a_notebook_with_nothing_owed():
    """An empty plan is a broken plan, not a finished deliverable.

    Writing an empty queue over `c4_finish_all.ipynb` would leave the researcher with a document that
    announces a run and does nothing, and the `--check` above would go on calling it a match.
    """
    before, old = NB.read_text(encoding="utf-8"), gen.plan
    gen.plan = lambda: ([], [])
    try:
        refused = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                gen.write()
        except SystemExit as exc:
            refused = str(exc)
    finally:
        gen.plan = old
    assert refused is not None, "the generator wrote a notebook that queues nothing"
    assert "nothing is pending" in refused, refused
    assert NB.read_text(encoding="utf-8") == before, "the refusal still rewrote the document"


def test_the_bundled_checks_all_pass_over_this_document():
    """The structural promises are the scaling notebook's and run over every generated document."""
    out = subprocess.run([sys.executable, str(REPO / "colab" / "build_bundle.py"),
                          "--check", "--all"], capture_output=True, text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stdout + out.stderr
    for script in ("build_reads_notebooks.py", "build_followup_notebooks.py"):
        out = subprocess.run([sys.executable, str(REPO / "colab" / script), "--check"],
                             capture_output=True, text=True, cwd=str(REPO))
        assert out.returncode == 0, "%s --check: %s%s" % (script, out.stdout, out.stderr)


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
