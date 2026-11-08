"""Contract tests for `runs/_integrate_reads.py`, the cell that decides what enters `data/`.

The notebook ships a zip when a gate is complete, but "complete" is a claim about files existing,
not about them agreeing with the arms they read. This script is the only thing standing between a
browser download and the published tables, so six promises are tested here, each against a mutant
built the way an accident would build it:

1. a read that does not reproduce the certificate the arm logged beside it is the wrong weights,
   and the whole ingest stops - a repaired read against `repair.final`, a base read against
   `base.certificate`, both only when the read is on the suite that certificate was computed over.
2. a union read has to stay the arithmetic `sec:union` argues about: `W_U = min(W_A, W_B)`, the
   drawn counts add, the exact mean is the mean of the two halves. Broken in any of those four and
   nothing is written.
3. a suite inside the zip that is not byte-identical to the committed one is a different deck, so
   the comparison is not the published one - the refusal happens before any read is even parsed.
4. an archived measurement is never overwritten, and a file the queue does not owe is refused
   rather than trusted: that is how a zip from an older generation announces itself.
5. the report is about this repository: `--check` writes nothing, a re-run writes nothing, and the
   queue it prints at the end shrinks by exactly the reads that landed.
6. a gate that *trains* ships the archive its own reads are measured against. That archive has to
   be the depth, seed, deck and budget the plan asked for, and the reads have to be checked against
   the copy inside the zip - reading it from disk would find nothing and check no read at all.

Nothing here touches `~/Downloads` or `data/`: every sandbox is a temp directory with its own fake
repository, fake Downloads and fake plan, and the module globals are restored after each test. The
fabricated arm logs and verifier JSONs are built from the same four keys the script compares, so a
green run means the refusal fired for the reason named in it.

Run with pytest, or directly: python tests/test_integrate_reads.py
"""
import contextlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import zipfile

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO / "colab", REPO / "runs"):
    sys.path.insert(0, str(_p))
import reads_plan as rp                                                  # noqa: E402
import _integrate_reads as ir                                            # noqa: E402

SUITE_A = "data/c4_endgame_roots_8_40_s0.json"
SUITE_B = "data/c4_endgame_roots_8_40_holdoutB.json"
SUITE_U = "data/c4_endgame_roots_8_80_AplusB.json"
SUITE_A14 = "data/c4_endgame_roots_14_40_s0.json"
SUITE_B14 = "data/c4_endgame_roots_14_40_holdoutB.json"
# The names the script itself goes looking for, spelled out of `ir.GATES` so a rename in the
# generator cannot leave these tests asserting that a missing download was refused for the wrong
# reason. An arm gate is named for its depth, with no `reads_` prefix to strip.
GATE_NAMES = {f.split("_", 1)[1] if "_" in f else f: "c4_finishall_%s_results.zip" % f
              for f in ir.GATES}
# the plan's own arm rows, before a sandbox replaces the function with its fake queue
REAL_ARM_ROWS = rp.arm_rows
# the published budget, read off the real twelve-cell archives once, at import: a number typed
# here would be the second table the plan exists to avoid
BUDGET_14 = rp.arm_budget(14)


def cert(w=1.0, b=1.0, d=1.0, drawn=40, roots=40):
    """A certificate as the loop writes it: the four keys the script compares, plus its size."""
    return {"worst_draw_rate": w, "best_draw_rate": b, "random_draw_rate_exact": d,
            "n_roots_drawn_worst": drawn, "roots": roots}


def read_json(c, ckpt, suite, **over):
    """A verifier JSON carrying every field the script insists on reading."""
    v = dict(c)
    v.update({"ckpt": ckpt, "roots_file": suite, "deterministic_orders": 120,
              "order_min": c["worst_draw_rate"], "order_mean": 0.9991, "order_max": 1.0,
              "adaptive_games_per_root": 40, "adaptive_mean": 0.9912, "adaptive_min": 0.8})
    v.update(over)
    return v


def arm_doc(roots_file, base, final, rounds=3, args=None):
    """An archive as the loop writes it: the numbers every read of this arm is measured against."""
    return {"args": {"roots_file": roots_file, **(args or {})},
            "base": {"certificate": base, "fit_epochs": 40, "fit_loss": 0.12},
            "repair": {"final": final, "log": [{"round": i} for i in range(rounds)],
                       "aggregate": [0] * 5, "per_root": [],
                       "checkpoint": "runs/x/c4_gru.pt",
                       "base_checkpoint": "runs/x/c4_gru_base.pt"}}


def row(kind, arm, suite, out, suite_file=SUITE_A, gate="reads_closure", group="closure",
        arm_json=None):
    return {"kind": kind, "arm": arm, "E": 8, "suite": suite, "suite_file": suite_file,
            "ckpt": "runs/%s/c4_gru%s.pt" % (arm, "_base" if kind.startswith("base") else ""),
            "out": out, "group": group, "gate": gate,
            "arm_json": arm_json or ("data/%s.json" % arm), "key": "%s8" % suite}


class Box:
    """A temp repository, a temp Downloads and a temp plan, wired into both modules."""

    def __init__(self, tag):
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="c4int_%s_" % tag))
        self.repo = self.root / "repo"
        self.data = self.repo / "data"
        self.data.mkdir(parents=True)
        self.dl = self.root / "Downloads"
        self.dl.mkdir()
        self.rows = []
        self.arm_rows = []
        self.saved = (ir.REPO, ir.DOWNLOADS, rp.REPO, rp.pending, rp.exists, rp.arm_rows)

    def suite(self, rel):
        p = self.repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"roots": list(range(40 if "_40_" in rel else 80))}),
                     encoding="utf-8")
        return rel

    def arm(self, rel, roots_file, base, final, rounds=3, args=None):
        """An archived arm: the loop's own numbers, which every read is measured against."""
        (self.data / pathlib.Path(rel).name).write_text(
            json.dumps(arm_doc(roots_file, base, final, rounds, args)), encoding="utf-8")

    def gate(self, name, entries):
        """Rewrite one gate zip: JSONs under data/, suites under data/, anything else a log."""
        with zipfile.ZipFile(self.dl / GATE_NAMES[name], "w") as z:
            for path, payload in entries.items():
                z.writestr(path, payload if isinstance(payload, (bytes, str))
                           else json.dumps(payload))

    def send(self, payload, gate, extra=None):
        """One shipped read, in the zip that would have carried it, with its decks."""
        entries = {**all_suites(self), "data/" + gate["out"].split("/", 1)[1]: payload}
        entries.update(extra or {})
        self.gate(gate["gate"].replace("reads_", ""), entries)
        return gate

    def install(self, rows):
        self.rows = rows
        ir.REPO, ir.DOWNLOADS = self.repo, self.dl
        rp.REPO = self.repo
        rp.pending = lambda all_reads=None: list(self.rows)
        rp.exists = lambda rel: (self.repo / rel).is_file()
        rp.arm_rows = lambda: list(self.arm_rows)
        return self

    def close(self):
        (ir.REPO, ir.DOWNLOADS, rp.REPO, rp.pending, rp.exists,
         rp.arm_rows) = self.saved
        shutil.rmtree(self.root, ignore_errors=True)

    def suite_bytes(self, rel):
        return (self.repo / rel).read_text(encoding="utf-8")

    def out(self, rel):
        return self.repo / rel

    def run(self, *flags):
        """Invoke main() the way the shell would; return (exit code, everything it said).

        The refusals are `raise SystemExit(message)`, so their sentence lands on stderr; the tests
        read the run the way a person at a terminal would, both streams together.
        """
        buf = io.StringIO()
        argv = sys.argv
        sys.argv = ["_integrate_reads.py"] + list(flags)
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = ir.main()
            return (0 if rc in (0, None) else rc), buf.getvalue()
        except SystemExit as exc:
            code = exc.code
            said = buf.getvalue()
            if isinstance(code, str):
                # the interpreter would print this to stderr; here it only exists on the exception
                said += code + "\n"
                code = 1
            return (code if isinstance(code, int) else 0), said
        finally:
            sys.argv = argv


def all_suites(box, suites=(SUITE_A, SUITE_B, SUITE_U)):
    return {"data/" + s.split("/", 1)[1]: box.suite_bytes(s) for s in suites}


@contextlib.contextmanager
def sandbox(tag, rows):
    """A sandboxed ingest: the plan is `rows`, and all three gate zips exist as they must."""
    box = Box(tag)
    for s in (SUITE_A, SUITE_B, SUITE_U):
        box.suite(s)
    box.install(rows)
    for g in GATE_NAMES:
        box.gate(g, all_suites(box))
    try:
        yield box
    finally:
        box.close()


# --------------------------------------------------------------------------- the happy path ----
def test_an_honest_zip_is_ingested_and_reports_no_refusals():
    r = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    logged = cert()
    payload = read_json(logged, r["ckpt"], SUITE_A)
    with sandbox("honest", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, cert(w=0.5), logged)
        box.send(payload, r)
        rc, out = box.run()
        assert rc == 0, out
        assert json.loads(box.out(r["out"]).read_text(encoding="utf-8")) == payload
        assert "COINCIDE" in out, out
        assert "0 rechazos" in out, out


def test_the_log_rides_in_the_zip_and_lands_outside_the_signature():
    logged = cert()
    r = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    with sandbox("log", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, cert(w=0.5), logged)
        box.send(read_json(logged, r["ckpt"], SUITE_A), r,
                 extra={"runs/_colab_finishall_closure.log": "done\n"})
        rc, out = box.run()
        assert rc == 0, out
        assert (box.repo / "runs/_colab_finishall_closure.log").is_file(), out


def test_the_reported_queue_shrinks_by_exactly_what_landed():
    a = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    b = row("base-on-B", "c4_end8_s1", "B", "data/c4_end8_base_s1_onB.json",
            suite_file=SUITE_B, gate="reads_pairing", group="pairing")
    with sandbox("shrink", [a, b]) as box:
        box.arm(a["arm_json"], SUITE_A, cert(w=0.5), cert())
        box.arm(b["arm_json"], SUITE_A, cert(w=0.4), cert(d=0.6))
        box.send(read_json(cert(), a["ckpt"], SUITE_A), a)
        rc, out = box.run()
        assert rc == 0, out
        assert "1 archivos nuevos" in out, out
        assert "la cola que queda: 1" in out, out
        assert box.out(a["out"]).is_file() and not box.out(b["out"]).exists(), out


# --------------------------------------------------------------------------- promise 1 ----
def test_a_repaired_read_that_disagrees_with_the_arm_is_the_wrong_weights():
    logged = cert(w=1.0, drawn=40)
    r = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    bent = read_json(cert(w=0.975, drawn=39), r["ckpt"], SUITE_A)
    with sandbox("bent", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, cert(w=0.5), logged)
        box.send(bent, r)
        rc, out = box.run()
        assert rc != 0, out
        assert "NO COINCIDE" in out, out
        assert "no son los del brazo" in out, out
        assert not box.out(r["out"]).exists(), "a refused read still reached data/"


def test_an_arms_own_read_is_measured_against_the_certificate_it_ended_with():
    """The queue calls a new arm's read `arm-read-on-A`, and that is a read of repaired weights.

    `reads_plan.expected` used to recognise only `repaired-on-` and `union-repaired-`, so the
    guard this script runs compared a fourteen-empty arm's read against the *base* certificate
    the loop started from. Both halves are asserted here, because either one alone would pass
    with the comparison pointed at the wrong tier: the honest read has to be accepted, and a
    read of the base weights has to be refused.
    """
    base, final = cert(w=0.5, drawn=20), cert(w=1.0, drawn=40)
    assert base != final, "the two tiers agree, so this test could not tell them apart"
    r = row("arm-read-on-A", "c4_end14_s0", "A", "data/c4_end14_verify_s0.json")
    with sandbox("armtier", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, base, final)
        box.send(read_json(final, r["ckpt"], SUITE_A), r)
        rc, out = box.run()
        assert rc == 0, out
        assert "COINCIDE" in out, out
        assert box.out(r["out"]).is_file(), out
    with sandbox("armbasetier", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, base, final)
        box.send(read_json(base, r["ckpt"], SUITE_A), r)
        rc, out = box.run()
        assert rc != 0, "a read of the base weights passed for the repaired arm:\n" + out
        assert "NO COINCIDE" in out, out
        assert not box.out(r["out"]).exists(), "a refused read still reached data/"


def test_a_single_violation_is_enough_to_refuse_a_read():
    """The mutation an accident actually makes: 40 roots become 39, and nothing else moves."""
    logged = cert(w=1.0, drawn=40)
    r = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    with sandbox("one", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, cert(w=0.5), logged)
        box.send(read_json(logged, r["ckpt"], SUITE_A, n_roots_drawn_worst=39), r)
        rc, out = box.run()
        assert rc != 0, out
        assert "no son los del brazo c4_end8_s0" in out, out


def test_a_base_read_is_measured_against_the_certificate_the_loop_began_with():
    """A base checkpoint reproduces `base`, never the repaired `final` in a file of the same arm."""
    base, final = cert(w=0.6152, b=0.7, d=0.6152, drawn=14), cert()
    r = row("base-on-A", "c4_end12_s3", "A", "data/c4_end12_base_s3_onA.json",
            gate="reads_pairing", group="pairing")
    with sandbox("basemeas", [r]) as box:
        box.arm(r["arm_json"], "data/c4_endgame_roots_12_40_s0.json", base, final)
        box.send(read_json(final, r["ckpt"], SUITE_A), r)
        rc, out = box.run()
        assert rc != 0, out
        assert "no son los del brazo" in out, out
        assert not box.out(r["out"]).exists(), out
        box.send(read_json(base, r["ckpt"], SUITE_A), r)
        rc, out = box.run()
        assert rc == 0, out
        assert "COINCIDE" in out, out


def test_a_blind_base_read_carries_no_expectation_and_still_needs_its_fields():
    """Suite B has no archived certificate, so only the field check can refuse this one."""
    r = row("base-on-B", "c4_end12_s3", "B", "data/c4_end12_base_s3_onB.json",
            suite_file=SUITE_B, gate="reads_pairing", group="pairing")
    with sandbox("blindbase", [r]) as box:
        box.arm(r["arm_json"], "data/c4_endgame_roots_12_40_s0.json",
                 cert(w=0.6152), cert())
        good = read_json(cert(w=0.1727, b=0.3, d=0.1727, drawn=3), r["ckpt"], SUITE_B)
        box.send(good, r)
        rc, out = box.run()
        assert rc == 0, out
        assert "COINCIDE" not in out, out
        box.out(r["out"]).unlink()
        box.gate("pairing", {**all_suites(box), "data/" + r["out"].split("/", 1)[1]:
                             {k: v for k, v in good.items() if k != "adaptive_min"}})
        rc, out = box.run()
        assert rc != 0 and "le faltan las llaves" in out, out


def test_a_file_that_is_not_json_is_refused():
    r = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    with sandbox("notjson", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, cert(w=0.5), cert())
        box.send("{ worst_draw_rate: 1.0", r)
        rc, out = box.run()
        assert rc != 0, out
        assert "no es JSON leible" in out, out
        assert not box.out(r["out"]).exists(), out


# --------------------------------------------------------------------------- promise 2 ----
UNION_ARM = "data/c4_end8U_s0.json"
HALF_A, HALF_B = cert(w=1.0, drawn=40), cert(w=0.975, d=0.9917, drawn=39)
GOOD_U = cert(w=0.975, b=1.0, d=(1.0 + 0.9917) / 2, drawn=79, roots=80)


def union_rows(u=GOOD_U):
    rows, jsons = [], []
    for s, c, suite in (("A", HALF_A, SUITE_A), ("B", HALF_B, SUITE_B), ("U", u, SUITE_U)):
        r = row("union-repaired-%s" % s, "c4_end8U_s0", s, "data/c4_end8U_s0_%s.json" % s,
                suite_file=suite, gate="reads_union", group="union", arm_json=UNION_ARM)
        rows.append(r)
        jsons.append(read_json(c, r["ckpt"], suite, roots=c["roots"]))
    return rows, jsons


def ship_union(box, rows, jsons):
    entries = all_suites(box)
    for r, j in zip(rows, jsons):
        entries["data/" + r["out"].split("/", 1)[1]] = j
    box.gate("union", entries)


def test_a_union_that_is_not_the_minimum_of_its_halves_is_refused():
    rows, jsons = union_rows()
    with sandbox("union", rows) as box:
        box.arm(UNION_ARM, SUITE_U, cert(w=0.5), GOOD_U)
        ship_union(box, rows, jsons)
        rc, out = box.run()
        assert rc == 0, out
        assert "identidad W_U = min(W_A, W_B): OK" in out, out
        for r in rows:
            box.out(r["out"]).unlink()
        broken = (cert(w=1.0, d=GOOD_U["random_draw_rate_exact"], drawn=79, roots=80),
                  cert(w=0.975, b=1.0, d=GOOD_U["random_draw_rate_exact"], drawn=78, roots=80),
                  cert(w=0.975, b=1.0, d=GOOD_U["random_draw_rate_exact"] + 0.001, drawn=79,
                       roots=80),
                  cert(w=0.975, b=1.0, d=GOOD_U["random_draw_rate_exact"], drawn=79, roots=40))
        for bad in broken:
            _, jsons_bad = union_rows(bad)
            ship_union(box, rows, jsons_bad)
            rc, out = box.run()
            assert rc != 0, "%s passed the union identity: %s" % (bad, out)
            assert "no es el minimo de sus dos mitades" in out, out
            assert not any(box.out(r["out"]).exists() for r in rows), out


def test_the_union_identity_is_checked_even_when_the_halves_were_already_archived():
    rows, jsons = union_rows()
    with sandbox("uniondisk", rows) as box:
        box.arm(UNION_ARM, SUITE_U, cert(w=0.5), GOOD_U)
        for r, j in zip(rows[:2], jsons[:2]):
            box.out(r["out"]).write_text(json.dumps(j), encoding="utf-8")
        bad = cert(w=1.0, b=1.0, d=GOOD_U["random_draw_rate_exact"], drawn=79, roots=80)
        ship_union(box, rows[2:], union_rows(bad)[1])
        rc, out = box.run()
        assert rc != 0, out
        assert "identidad W_U = min(W_A, W_B): ROTO" in out, out
        assert not box.out(rows[2]["out"]).exists(), out


# --------------------------------------------------------------------------- promise 3 ----
def test_a_zip_carrying_a_different_deck_refuses_the_whole_ingest():
    """The other two gates carry a good copy of the same suite, and that must not launder the bad one.

    A deck is what a read is *about*: if the closure zip's suite A is not the committed forty
    positions, the certificate beside it is not the certificate the arm logged, whether or not some
    other zip happened to carry the right file.
    """
    r = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    with sandbox("deck", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, cert(w=0.5), cert())
        box.gate("closure", {**all_suites(box),
                             "data/" + SUITE_A.split("/", 1)[1]: json.dumps({"roots": [999]}),
                             "data/" + r["out"].split("/", 1)[1]:
                             read_json(cert(), r["ckpt"], SUITE_A)})
        rc, out = box.run()
        assert rc != 0, out
        assert "DISTINTA" in out, out
        assert "no es la suite confirmada" in out, out
        assert GATE_NAMES["closure"] in out, out
        assert not box.out(r["out"]).exists(), "a bad deck still let its read through"


# --------------------------------------------------------------------------- promise 4 ----
def test_a_file_the_queue_does_not_owe_is_refused_rather_than_trusted():
    r = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    stray = "data/c4_end9_verify_s0.json"
    with sandbox("stray", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, cert(w=0.5), cert())
        box.send(read_json(cert(), r["ckpt"], SUITE_A), r,
                 extra={"data/" + stray.split("/", 1)[1]: read_json(cert(), r["ckpt"], SUITE_A)})
        rc, out = box.run()
        assert rc != 0, out
        assert "que la cola no debe" in out, out
        assert not box.out(stray).exists(), out
        assert not box.out(r["out"]).exists(), "the honest half was written anyway"


def test_an_archived_read_is_never_overwritten():
    r = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    with sandbox("clobber", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, cert(w=0.5), cert())
        other = read_json(cert(), "runs/other_seed/c4_gru.pt", SUITE_A)
        box.out(r["out"]).write_text(json.dumps(other), encoding="utf-8")
        box.send(read_json(cert(), r["ckpt"], SUITE_A), r)
        rc, out = box.run()
        assert rc != 0, out
        assert "no se pisa una medicion archivada" in out, out
        assert json.loads(box.out(r["out"]).read_text(encoding="utf-8")) == other, out


# --------------------------------------------------------------------------- promise 5 ----
def test_check_reports_the_same_numbers_and_writes_nothing():
    r = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    with sandbox("check", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, cert(w=0.5), cert())
        box.send(read_json(cert(), r["ckpt"], SUITE_A), r,
                 extra={"runs/_colab_finishall_closure.log": "done\n"})
        rc, out = box.run("--check")
        assert rc == 0, out
        assert "nada se escribio" in out, out
        assert not box.out(r["out"]).exists(), out
        assert not (box.repo / "runs/_colab_finishall_closure.log").exists(), out


def test_a_reingest_of_the_same_bytes_writes_nothing_and_says_so():
    r = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    with sandbox("twice", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, cert(w=0.5), cert())
        box.send(read_json(cert(), r["ckpt"], SUITE_A), r)
        rc, first = box.run()
        assert rc == 0, first
        stamp = box.out(r["out"]).stat().st_mtime_ns
        rc, again = box.run()
        assert rc == 0, again
        assert "ya estaba" in again, again
        assert "0 archivos nuevos" in again, again
        assert box.out(r["out"]).stat().st_mtime_ns == stamp, "the second run rewrote the read"


def test_a_gate_zip_that_never_downloaded_stops_the_run():
    r = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    with sandbox("nozip", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, cert(w=0.5), cert())
        box.send(read_json(cert(), r["ckpt"], SUITE_A), r)
        (box.dl / GATE_NAMES["pairing"]).unlink()
        rc, out = box.run()
        assert rc != 0, out
        assert "no esta en" in out, out
        assert not box.out(r["out"]).exists(), out


def test_a_zip_from_an_older_generation_names_the_gate_that_is_stale():
    """A gate that owes nothing and ships something is the same refusal, one gate later."""
    r = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    with sandbox("stale", [r]) as box:
        box.arm(r["arm_json"], SUITE_A, cert(w=0.5), cert())
        box.gate("union", {**all_suites(box),
                           "data/c4_end8U_s9_A.json": read_json(cert(), "x", SUITE_A)})
        box.send(read_json(cert(), r["ckpt"], SUITE_A), r)
        rc, out = box.run()
        assert rc != 0, out
        assert "otra generacion" in out, out


# --------------------------------------------------------------------------- promise 6 ----
# A gate that trains ships the expectation its own reads are measured against. The rows below are
# the plan's real ones, so a rename in `reads_plan` cannot leave these tests asserting a fiction.
SUITES_14 = (SUITE_A14, SUITE_B14)
BASE_14 = cert(w=0.4856, b=1.0, d=0.4856, drawn=6)
FINAL_14 = cert(w=1.0, drawn=40)
BLIND_14 = cert(w=0.175, b=0.4, d=0.175, drawn=7)


def plan14(seeds=(0,)):
    """The plan's rows for depth fourteen, derived against whichever repository is installed."""
    saved = rp.ARMS
    rp.ARMS = ((14, tuple(seeds)),)
    try:
        return REAL_ARM_ROWS()
    finally:
        rp.ARMS = saved


def ship14(box, rows, base=BASE_14, final=FINAL_14, args=None, reads=None):
    """The end14 zip: the arm, the reads it makes published, and the decks they were taken on."""
    repair = next(r for r in rows if r["kind"] == "arm-repair")
    doc = arm_doc(repair["suite_file"], base, final,
                  args={"empties": repair["E"], "seed": repair["seed"], **BUDGET_14,
                        **(args or {})})
    entries = all_suites(box, SUITES_14)
    entries[repair["out"]] = doc
    for r in rows:
        if r["kind"] != "arm-repair":
            c = final if r["suite"] == "A" else BLIND_14
            entries[r["out"]] = read_json(c, r["ckpt"], r["suite_file"])
    entries.update(reads or {})
    box.gate("end14", entries)
    return doc


@contextlib.contextmanager
def sandbox14(tag, seeds=(0,)):
    """A sandbox whose plan is the real fourteen-empty one: three rows per seed, no reads owed.

    `reads()` knows nothing about a depth with no archived arm, so the arm's two reads arrive
    through `arm_rows()` - which is the whole reason this gate needs its own ingestion path.
    """
    box = Box(tag)
    for s in (SUITE_A, SUITE_B, SUITE_U) + SUITES_14:
        box.suite(s)
    box.install([])
    # a twelve-cell donor inside the sandbox, so the planned depth inherits the published budget
    box.arm("data/c4_end12_s0.json", rp.suite("A", 12), cert(w=0.5), cert(),
            args={"empties": 12, "seed": 0, **BUDGET_14})
    box.arm_rows = plan14(seeds)
    for g in GATE_NAMES:
        box.gate(g, all_suites(box, SUITES_14 if g == "end14" else (SUITE_A, SUITE_B, SUITE_U)))
    try:
        yield box, box.arm_rows
    finally:
        box.close()


def test_a_training_gate_lands_its_arm_and_the_reads_it_certifies():
    assert BASE_14 != FINAL_14, "the two tiers agree, so no read could tell them apart"
    with sandbox14("armgate") as (box, rows):
        repair = next(r for r in rows if r["kind"] == "arm-repair")
        doc = ship14(box, rows)
        rc, out = box.run()
        assert rc == 0, out
        assert "mazo y presupuesto publicados comprobados" in out, out
        assert "2 lecturas validadas, 1 brazos, 0 rechazos" in out, out
        assert "COINCIDE" in out, out
        # the archive lands byte for byte as the loop wrote it: re-serialising it here would change
        # the file the paper's emitters hash
        assert box.out(repair["out"]).read_bytes() == json.dumps(doc).encode("utf-8"), out
        assert all(box.out(r["out"]).is_file() for r in rows), out


def test_a_read_of_the_base_weights_is_refused_by_the_arm_that_shipped_beside_it():
    """The expectation has to come out of the zip: there is no arm on disk to read it from yet."""
    with sandbox14("armbase") as (box, rows):
        a = next(r for r in rows if r["kind"] == "arm-read-on-A")
        assert not box.out(a["arm_json"]).exists(), "the arm is already archived: proves nothing"
        ship14(box, rows, reads={a["out"]: read_json(BASE_14, a["ckpt"], a["suite_file"])})
        rc, out = box.run()
        assert rc != 0, "a read of the base weights passed for the arm that shipped beside it:\n" + out
        assert "NO COINCIDE" in out and "no son los del brazo" in out, out
        assert not any(box.out(r["out"]).exists() for r in rows), "a refused gate still wrote"


def test_an_arm_under_another_budget_is_refused_even_when_its_reads_agree_with_it():
    """Both reads reproduce this arm exactly; it is still not the arm the paper published."""
    with sandbox14("armbudget") as (box, rows):
        ship14(box, rows, args={"agg_repeat": BUDGET_14["agg_repeat"] // 3})
        rc, out = box.run()
        assert rc != 0, out
        assert "presupuesto distinto del publicado en agg_repeat" in out, out
        assert "lecturas validadas" not in out, "the reads were trusted before the arm was:\n" + out
        assert not any(box.out(r["out"]).exists() for r in rows), out


def test_the_arm_has_to_be_the_depth_seed_and_deck_the_plan_asked_for():
    for i, (over, said) in enumerate((({"seed": 1}, "el plan pidio 14 vacias semilla 0"),
                                      ({"empties": 12}, "el plan pidio 14 vacias semilla 0"),
                                      ({"roots_file": SUITE_B14}, "certifico sobre"))):
        with sandbox14("armwhich%d" % i) as (box, rows):
            ship14(box, rows, args=over)
            rc, out = box.run()
            assert rc != 0, "%r passed for the planned arm:\n%s" % (over, out)
            assert said in out, out
            assert not any(box.out(r["out"]).exists() for r in rows), out


def test_the_arm_gate_is_ingested_alone_while_other_gates_are_still_owed():
    """A depth that trains is the only work behind its zip, so the three read zips are not demanded."""
    other = row("repaired-on-A", "c4_end8_s0", "A", "data/c4_end8_verify_s0.json")
    with sandbox14("armonly") as (box, rows):
        box.rows = [other]
        for g in ("union", "closure", "pairing"):
            (box.dl / GATE_NAMES[g]).unlink()
        ship14(box, rows)
        rc, out = box.run("--only", "end14")
        assert rc == 0, out
        assert all(box.out(r["out"]).is_file() for r in rows), out
        assert not box.out(other["out"]).exists(), out
        assert "la cola que queda: 1" in out, out


def test_a_gate_the_repository_does_not_owe_is_refused_by_name():
    with sandbox14("armstale") as (box, rows):
        ship14(box, rows)
        rc, out = box.run("--only", "reads_union")
        assert rc != 0, out
        assert "no es una puerta que data/ deba ahora" in out, out
        assert "end14" in out, out
        assert not any(box.out(r["out"]).exists() for r in rows), out


def test_a_repository_that_owes_nothing_says_so_and_opens_no_zip():
    with sandbox14("armdone") as (box, rows):
        box.arm_rows = []
        rc, out = box.run()
        assert rc == 0, out
        assert "no debe ninguna lectura ni ningun brazo" in out, out
        assert "entradas leidas" not in out, out


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
