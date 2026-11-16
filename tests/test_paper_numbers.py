"""The paper's tables are emitted by scripts; this runs them, so a drifted JSON fails here.

`runs/_scaling_table.py`, `_blind_table.py`, `_scope_table.py` and `_end14_table.py` each read
`data/` and assert the numbers the paper prints: the published scaling rows, the loop/verifier
agreement on both suites, the union identity, and the fourteen-cell stops. Those assertions were
real but nothing ran them. Flipping `data/c4_end12_s1.json` from `W=0, 39/40` to `W=1, 40/40`
left all 184 tests green, because the only guard of the paper's numbers was a manual run.

Each script is run as a subprocess with the repository as working directory: they resolve `data/`
relatively and exit non-zero the moment an assertion stops holding, which is what the test needs
to see. The second test covers the failure mode itself - a new `*_table.py` added to `runs/` and
forgotten here would reopen the same hole.
"""
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
EMITTERS = (
    "runs/_scaling_table.py",
    "runs/_blind_table.py",
    "runs/_scope_table.py",
    "runs/_end14_table.py",
)


def test_every_table_emitter_still_agrees_with_the_archived_json():
    for rel in EMITTERS:
        done = subprocess.run([sys.executable, rel], cwd=str(REPO),
                              capture_output=True, text=True)
        assert done.returncode == 0, (
            "%s no longer agrees with data/, so a number the paper prints has drifted:\n%s\n%s"
            % (rel, done.stdout[-4000:], done.stderr[-4000:]))


def test_the_emitter_list_names_every_table_script_under_runs():
    """An unwired emitter is how the numbers went unguarded in the first place."""
    present = {p.name for p in (REPO / "runs").glob("*_table.py")}
    named = {pathlib.Path(rel).name for rel in EMITTERS}
    assert present == named, (
        "runs/ has table scripts this test does not run: %s" % sorted(present - named))
