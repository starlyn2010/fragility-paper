"""The reproducibility section names scripts; a clean clone has to contain them.

`docs/REPRODUCING.md`, `README.md`, `CHANGELOG.md` and the paper all tell a reader to run things
under `runs/`. That directory is also where every checkpoint, transcript and scratch folder lands,
so `.gitignore` excludes it wholesale - and the exclusion silently swallowed the drivers too:
eighteen emitters and guards the paper cites were invisible to git, which means the promise "run
this after a clean clone" was false for every one of them. Nothing in the test suite noticed,
because nothing in the test suite asked git what it could see.

This asks. A cited script has to exist, and it has to be a file git is willing to take: ignored is
the failure mode that hides in plain sight, because the file is right there on the author's disk.
Whether it has been committed yet is the author's decision, not a property of the repository, so
being visible is what is asserted here.

Run with pytest, or directly: python tests/test_reproducibility.py
"""
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
DOCS = ("README.md", "CHANGELOG.md", "docs/REPRODUCING.md", "docs/DATA_CARD.md",
        "paper/paper_v2.tex")
# the paper escapes underscores for LaTeX, so `runs/\_refit\_bases.py` names a real file
CITED = re.compile(r"runs/[A-Za-z0-9_\\-]+\.(?:py|sh)")


def cited_scripts():
    """Every concrete runs/ script the prose names, with the LaTeX escapes undone."""
    found = set()
    for rel in DOCS:
        p = REPO / rel
        if not p.is_file():
            continue
        # the paper breaks a long \texttt path with \allowbreak, which is typography and not
        # part of the name: without removing it first, two cited scripts read as files that
        # do not exist and the test blames the paper for its own line breaking
        text = p.read_text(encoding="utf-8").replace("\\allowbreak", "")
        for hit in CITED.findall(text):
            name = hit.replace("\\", "")
            if "*" in name or "{" in name:
                continue
            found.add(name)
    return sorted(found)


def git(*args):
    return subprocess.run(["git"] + list(args), cwd=str(REPO), capture_output=True, text=True)


def test_the_prose_names_scripts_that_exist():
    named = cited_scripts()
    assert len(named) >= 10, "the prose cites almost nothing under runs/, so this test is vacuous"
    missing = [s for s in named if not (REPO / s).is_file()]
    assert not missing, "these are cited but not in the repository: %s" % ", ".join(missing)


def test_no_cited_script_is_hidden_from_git():
    named = cited_scripts()
    ignored = [s for s in named if git("check-ignore", "-q", s).returncode == 0]
    assert not ignored, (
        "git ignores these, so a clean clone cannot run what the paper says to run: %s"
        % ", ".join(ignored))


def test_the_artifact_ships_the_drivers_it_prunes_the_weights_from():
    """`prune runs` in MANIFEST.in threw the drivers out of the sdist along with the checkpoints."""
    manifest = (REPO / "MANIFEST.in").read_text(encoding="utf-8")
    assert not re.search(r"^prune runs$", manifest, re.M), (
        "the sdist prunes the whole of runs/, including every driver the paper cites")
    assert re.search(r"^include runs/\*\.py$", manifest, re.M), manifest


if __name__ == "__main__":
    for fn in (test_the_prose_names_scripts_that_exist, test_no_cited_script_is_hidden_from_git,
               test_the_artifact_ships_the_drivers_it_prunes_the_weights_from):
        fn()
        print("ok %s" % fn.__name__)
    print("%d scripts cited" % len(cited_scripts()))