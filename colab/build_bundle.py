"""Embed into a scaling notebook the copy of the code and the root suites it runs.

Why this exists: a Colab runtime that has to fetch a private repository needs either a
GitHub authorisation the human has to complete in a browser, a token pasted into a
notebook, or a file uploaded through the sidebar. All three are human work, and the first
fails silently when the one-time code expires. The queues only need four modules, the
published root suite of each depth, its blind twin, the eighty-position unions of the depths
that have them and the archived arm
whose controls-from route re-derives it, roughly 340 KB, so each notebook carries them itself
as a base64 zip and its setup cell unpacks them into a working copy. Nothing is asked of the
person running it.

The suites travel for a scientific reason, not a convenience one: every published arm read
`data/c4_endgame_roots_<E>_40_s0.json`, and the seeds are comparable because they share it.
A runtime that sampled its own suite would answer a different question - and where two
notebooks run on two accounts, each sampling its own holdout would mean they never ran the
same test.

The bundle is not trusted: it records the sha256 of every file it holds and of the archive
as a whole, and the setup cell re-computes both. A mismatch falls back to the clone rather
than running modified code, and tests/test_colab_notebook.py asserts the embedded hashes
still equal the repository's own files, so the bundle cannot drift away from the code the
paper cites without a test failing.

    python colab/build_bundle.py                    # rebuild the scaling notebook
    python colab/build_bundle.py --all              # every notebook that carries one
    python colab/build_bundle.py --check --all      # exit 1 if any bundle is stale
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SCALING_NB = REPO / "colab" / "c4_endgame_scaling.ipynb"
NB = SCALING_NB                                  # the notebook built by hand so far
# every notebook that carries a bundle: the two generated ones are listed in
# colab/build_followup_notebooks.py, and each is injected from there
NOTEBOOKS = [SCALING_NB,
             REPO / "colab" / "c4_ladder_finish.ipynb",
             REPO / "colab" / "c4_finish_all.ipynb"]
CELL_MARKER = "BUNDLE_SHA256 = "     # the assignment, which only the bundle cell holds
SETUP_MARKER = "def bootstrap("
B64_WIDTH = 100

# exactly what the queues import or launch, and every root suite the repository holds -
# the published ones the arms read plus the blind holdouts sampled for the same depths (a
# runtime that re-sampled a suite would no longer share it with the arms already in the
# paper, nor with the other account running the companion notebook), and the archived seed-1
# arm the optional controls-from arms re-derive; stored under their repo-relative paths.
# The suites are globbed rather than listed, because "the bundle carries every suite the
# repository has committed" is the promise, and a hand-written list goes stale the day a new
# depth is sampled - which is exactly what --check is for.
BUNDLE_MODULES = [
    "src/c4_game.py",
    "src/c4_repair.py",
    "src/c4_endgame.py",
    "src/c4_endgame_verify.py",
    "data/c4_end10_s1.json",
]


def bundle_files():
    """Every root suite the notebooks read: the published one per depth, its holdout and the union.

    Three shapes. `c4_endgame_roots_<E>_40_s0.json` is what every published arm was certified over,
    `..._40_holdoutB.json` is the blind twin the repairs never saw, and `..._80_AplusB.json` is the
    first one followed by the second - the set a union arm certifies against, which the reads-only
    notebooks re-read and the scope claim depends on. `data/c4_endgame_roots_8_40_s1.json` and
    `_s2.json` are left over from the first depth-eight experiment, when each seed sampled its own
    suite; the published arms all read the `_s0` file (their archived `args.roots_file` says so), so
    those two are not inputs to anything and do not ride along.
    """
    data = REPO / "data"
    suites = sorted(p.relative_to(REPO).as_posix()
                    for p in data.glob("c4_endgame_roots_*_40_*.json")
                    if p.name.endswith(("_s0.json", "_holdoutB.json")))
    unions = sorted(p.relative_to(REPO).as_posix()
                    for p in data.glob("c4_endgame_roots_*_80_AplusB.json"))
    return BUNDLE_MODULES + suites + unions


BUNDLE_FILES = bundle_files()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_archive() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in BUNDLE_FILES:
            p = REPO / rel
            if not p.exists():
                raise SystemExit("%s is missing from the repository" % rel)
            zf.writestr(zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0)),
                        p.read_bytes())
    return buf.getvalue()


def bundle_cell(archive: bytes, commit: str) -> str:
    files = {rel: sha256((REPO / rel).read_bytes()) for rel in BUNDLE_FILES}
    blob = base64.b64encode(archive).decode()
    b64 = "\n".join("    '%s'" % blob[i:i + B64_WIDTH]
                    for i in range(0, len(blob), B64_WIDTH))
    return "\n".join([
        "# The bundle: this notebook carries the code it runs, so a runtime needs nothing",
        "# from GitHub - no authorisation, no token, no upload. Built by",
        "# colab/build_bundle.py from commit " + commit + ". Every file's sha256 is here and",
        "# the setup cell re-computes them, so a hand-edited or stale bundle falls back to",
        "# the clone instead of running modified code.",
        "import base64 as _b64, hashlib as _hl, io, zipfile",
        "",
        "BUNDLE_COMMIT = %r" % commit,
        "BUNDLE_SHA256 = %r" % sha256(archive),
        "BUNDLE_FILES = " + json.dumps(files, indent=4),
        "BUNDLE_B64 = (",
        b64,
        ")",
        "",
        "_archive = _b64.b64decode(BUNDLE_B64)",
        "assert _hl.sha256(_archive).hexdigest() == BUNDLE_SHA256, 'the bundle is corrupt'",
        "_zf = zipfile.ZipFile(io.BytesIO(_archive))",
        "assert sorted(_zf.namelist()) == sorted(BUNDLE_FILES), 'the bundle holds other files than it claims'",
        "for _n in _zf.namelist():",
        "    assert _hl.sha256(_zf.read(_n)).hexdigest() == BUNDLE_FILES[_n], 'bundle was modified: ' + _n",
        "print('this notebook carries its own code: commit %s, %d files, %.0f KB unpacked,'",
        "      ' %.0f KB inside the notebook - no GitHub sign-in needed'",
        "      % (BUNDLE_COMMIT, len(BUNDLE_FILES),",
        "        sum(z.file_size for z in _zf.infolist()) / 1024, len(_archive) / 1024))",
        "",
    ])


def repo_commit() -> str:
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() or "uncommitted"


def notebook_bundle_source(nb_path=None) -> str:
    for c in json.loads((nb_path or NB).read_text(encoding="utf-8"))["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and CELL_MARKER in src:
            return src
    return ""


def check(nb_path=None) -> int:
    """Compare the hashes the notebook claims against the repository as it stands."""
    src = notebook_bundle_source(nb_path)
    if not src:
        print("the notebook carries no bundle; run python colab/build_bundle.py")
        return 1
    ns = {}
    exec(src[: src.index("_archive =")], ns)
    missing = sorted(set(BUNDLE_FILES) - set(ns["BUNDLE_FILES"]))
    if missing:
        print("BUNDLE IS INCOMPLETE: the notebook carries no " + ", ".join(missing))
        print("rebuild with: python colab/build_bundle.py")
        return 1
    drift = [rel for rel, h in ns["BUNDLE_FILES"].items()
             if not (REPO / rel).exists() or sha256((REPO / rel).read_bytes()) != h]
    if drift:
        print("BUNDLE IS STALE for: " + ", ".join(drift))
        print("rebuild with: python colab/build_bundle.py")
        return 1
    if ns["BUNDLE_COMMIT"] != repo_commit():
        print("note: the bundle was built at %s and HEAD is now %s; its files still match, "
              "so it is usable" % (ns["BUNDLE_COMMIT"], repo_commit()))
    print("bundle matches the repository (%d files)" % len(ns["BUNDLE_FILES"]))
    return 0


def inject(source: str, nb_path=None) -> tuple:
    NB = pathlib.Path(nb_path) if nb_path else SCALING_NB
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def where(marker):
        return [i for i, c in enumerate(cells)
                if c["cell_type"] == "code" and marker in "".join(c["source"])]

    setup, old = where(SETUP_MARKER), where(CELL_MARKER)
    if len(setup) != 1:
        raise SystemExit("expected exactly one setup cell (with %s), found %d"
                         % (SETUP_MARKER, len(setup)))
    if old and old[0] == setup[0]:
        raise SystemExit("the setup cell itself holds the bundle marker; refusing to "
                         "overwrite it - that would delete bootstrap()")
    before = len(cells)
    cell = {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [x + "\n" for x in source.split("\n")][:-1]}
    if old:
        if old[0] < setup[0]:
            cells[old[0]] = cell
        else:
            cells.pop(old[0])                       # a bundle sitting after setup: move it
            cells.insert(setup[0], cell)
    else:
        cells.insert(setup[0], cell)
    assert len(cells) == before + (0 if old else 1)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    after = json.loads(NB.read_text(encoding="utf-8"))["cells"]
    new_setup = [i for i, c in enumerate(after)
                 if c["cell_type"] == "code" and SETUP_MARKER in "".join(c["source"])]
    new_bundle = where_in(after, CELL_MARKER)
    if len(new_setup) != 1 or len(new_bundle) != 1 or new_bundle[0] > new_setup[0]:
        raise SystemExit("the notebook lost a cell: setup at %s, bundle at %s"
                         % (new_setup, new_bundle))
    return new_bundle[0], len(after), new_setup[0]


def where_in(cells, marker):
    return [i for i, c in enumerate(cells)
            if c["cell_type"] == "code" and marker in "".join(c["source"])]


def build_for(nb_path) -> int:
    archive = build_archive()
    source = bundle_cell(archive, repo_commit())
    compile(source, "bundle cell", "exec")
    position, ncells, setup_at = inject(source, nb_path)
    print("%s: bundle cell at index %d of %d, ahead of the setup cell at %d | archive %.0f KB,"
          " cell %.0f KB" % (nb_path.name, position, ncells, setup_at, len(archive) / 1024,
                             len(source) / 1024))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if a notebook's bundle no longer matches the repository")
    ap.add_argument("--notebook", default=None,
                    help="one notebook by filename (default: %s)" % SCALING_NB.name)
    ap.add_argument("--all", action="store_true",
                    help="do every notebook that carries a bundle")
    args = ap.parse_args()

    targets = NOTEBOOKS if args.all else [
        next((n for n in NOTEBOOKS if n.name == args.notebook), REPO / "colab" /
             (args.notebook or SCALING_NB.name))]
    targets = [t for t in targets if t.exists()]
    if not targets:
        raise SystemExit("no such notebook under colab/: %s" % args.notebook)

    if args.check:
        rc = 0
        for t in targets:
            print("-- %s" % t.name)
            rc |= check(t)
        return rc

    for t in targets:
        build_for(t)
    print("verify with: python -m pytest tests/test_colab_notebook.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
