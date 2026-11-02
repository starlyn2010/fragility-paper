"""Embed into colab/c4_endgame_scaling.ipynb the copy of the code it runs.

Why this exists: a Colab runtime that has to fetch a private repository needs either a
GitHub authorisation the human has to complete in a browser, a token pasted into a
notebook, or a file uploaded through the sidebar. All three are human work, and the first
fails silently when the one-time code expires. The queue only needs four modules and two
suites, roughly 200 KB, so the notebook carries them itself as a base64 zip and the setup
cell unpacks them into a working copy. Nothing is asked of the person running it.

The bundle is not trusted: it records the sha256 of every file it holds and of the archive
as a whole, and the setup cell re-computes both. A mismatch falls back to the clone rather
than running modified code, and tests/test_colab_notebook.py asserts the embedded hashes
still equal the repository's own files, so the bundle cannot drift away from the code the
paper cites without a test failing.

    python colab/build_bundle.py            # rebuild and inject
    python colab/build_bundle.py --check    # exit 1 if the notebook's bundle is stale
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
NB = REPO / "colab" / "c4_endgame_scaling.ipynb"
CELL_MARKER = "BUNDLE_SHA256 = "     # the assignment, which only the bundle cell holds
SETUP_MARKER = "def bootstrap("
B64_WIDTH = 100

# exactly what the queue imports or launches, plus the two suites the optional
# controls-from arms read; stored under their repo-relative paths
BUNDLE_FILES = [
    "src/c4_game.py",
    "src/c4_repair.py",
    "src/c4_endgame.py",
    "src/c4_endgame_verify.py",
    "data/c4_endgame_roots_10_40_s0.json",
    "data/c4_end10_s1.json",
]


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


def notebook_bundle_source() -> str:
    for c in json.loads(NB.read_text(encoding="utf-8"))["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and CELL_MARKER in src:
            return src
    return ""


def check() -> int:
    """Compare the hashes the notebook claims against the repository as it stands."""
    src = notebook_bundle_source()
    if not src:
        print("the notebook carries no bundle; run python colab/build_bundle.py")
        return 1
    ns = {}
    exec(src[: src.index("_archive =")], ns)
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


def inject(source: str) -> tuple:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the notebook's bundle no longer matches the repository")
    args = ap.parse_args()
    if args.check:
        return check()

    archive = build_archive()
    source = bundle_cell(archive, repo_commit())
    compile(source, "bundle cell", "exec")
    position, ncells, setup_at = inject(source)
    print("bundle cell at index %d of %d, ahead of the setup cell at %d | archive %.0f KB, "
          "cell %.0f KB" % (position, ncells, setup_at, len(archive) / 1024,
                            len(source) / 1024))
    print("verify with: python -m pytest tests/test_colab_notebook.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
