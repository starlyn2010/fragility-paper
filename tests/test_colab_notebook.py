"""Contract tests for colab/c4_endgame_scaling.ipynb's setup cell.

The notebook is the harness for the 10/12/14-empty-cell scaling arms, and its setup cell
has to do two things in a machine with no terminal, no browser and no clipboard: show the
user the GitHub one-time code, and get a copy of a private repository into the runtime.
Both broke once each on a real runtime, and both are testable without one:

1. every code cell parses, with IPython magics treated as statements rather than Python;
2. the device-flow helper strips the ANSI cursor moves that Colab's output frame chokes
   on, sends the Enter that starts gh's polling, sets BROWSER to something that cannot
   fail, and still shows the one-time code;
3. the archive fallback finds the repository whichever layout the zip uses, because
   `git archive` stores paths flat from the root while GitHub's Download ZIP wraps them
   in one folder --- and an archive that is not the repository is refused rather than
   half-trusted.

A no-op stripper and the old single-root lookup are both asserted to fail, so these
checks have teeth: breaking the cell breaks the test.

Run with pytest, or directly: python tests/test_colab_notebook.py
"""
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
import zipfile

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB_PATH = REPO / "colab" / "c4_endgame_scaling.ipynb"
MARKER = pathlib.Path("src") / "c4_endgame.py"
ESC = chr(27)
SETUP_TAG = "def bootstrap("
BUNDLE_TAG = "BUNDLE_SHA256 = "

# what a terminal program can put on the wire: cursor/erase/SGR/charset/mode forms
ESCAPE_FORMS = [ESC + "[?25l", ESC + "[?25h", ESC + "[0m", ESC + "[1m", ESC + "[7m",
                ESC + "[4m", ESC + "[24m", ESC + "[38;5;208m", ESC + "[K", ESC + "[>4;2m",
                ESC + "[?1049h", ESC + "(B", ESC + "[s", ESC + "[u",
                ESC + "=", ESC + ">", ESC + "<"]

GH_OUTPUT = (ESC + "[?25l" + ESC + "[0m! Failed to copy one-time code to clipboard\n"
             + ESC + "[1m! First copy your one-time code: " + ESC + "[7m6D57-DF15"
             + ESC + "[0m\nOpen this URL to continue in your web browser: "
             "https://github.com/login/device" + ESC + "[0m\n"
             + ESC + "[?25hWaiting for your authorization" + ESC + "[K...\n")

STREAM = [GH_OUTPUT]


def _cells():
    return json.loads(NB_PATH.read_text(encoding="utf-8"))["cells"]


def _code_cells():
    return [(i, "".join(c["source"])) for i, c in enumerate(_cells())
            if c["cell_type"] == "code"]


def _cell(tag):
    """Fetch a cell by what it defines, not by its index: the bundle moved when it landed."""
    for i, src in _code_cells():
        if tag in src:
            return src
    raise AssertionError("no code cell contains %r" % tag)


def _cell_source():
    return _cell(SETUP_TAG)


def _bundle_source():
    return _cell(BUNDLE_TAG)


def _exec_setup(source=None):
    """Run the setup cell up to the line that clones, and hand back its namespace."""
    src = _cell_source() if source is None else source
    ns = {}
    exec(src[: src.index("REPO = bootstrap()")], ns)
    return ns


class _FakeStdin:
    def __init__(self):
        self.written = []

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass


class _FakeProc:
    """A stand-in for gh that speaks the way gh speaks on a headless machine."""

    live = []

    def __init__(self, argv, **kw):
        _FakeProc.live.append(self)
        self.argv, self.kw = argv, kw
        self.stdin = _FakeStdin()
        self.stdout = io.StringIO(STREAM[0])

    def wait(self):
        return 0


def _run_device_login(source=None):
    ns = _exec_setup(source)
    ns["subprocess"] = type("fake_subprocess", (), {
        "Popen": _FakeProc, "PIPE": "PIPE", "STDOUT": "STDOUT"})
    ns["_gh_ready"] = lambda: True
    _FakeProc.live = []
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ok = ns["_gh_device_login"]()
    return ok, out.getvalue().splitlines(), _FakeProc.live[0]


def test_every_code_cell_parses():
    for i, cell in enumerate(_cells()):
        if cell["cell_type"] != "code":
            continue
        lines = ["" if l.startswith(("%", "!")) else l
                 for l in "".join(cell["source"]).split("\n")]
        compile("\n".join(lines), "cell %d" % i, "exec")


def test_device_login_shows_the_code_as_plain_text():
    STREAM[0] = GH_OUTPUT
    ok, printed, proc = _run_device_login()
    assert ok is True
    assert proc.argv == ["gh", "auth", "login", "--hostname", "github.com",
                         "--git-protocol", "https", "--web"]
    assert proc.kw["env"]["BROWSER"] == "echo", "there is no browser in a runtime"
    assert proc.kw["stdin"] == "PIPE" and proc.kw["stderr"] == "STDOUT"
    assert proc.kw["bufsize"] == 1, "line-buffered or the code arrives too late"
    assert proc.stdin.written == [chr(10)], "Enter is what starts the polling"
    assert any("6D57-DF15" in l for l in printed), printed
    assert all(ESC not in l for l in printed), printed
    assert not any(re.search(r"\[[0-9;?]*[mHK]", l) for l in printed), printed


def test_the_cells_own_cleaner_handles_every_escape_form():
    STREAM[0] = "".join(f + "x" for f in ESCAPE_FORMS) + "\n"
    _, printed, _ = _run_device_login()
    assert printed[1:] == ["x" * len(ESCAPE_FORMS)], printed
    STREAM[0] = "plain text, nothing to strip\n"
    assert _run_device_login()[1][1:] == ["plain text, nothing to strip"]


def test_a_no_op_stripper_would_leak_cursor_junk():
    neutered = _cell_source().replace('ansi.sub("", raw)', "raw")
    assert neutered != _cell_source()
    STREAM[0] = GH_OUTPUT
    _, printed, _ = _run_device_login(neutered)
    assert any(re.search(r"\[[0-9;?]*m", l) for l in printed), printed


def _zip_case(entries, name="fragility-paper.zip", ns=None):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb_zip_"))
    try:
        with zipfile.ZipFile(tmp / name, "w") as fh:
            for k, v in entries.items():
                fh.writestr(k, v)
        ns = _exec_setup() if ns is None else ns
        ns["CONTENT"] = tmp
        got = ns["_unzip_repo"]()
        return None if got is None else got.relative_to(tmp).as_posix()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_flat_and_wrapped_archives_both_resolve():
    flat = {".dockerignore": "x", str(MARKER): "code", "data/any.json": "{}"}
    wrapped = {"fragility-paper-main/README.md": "r",
               "fragility-paper-main/" + str(MARKER): "code"}
    assert _zip_case(flat) == "."
    assert _zip_case(wrapped) == "fragility-paper-main"
    assert _zip_case({"README.md": "not the repo"}) is None


def test_the_old_single_root_lookup_would_have_failed_the_flat_archive():
    src = _cell_source()
    old = src.replace(
        '        tops = sorted({n.split("/")[0] for n in names if "/" in n})\n'
        '        roots = [z.parent] + [z.parent / t for t in tops]',
        "        roots = [z.parent / names[0].split('/')[0]]")
    assert old != src, "the cleaner lookup being tested is not the one the test mutates"
    assert _zip_case({".dockerignore": "x", str(MARKER): "code"},
                     ns=_exec_setup(old)) is None


def test_the_bundle_cell_runs_before_the_setup_cell():
    positions = {}
    for tag, name in (("bundle", BUNDLE_TAG), ("setup", SETUP_TAG)):
        positions[tag] = next(i for i, src in _code_cells() if name in src)
    assert positions["bundle"] < positions["setup"], positions


def test_the_bundle_still_matches_the_repository():
    ns = {}
    exec(_bundle_source()[: _bundle_source().index("_archive =")], ns)
    for rel, digest in ns["BUNDLE_FILES"].items():
        path = REPO / rel
        assert path.exists(), "%s is gone from the repo; rebuild the bundle" % rel
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (
            "%s changed since the bundle was built - run python colab/build_bundle.py" % rel)
    out = subprocess.run([sys.executable, str(REPO / "colab" / "build_bundle.py"), "--check"],
                         capture_output=True, text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stdout + out.stderr


def _boot(tmp, bundle=None):
    """Execute the bundle cell and the setup cell's bootstrap in a fake /content.

    subprocess is replaced by something that raises, so a boot that reaches for gh or for
    git fails loudly instead of quietly cloning. Returns the namespace it ran in.
    """
    bundle_src = _bundle_source() if bundle is None else bundle
    setup_src = _cell_source()
    head = setup_src[: setup_src.index("REPO = bootstrap()")]
    call = setup_src[setup_src.index("REPO = bootstrap()"):
                     setup_src.index("print('repo at'")]

    class Exploding:
        PIPE = "PIPE"
        STDOUT = "STDOUT"

        def run(self, *a, **kw):
            raise AssertionError("the bundle boot reached for a subprocess: %r" % (a,))

        def Popen(self, *a, **kw):
            raise AssertionError("the bundle boot reached for a subprocess")

        def which(self, name):
            raise AssertionError("the bundle boot looked for an executable")

    old = os.environ.get("COLAB_CONTENT")
    os.environ["COLAB_CONTENT"] = str(tmp)
    ns = {}
    cwd = os.getcwd()
    try:
        # a runtime starts in /content, and _find_repo() also looks at the working
        # directory: without this the laptop's own checkout would satisfy the boot
        os.chdir(tmp)
        exec(bundle_src + "\n\n" + head, ns)
        ns["subprocess"] = Exploding()
        exec(call, ns)
    finally:
        os.chdir(cwd)
        if old is None:
            os.environ.pop("COLAB_CONTENT", None)
        else:
            os.environ["COLAB_CONTENT"] = old
    return ns


def test_a_runtime_boots_from_the_bundle_alone():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb_boot_"))
    cwd, path = os.getcwd(), list(sys.path)
    try:
        ns = _boot(tmp)
        repo = ns["REPO"]
        assert repo == (tmp / "fragility-paper").resolve(), repo
        assert (repo / MARKER).exists()
        for rel in ns["BUNDLE_FILES"]:
            embedded = (repo / rel).read_bytes()
            assert embedded == (REPO / rel).read_bytes(), rel
        compile((repo / "src/c4_endgame.py").read_text(encoding="utf-8"), "bundled", "exec")
        assert "class Oracle" in (repo / "src/c4_game.py").read_text(encoding="utf-8")
        assert json.loads((repo / "data/c4_end10_s1.json").read_text(encoding="utf-8"))
    finally:
        os.chdir(cwd)
        sys.path[:] = path
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_modified_bundle_is_refused_and_the_boot_falls_through():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb_tamper_"))
    cwd, path = os.getcwd(), list(sys.path)
    try:
        src = _bundle_source()
        cut = src.index("BUNDLE_B64 = (") + 40
        tampered = src[:cut] + ("A" if src[cut] != "A" else "B") + src[cut + 1:]
        assert tampered != src
        try:
            _boot(tmp, bundle=tampered)
        except AssertionError as exc:
            assert "corrupt" in str(exc) or "subprocess" in str(exc), exc
        else:
            raise AssertionError("a tampered bundle booted anyway")
    finally:
        os.chdir(cwd)
        sys.path[:] = path
        shutil.rmtree(tmp, ignore_errors=True)


def test_plain_clone_never_asks_for_a_username():
    src = _cell_source()
    assert "'GIT_TERMINAL_PROMPT': '0'" in src
    assert src.index("def _unzip_repo") < src.index("def _clone")


if __name__ == "__main__":
    names = sorted(n for n in globals() if n.startswith("test_"))
    failed = []
    for n in names:
        try:
            globals()[n]()
            print("ok    ", n)
        except Exception as exc:
            failed.append(n)
            print("FAIL  ", n, type(exc).__name__, exc)
    print("%d passed, %d failed" % (len(names) - len(failed), len(failed)))
    raise SystemExit(1 if failed else 0)
