"""Contract tests for colab/c4_endgame_scaling.ipynb's setup cell.

The notebook is the harness for the 10/12/14-empty-cell scaling arms, and in a machine
with no terminal, no browser and no clipboard its setup cell has to show the user the
GitHub one-time code and get a copy of a private repository into the runtime. Each of the
five promises below broke at least once while the notebook was being built, and all five
are testable without a runtime:

1. every code cell parses, with IPython magics treated as statements rather than Python;
2. the device-flow helper strips the ANSI cursor moves that Colab's output frame chokes
   on, sends the Enter that starts gh's polling, sets BROWSER to something that cannot
   fail, and still shows the one-time code;
3. the archive fallback finds the repository whichever layout the zip uses, because
   `git archive` stores paths flat from the root while GitHub's Download ZIP wraps them
   in one folder --- and an archive that is not the repository is refused rather than
   half-trusted;
4. the cells that sample endgames in the kernel import the game the same way the scripts
   do, because `src/c4_endgame.py` imports its sibling by the flat name `c4_game`: a cell
   that instead imports `src.c4_game` gets a second copy of the module, configures that
   copy to 6x7, and hands the sampler a 6x7 oracle pointed at 4x4 boards;
5. the queue cell makes `runs/` before it opens the log of the first arm, because the
   working copy the notebook builds from its own bundle contains `src/` and `data/` and
   nothing else --- only a clone would already have `runs/`, from the tracked drivers;
6. every code cell runs, in document order, on a bare runtime whose only source of code is
   the bundle: the arms launch in dependency order, the hands-free pump returns by itself,
   and the results cell reads fields the scripts really write.

A no-op stripper, the old single-root lookup, the package-qualified import, the missing
`mkdir` and the verification line that printed `None` are each asserted to fail, so these
checks have teeth: breaking a cell breaks the test.

Run with pytest, or directly: python tests/test_colab_notebook.py
"""
import builtins
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
import time
import zipfile

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB_PATH = REPO / "colab" / "c4_endgame_scaling.ipynb"
# the notebooks colab/build_followup_notebooks.py and colab/build_reads_notebooks.py write;
# they carry the same bundle and the same tested cells, so the structural promises below are
# checked on all three
FOLLOWUPS = [REPO / "colab" / "c4_ladder_finish.ipynb",
             REPO / "colab" / "c4_finish_all.ipynb"]
ALL_NBS = [NB_PATH] + FOLLOWUPS
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


def _cells(nb=NB_PATH):
    return json.loads(nb.read_text(encoding="utf-8"))["cells"]


def _code_cells(nb=NB_PATH):
    return [(i, "".join(c["source"])) for i, c in enumerate(_cells(nb))
            if c["cell_type"] == "code"]


def _cell(tag, nb=NB_PATH):
    """Fetch a cell by what it defines, not by its index: the bundle moved when it landed."""
    for i, src in _code_cells(nb):
        if tag in src:
            return src
    raise AssertionError("no code cell of %s contains %r" % (nb.name, tag))


def _cell_source(nb=NB_PATH):
    return _cell(SETUP_TAG, nb)


def _bundle_source(nb=NB_PATH):
    return _cell(BUNDLE_TAG, nb)


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


def _runnable(src):
    """A code cell as Python: the IPython magics become blank statements.

    `%cd` and `!python` are kernel sugar and are not what the queue does, so dropping them
    both lets the parse test compile a cell and lets the run-all test below execute it.
    """
    return "\n".join("" if l.startswith(("%", "!")) else l for l in src.split("\n"))


def _present():
    """The notebooks that exist. Every structural promise below is checked on each of them."""
    return [n for n in ALL_NBS if n.exists()]


def test_every_notebook_that_should_exist_does():
    missing = [n.name for n in ALL_NBS if not n.exists()]
    assert not missing, ("these notebooks are part of the harness: %r - rebuild with "
                         "python colab/build_followup_notebooks.py" % (missing,))
    assert len(_present()) == 3


def test_the_generated_notebooks_match_their_generator():
    """A notebook edited by hand in the browser is not the notebook the repository builds.

    The generated notebooks are written out of cells this repository already tested, so a
    divergence means somebody changed one copy and not the other, and the paper would then cite a
    document the code cannot produce. Two generators write them, so both are asked.
    """
    for script in ("build_followup_notebooks.py", "build_reads_notebooks.py"):
        out = subprocess.run([sys.executable, str(REPO / "colab" / script), "--check"],
                             capture_output=True, text=True, cwd=str(REPO))
        assert out.returncode == 0, "%s --check: %s%s" % (script, out.stdout, out.stderr)


def test_every_code_cell_parses():
    for nb in _present():
        for i, cell in enumerate(_cells(nb)):
            if cell["cell_type"] != "code":
                continue
            compile(_runnable("".join(cell["source"])), "%s cell %d" % (nb.name, i), "exec")


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
    for nb in _present():
        positions = {}
        for tag, name in (("bundle", BUNDLE_TAG), ("setup", SETUP_TAG)):
            positions[tag] = next(i for i, src in _code_cells(nb) if name in src)
        assert positions["bundle"] < positions["setup"], (nb.name, positions)


def test_the_bundle_still_matches_the_repository():
    for nb in _present():
        src = _bundle_source(nb)
        ns = {}
        exec(src[: src.index("_archive =")], ns)
        for rel, digest in ns["BUNDLE_FILES"].items():
            path = REPO / rel
            assert path.exists(), "%s is gone from the repo; rebuild the bundle" % rel
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (
                "%s changed since %s was bundled - run python colab/build_bundle.py --all"
                % (rel, nb.name))
    out = subprocess.run([sys.executable, str(REPO / "colab" / "build_bundle.py"),
                          "--check", "--all"], capture_output=True, text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stdout + out.stderr


def _boot(tmp, bundle=None, nb=NB_PATH):
    """Execute the bundle cell and the setup cell's bootstrap in a fake /content.

    subprocess is replaced by something that raises, so a boot that reaches for gh or for
    git fails loudly instead of quietly cloning. Returns the namespace it ran in.
    """
    bundle_src = _bundle_source(nb) if bundle is None else bundle
    setup_src = _cell_source(nb)
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
    for nb in _present():
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb_boot_"))
        cwd, path = os.getcwd(), list(sys.path)
        try:
            ns = _boot(tmp, nb=nb)
            repo = ns["REPO"]
            assert repo == (tmp / "fragility-paper").resolve(), (nb.name, repo)
            assert (repo / MARKER).exists()
            for rel in ns["BUNDLE_FILES"]:
                embedded = (repo / rel).read_bytes()
                assert embedded == (REPO / rel).read_bytes(), (nb.name, rel)
            compile((repo / "src/c4_endgame.py").read_text(encoding="utf-8"),
                    "bundled", "exec")
            assert "class Oracle" in (repo / "src/c4_game.py").read_text(encoding="utf-8")
            assert json.loads((repo / "data/c4_end10_s1.json").read_text(encoding="utf-8"))
        finally:
            os.chdir(cwd)
            sys.path[:] = path
            shutil.rmtree(tmp, ignore_errors=True)


def test_a_modified_bundle_is_refused_and_the_boot_falls_through():
    for nb in _present():
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb_tamper_"))
        cwd, path = os.getcwd(), list(sys.path)
        try:
            src = _bundle_source(nb)
            cut = src.index("BUNDLE_B64 = (") + 40
            tampered = src[:cut] + ("A" if src[cut] != "A" else "B") + src[cut + 1:]
            assert tampered != src
            try:
                _boot(tmp, bundle=tampered, nb=nb)
            except AssertionError as exc:
                assert "corrupt" in str(exc) or "subprocess" in str(exc), exc
            else:
                raise AssertionError("%s booted a tampered bundle anyway" % nb.name)
        finally:
            os.chdir(cwd)
            sys.path[:] = path
            shutil.rmtree(tmp, ignore_errors=True)


def test_plain_clone_never_asks_for_a_username():
    src = _cell_source()
    assert "'GIT_TERMINAL_PROMPT': '0'" in src
    assert src.index("def _unzip_repo") < src.index("def _clone")


PKG_IMPORT = re.compile(r"^\s*(?:from|import)\s+src\.")


def _package_imports(cells):
    """(cell index, line) for every cell that reaches into the repo as a package.

    `src/c4_endgame.py` does `import c4_game`, so `src.c4_game` is a second module object
    over the same file. A cell that configures the second one leaves the sampler building
    boards for the first: 4x4 boards handed to a 6x7 solver.
    """
    return [(i, line.strip()) for i, src in cells for line in src.split("\n")
            if PKG_IMPORT.match(line)]


def test_no_cell_imports_the_code_through_the_package_name():
    for nb in _present():
        hits = _package_imports(_code_cells(nb))
        assert hits == [], ("these cells would double-load c4_game and configure the wrong "
                            "copy: %r" % (hits,))
        mutant = [(i, src.replace("import c4_game as g", "import src.c4_game as g"))
                  for i, src in _code_cells(nb)]
        assert _package_imports(mutant), (
            "%s has no flat c4_game import for the check to catch, so nothing is being "
            "tested" % nb.name)


def test_the_setup_cell_adds_src_to_the_path_not_just_the_root():
    for nb in _present():
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb_srcpath_"))
        cwd, path = os.getcwd(), list(sys.path)
        try:
            ns = _boot(tmp, nb=nb)
            repo = ns["REPO"]
            assert str(repo) in ns["sys"].path, repo
            assert str((repo / "src").resolve()) in ns["sys"].path, (
                "the flat imports the cells use cannot resolve without src on sys.path")
        finally:
            os.chdir(cwd)
            sys.path[:] = path
            shutil.rmtree(tmp, ignore_errors=True)


_FLAT_SAMPLE = """
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
import c4_game as g
import c4_endgame
from c4_endgame import random_games
g.configure(6, 7, 4)
assert c4_endgame.g is g, 'the sampler runs on a different module copy'
roots, games = random_games(g.Oracle(), 6, 2, 7, 0, max_games=400)
print('ok', len(roots), 'roots from', games, 'games')
"""

_PACKAGE_SAMPLE = (_FLAT_SAMPLE
                   .replace("import c4_game as g", "import src.c4_game as g")
                   .replace("import c4_endgame\n", "")
                   .replace("assert c4_endgame.g is g, 'the sampler runs on a different "
                            "module copy'\n", "")
                   .replace("from c4_endgame import random_games",
                            "from src.c4_endgame import random_games"))


def _run(script):
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          cwd=str(REPO), timeout=300)


def test_the_flat_import_samples_6x7_endgames():
    out = _run(_FLAT_SAMPLE)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("ok"), out.stdout


def test_the_package_import_is_the_index_error_the_runtime_raised():
    out = _run(_PACKAGE_SAMPLE)
    assert out.returncode != 0, ("the package import solved endgames after all, so the "
                                 "flat-import rule above guards nothing")
    assert "IndexError" in out.stderr, out.stderr[-500:]


QUEUE_TAG = "def build(empties,"
TICK_TAG = "def tick():"
MKDIR = "        pathlib.Path(j['log']).parent.mkdir(parents=True, exist_ok=True)\n"


class _NoLaunch:
    """Records the launches `tick()` would have made instead of making them."""

    PIPE = "PIPE"
    STDOUT = "STDOUT"
    DEVNULL = "DEVNULL"

    def __init__(self):
        self.calls = []

    def Popen(self, cmd, **kw):
        self.calls.append((cmd, kw))

        class Done:
            @staticmethod
            def poll():
                return None

        return Done()

    def run(self, *a, **kw):
        raise AssertionError("tick() should not shell out while its jobs are recorded")


def _tick_definition(src):
    """The queue cell's two helpers, without the bare `tick()` call it ends with.

    Slicing on `src.index('tick()')` would cut inside `def tick():` instead, so the
    trailing call is dropped line by line.
    """
    lines = src[src.index("def pids("):].split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    assert lines[-1] == "tick()", lines[-3:]
    return "\n".join(lines[:-1])


def _launch_queue(tmp, tick_src=None):
    """Boot the bundle into a fake /content, build the 12-empty queue, run one tick().

    The point of booting from the bundle is that its working copy has no `runs/`
    directory: the bundle ships `src/` and `data/` only, so the queue cell's own log
    files are the first thing that has to appear there.
    """
    ns = _boot(tmp)
    repo = ns["REPO"]
    assert not (repo / "runs").exists(), (
        "the booted copy already has runs/, so this test no longer proves the mkdir")
    env = {"REPO": repo, "os": os, "json": json, "pathlib": pathlib,
           "MAX_PAR": 1, "THREADS": 1, "snapshot": lambda e: None}
    queue_src = _cell(QUEUE_TAG)
    exec(queue_src[queue_src.index("SEEDS = ("):], env)
    exec(_tick_definition(_cell(TICK_TAG) if tick_src is None else tick_src), env)
    env["pids"] = lambda needle: []
    env["subprocess"] = launched = _NoLaunch()
    cwd = os.getcwd()
    try:
        os.chdir(repo)
        env["tick"]()
    finally:
        os.chdir(cwd)
    return env, launched


def test_the_queue_makes_runs_before_it_opens_a_log():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb_runs_"))
    try:
        env, launched = _launch_queue(tmp)
        repo = env["REPO"]
        assert (repo / "runs").is_dir(), "tick() opened its log without making runs/"
        assert len(launched.calls) == 1, launched.calls
        cmd, kw = launched.calls[0]
        assert kw["cwd"] == str(repo), kw
        assert pathlib.Path(repo / env["JOBS"][0]["log"]).exists(), env["JOBS"][0]["log"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_without_the_mkdir_the_queue_dies_the_way_it_died_in_colab():
    src = _cell(TICK_TAG).replace(MKDIR, "")
    assert src != _cell(TICK_TAG), "the mkdir vanished from the cell; nothing to mutate"
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb_runs_mutant_"))
    try:
        _launch_queue(tmp, tick_src=src)
    except FileNotFoundError as exc:
        assert "runs/" in str(exc), exc
    else:
        raise AssertionError("the queue launched with no runs/ directory, so the "
                             "test above would pass without the mkdir")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


PROBE_TAG = "def probe(empties"
RESULTS_TAG = "a = snapshot(E)"           # the scaling notebook's final line
RESULTS_TAGS = (RESULTS_TAG, "for a in snapshot()")
PUMP_TAG = "MAX_MINUTES = "


def _results_cell(nb):
    """The cell that prints the table, whichever shape the final download line has."""
    for i, src in _code_cells(nb):
        if any(t in src for t in RESULTS_TAGS) and "verify" in src:
            return src
    raise AssertionError("%s has no results cell (looked for %r)" % (nb.name, RESULTS_TAGS))


def _queue_cell(nb):
    """The cell that builds the commands, whichever name its functions have.

    The documents grew their queue cells at different times, so the one thing they do share is
    that they name the script each arm runs - which is exactly what the flag test below needs,
    and a tag as specific as `def build(empties,` would stop matching the moment a cell is
    reshaped.
    """
    for i, src in _code_cells(nb):
        if QUEUE_TAG in src or "'src/c4_endgame.py'" in src:
            return src
    raise AssertionError("%s has no queue cell that names src/c4_endgame.py" % nb.name)


class _StubSubprocess:
    """`subprocess` as the cells see it: `Popen` recorded, `pgrep` answered, rest real.

    The launch cell ends by starting arms, and the machine running this test is running the
    researcher's own training job on its two cores, so a test that executes that cell as
    written must not reach a real `Popen`. The swap happens in `sys.modules`, not in the
    namespace, because `import subprocess` inside a cell rebinds the real module over
    whatever was merely placed there.
    """

    PIPE = subprocess.PIPE
    STDOUT = subprocess.STDOUT
    DEVNULL = subprocess.DEVNULL

    def __init__(self):
        self.launches = []

    def Popen(self, cmd, **kw):
        fh = kw.get("stdout")
        if hasattr(fh, "close"):
            fh.close()               # an open log inside the temp repo blocks its deletion
            kw = dict(kw, stdout=fh.name)
        self.launches.append((list(cmd), kw))

        class Ended:
            @staticmethod
            def poll():
                return 0

        return Ended()

    def run(self, cmd, *a, **kw):
        if cmd and cmd[0] == "pgrep":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.run(cmd, *a, **kw)      # torch and friends still get a real one

    def __getattr__(self, name):
        return getattr(subprocess, name)


class _StubTime:
    """A clock that jumps an hour per reading, so the pump's five-hour guard is reached
    in milliseconds instead of minutes.

    Sleeping is refused after a bounded number of turns: if the pump's guard ever stops
    working, the cell would hold a runtime forever, and a test that hangs forever is worth
    nothing to anyone.
    """

    def __init__(self):
        self.now = 0.0
        self.sleeps = 0

    def time(self):
        self.now += 3600.0
        return self.now

    def sleep(self, _seconds):
        self.sleeps += 1
        if self.sleeps > 20:
            raise AssertionError("the pump never reached its guard; it would keep the "
                                 "runtime busy for the rest of the session")

    def __getattr__(self, name):
        return getattr(time, name)


def _run_cells(ns, stub, at_results=None):
    """Execute the notebook's code cells in document order, the way Cell -> Run all does.

    The bundle and setup cells are what `_boot` already ran with `subprocess` forbidden, so
    a boot that reached for `git` or `gh` fails that test instead of cloning here. The cost
    probe is skipped because it solves 12- and 14-empty-cell roots: that is the compute the
    notebook exists to spend once, and its import pattern is the flat-import test's business.
    Reordering the cells to break this order is exactly what the test should catch, so
    nothing here is reordered to make it pass.
    """
    ran, skipped = [], []
    for i, src in _code_cells():
        if any(tag in src for tag in (BUNDLE_TAG, SETUP_TAG, PROBE_TAG)):
            skipped.append(i)
            continue
        if TICK_TAG in src:
            sys.modules["subprocess"] = stub
        if PUMP_TAG in src:
            ns["time"] = _StubTime()
        if RESULTS_TAG in src and at_results is not None:
            at_results()
        exec(_runnable(src), ns)
        ran.append(i)
    return ran, skipped


def test_run_all_works_on_a_bare_runtime_that_only_has_the_bundle():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nb_runall_"))
    cwd, path, sub = os.getcwd(), list(sys.path), sys.modules["subprocess"]
    old_env = os.environ.get("COLAB_CONTENT")
    out = io.StringIO()

    def plant_arms(repo):
        # The arms are still in flight at this point in the document, so feed the results
        # cell real archived outputs (a repair arm, a standalone controls file, a
        # verification file) under the names this depth uses.
        for src_rel, dst in (("data/c4_end10_s0.json", "data/c4_end12_s0.json"),
                             ("data/c4_end_ctrl_s0.json", "data/c4_end12_s0_all.json"),
                             ("data/c4_end10_verify_s0.json", "data/c4_end12_verify_s0.json")):
            shutil.copyfile(REPO / src_rel, repo / dst)

    try:
        os.environ["COLAB_CONTENT"] = str(tmp)
        os.chdir(tmp)
        ns = _boot(tmp)                       # a runtime with no terminal and no repo
        repo = ns["REPO"]
        # cell 12 samples a 12-empty suite, which is oracle work on the order of an hour on
        # these cores; handing it the suite file it would have written takes the reuse branch
        # and leaves the sampling branch to the flat-import subprocess tests above.
        shutil.copyfile(REPO / "data/c4_endgame_roots_10_40_s0.json",
                        repo / "data/c4_endgame_roots_12_40_s0.json")
        stub = _StubSubprocess()
        with contextlib.redirect_stdout(out):
            ran, skipped = _run_cells(ns, stub, lambda: plant_arms(repo))
        text = out.getvalue()

        assert ran == sorted(ran), "the cells did not run in document order"
        assert set(ran) | set(skipped) == {i for i, _ in _code_cells()}, (ran, skipped)
        assert len(skipped) == 3, skipped
        assert "reusing 40 roots" in text, text[:400]

        repairs = ["data/c4_end12_s%d.json" % s for s in (0, 1, 2)]
        started = [c[c.index("--out") + 1] for c, _ in stub.launches]
        assert set(started) <= set(repairs), (
            "the controls and the verifications wait for a repair JSON on disk; launching "
            "them first would run them on the wrong policy: %r" % (sorted(set(started)),))
        cap = ns["MAX_PAR"]
        assert cap <= ns["CORES"] and (ns["RAM"] is not None or cap <= 2), (
            "with no way to read RAM the queue must hold at two arms, because an "
            "out-of-memory kill loses the run: %r" % (ns,))
        # with pgrep answering "nothing is running", every tick starts the same first wave
        # and stops at the cap - the cap holding, and the arms beyond it waiting for a slot
        # rather than oversubscribing the runtime.
        assert started[:min(cap, len(repairs))] == repairs[:min(cap, len(repairs))], (
            started, cap)
        assert all("--no-controls" in c for c, _ in stub.launches), started
        cmd, kw = stub.launches[0]
        assert cmd[:2] == ["python", "src/c4_endgame.py"], cmd
        assert cmd[cmd.index("--roots-file") + 1] == "data/c4_endgame_roots_12_40_s0.json"
        assert cmd[cmd.index("--empties") + 1] == "12"
        assert kw["cwd"] == str(repo) and kw["stdin"] == subprocess.DEVNULL
        assert kw["stdout"].endswith("_colab_c4_end12_s0.log"), kw
        assert (repo / "runs").is_dir(), "the queue never made its log folder"

        assert "run this cell again" in text, "the hands-free pump did not return by itself"
        assert "seed 0: W" in text and "control all : W" in text, text[-900:]
        assert "verify : W" in text and "None" not in text, text[-900:]
        assert (tmp / "c4_end12_results.zip").exists(), "the results cell shipped nothing"
    finally:
        sys.modules["subprocess"] = sub
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None)
            if f and _under(pathlib.Path(f).resolve(), tmp):
                del sys.modules[name]
        sys.path[:] = path
        os.chdir(cwd)
        if old_env is None:
            os.environ.pop("COLAB_CONTENT", None)
        else:
            os.environ["COLAB_CONTENT"] = old_env
        shutil.rmtree(tmp, ignore_errors=True)


def _under(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


RESOURCE_TAG = "def quota_cores()"
UNGUARDED_AVAIL = ("def avail_gb():\n"
                   "    for line in open('/proc/meminfo'):\n"
                   "        if line.startswith('MemAvailable'):\n"
                   "            return int(line.split()[1]) / 2 ** 20\n")


def _no_proc(fn):
    """Run `fn` with /proc made to not exist, the way a Mac or a Windows box sees it."""
    real_open = builtins.open

    def no_proc(path, *a, **kw):
        if str(path).startswith("/proc"):
            raise FileNotFoundError(2, "No such file or directory", str(path))
        return real_open(path, *a, **kw)

    try:
        builtins.open = no_proc
        return fn()
    finally:
        builtins.open = real_open


def test_the_resource_cell_works_where_proc_does_not_exist():
    """`/proc/meminfo` is a Linux file, and a reviewer can open this notebook on a Mac.

    Colab is Linux, so only a local Jupyter takes that branch, and the cell used to die
    there before section 3 could say how many arms to run. The same cell with its guard
    removed is asserted to raise, so what is being tested is the fix rather than the machine
    running the suite. The cell is copied into the two generated notebooks, so every notebook
    that queues arms is checked rather than only the first one.
    """
    for nb in _present():
        src = _cell(RESOURCE_TAG, nb)
        cut = src.index("def avail_gb"), src.index("CORES, RAM")
        mutant = src[:cut[0]] + UNGUARDED_AVAIL + "\n" + src[cut[1]:]
        assert mutant != src, ("%s has no guard left in its capacity cell, so there is "
                               "nothing this test can prove about it" % nb.name)

        def run(src=src, mutant=mutant, nb=nb):
            ns = {}
            exec(_runnable(src), ns)
            assert ns["RAM"] is None, ns
            assert 1 <= ns["MAX_PAR"] <= min(ns["CORES"], 2), (
                "with no RAM reading the queue must hold at two arms, not oversubscribe: %r"
                % ns)
            try:
                exec(_runnable(mutant), {})
            except FileNotFoundError:
                return True
            raise AssertionError("%s: the unguarded cell ran on a machine with no /proc, so "
                                 "the guard above protects nothing" % nb.name)

        _no_proc(run)


def _dangling_markdown(cells):
    """Markdown cells whose prose stops before the sentence does.

    A reviewer reads this prose as the procedure, so `...hands every finished JSON and
    checkpoint to the` is not a step of anything. Two cells of this notebook were cut like
    that by an edit and nobody noticed until the upload audit.
    """
    out = []
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"]).rstrip()
        if text and not re.search(r"[.!?:\u201d\u2019]$", text):
            out.append((i, text[-58:]))
    return out


def test_no_markdown_cell_stops_mid_sentence():
    for nb in _present():
        assert _dangling_markdown(_cells(nb)) == [], (nb.name, _dangling_markdown(_cells(nb)))
        cut = [dict(c, source=("".join(c["source"]).rsplit(". ", 1)[0] if i == 0
                               else "".join(c["source"])))
               for i, c in enumerate(_cells(nb))]
        hit = _dangling_markdown(cut)
        assert len(hit) == 1 and hit[0][0] == 0, (
            "%s: the detector does not see the cut this notebook actually had: %r"
            % (nb.name, hit))


def test_every_flag_the_notebook_passes_is_declared_by_the_script_that_gets_it():
    """A mistyped flag does not fail until the arm has been running for an hour.

    The queue cell builds one command per arm and the scaling notebook leaves a commented
    bash version for the laptop; both are checked here against the argparse blocks they call,
    because `argparse` only refuses an unknown option when the process starts and the arms are
    hours long. The verify arm is checked against its own script, which declares a different
    set: `--orders` and `--ckpt` exist there and nowhere else. All three notebooks queue arms,
    so all three are checked.
    """
    def declared(rel):
        text = (REPO / rel).read_text(encoding="utf-8")
        return set(re.findall(r'add_argument\("(--[a-z-]+)"', text))

    def flags(text):
        return set(re.findall(r"['\"](--[a-z-]+)['\"]", text))

    def bash_flags(text):
        """The commented section-7 block spells its options bare, after `BASE=(`."""
        return set(re.findall(r"--[a-z][a-z-]*", text))

    endgame = declared("src/c4_endgame.py")
    verify_script = declared("src/c4_endgame_verify.py")

    for nb in _present():
        queue = _queue_cell(nb)
        ver = queue[queue.index("def verify(s"):]
        wrong_endgame = sorted(f for f in flags(queue.replace(ver, ""))
                               if f not in endgame)
        wrong_verify = sorted(f for f in flags(ver) if f not in verify_script)
        assert not wrong_endgame, "%s: src/c4_endgame.py declares no %r" % (
            nb.name, wrong_endgame)
        assert not wrong_verify, "%s: src/c4_endgame_verify.py declares no %r" % (
            nb.name, wrong_verify)
        assert "--orders" in flags(ver) and "--orders" not in endgame, (
            "%s: the verify arm stopped passing its own flag, so this test is reading the "
            "wrong cell" % nb.name)

        mutant = ver.replace("'--orders'", "'--order'")
        assert mutant != ver
        assert sorted(f for f in flags(mutant) if f not in verify_script) == ["--order"], (
            "%s: an invented flag would have passed every check, so nothing was tested"
            % nb.name)

        try:
            bash = _cell("%%bash", nb)
        except AssertionError:
            continue                       # only the scaling notebook keeps the laptop copy
        assert len(bash_flags(bash)) > 10, "the commented block no longer holds a command"
        leftover = sorted(f for f in bash_flags(bash) if f not in endgame)
        assert not leftover, "the commented bash block uses flags %s no script declares" % (
            leftover,)


def test_the_results_cell_reads_fields_the_verify_script_writes():
    """The results line for a verification arm used to ask that file for `summary`.

    `src/c4_endgame_verify.py` writes no such key, so the published table printed `None` for
    every seed. Reading the keys off the archived real verification JSON catches that for
    every field the cell touches, in every notebook that prints a table.
    """
    keys = set(json.loads((REPO / "data/c4_end10_verify_s0.json").read_text()))
    assert "summary" not in keys
    arm = json.loads((REPO / "data/c4_end12_s0.json").read_text())["repair"]
    real_repair, real_arm = set(arm["final"]), set(arm)
    for nb in _present():
        src = _results_cell(nb)
        got = set(re.findall(r"v\['([a-z_]+)'\]", src))
        assert got, "%s no longer reads a verification arm through v[...]" % nb.name
        assert got <= keys, "%s: the cell reads %r, the script writes %r" % (
            nb.name, sorted(got - keys), sorted(keys))
        fin = set(re.findall(r"fin\['([a-z_]+)'\]", src))
        assert fin, "%s: the cell no longer reads the repair's own final line" % nb.name
        assert fin <= real_repair, "%s: the cell asks the repair's final certificate for %r, " \
            "which the script does not write" % (nb.name, sorted(fin - real_repair))
        rd = set(re.findall(r"d\['repair'\]\['([a-z_]+)'\]", src))
        assert rd <= real_arm, "%s: the cell asks the repair block for %r, which it does not " \
            "hold" % (nb.name, sorted(rd - real_arm))


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
