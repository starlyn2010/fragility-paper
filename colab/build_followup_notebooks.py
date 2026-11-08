"""Write the ladder-finisher notebook, so a runtime needs nothing but the notebook.

Why it exists: the paper's endgame scaling rests on three seeds per depth, repaired and
then certified against one 40-position suite. Three seeds give a 95% interval more than a
hand wide, and a certificate that is exact *conditional on the suite it was repaired
against* says nothing about the game. At twelve empty cells the seeds disagree - four of the
seven now archived certify, and three spend the whole 24-round cap and stop one, two and
four violating histories short - so what that depth reports is still 0.25-0.84. The gap
needed compute the two-core laptop cannot give it, so it was packed here as a queue a free
runtime could drain.

One notebook: `c4_ladder_finish.ipynb` - the nine arms the first ladder run did not deliver:
seeds 7 to 9 at twelve empty cells (3 to 6 arrived and are archived) and seeds 3 to 5 at ten and
eight, so twelve reaches ten seeds and the smaller depths reach six. It differs from every other
queue here in when it downloads: nothing is zipped or handed to the browser while an arm trains,
because that cost the arms wall time in the last run. A depth ships when its last arm lands, and a
`ship()` cell at the bottom sweeps whatever exists if the runtime is being taken away. Its nine
arms are archived, so this document is the record of how they were run rather than a queue anyone
still has to drain. What the curve needed past twelve empty cells - three arms at fourteen, and the
reads the paper owes - moved to `colab/c4_finish_all.ipynb`, written by
`colab/build_reads_notebooks.py`; the notebook that used to hold the fourteen-empty arms is
archived at `runs/_superseded_c4_endgame_14_holdout.ipynb`.

Both queues run the same three-arm triple per seed: repair against suite A, then have
`src/c4_endgame_verify.py` re-derive the certificate from the saved weights on suite A and
again on a blind suite B that the repair never saw. Nothing is asked of the person who opens
them: no sign-in, no token, no upload, no editing a cell. Run the cells in order and the
browser takes the zips.

The cells that are already contract-tested in `colab/c4_endgame_scaling.ipynb` - the setup
cell that unpacks the bundle, the machine-capacity cell, the monitor and the hands-free
pump - are copied from that notebook rather than rewritten, so a fix there lands here too.
`tests/test_colab_notebook.py` runs its promises over every notebook and
`tests/test_followup_notebooks.py` the ones specific to this one, including the budget
table below: a new seed is comparable with its own depth only under the numbers that depth
was published with, and those are read out of the archived arm JSONs rather than restated.

    python colab/build_followup_notebooks.py          # write it, then bundle it
    python colab/build_followup_notebooks.py --check  # exit 1 if a file no longer matches
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_bundle                                     # noqa: E402

REPO = build_bundle.REPO
SCALING = build_bundle.SCALING_NB
OUT_DIR = REPO / "colab"

# --- cells lifted from the tested scaling notebook ------------------------------
TAGS = {"setup": "def bootstrap(", "resource": "def quota_cores()",
        "monitor": "print(f'  wait  {j[\"name\"]}')", "pump": "MAX_MINUTES = ",
        "tick": "def tick():"}


def scaling_cells():
    return json.loads(SCALING.read_text(encoding="utf-8"))["cells"]


def scaling_cell(tag):
    """A cell of the scaling notebook by what it defines, never by its index."""
    for c in scaling_cells():
        src = "".join(c["source"])
        if c["cell_type"] == "code" and TAGS[tag] in src:
            return src
    raise SystemExit("the scaling notebook no longer has a %s cell; these notebooks are "
                     "built out of it, so nothing can be written" % tag)


# ----------------------------------------------------------------------------- prose ----
TITLE = """# Connect Four endgames: @@TITLE@@

**What this queue adds.** @@WHAT@@

**Why more seeds are not enough on their own.** Every published arm was repaired against a
40-position suite and then certified against that same suite. The certificate is exact over
all of the teacher's optimal tie-breaks, but it is conditional on those forty positions. So
each arm here is read a second time, by another script, from the saved weights: once on the
suite it was repaired against, and once on a blind suite sampled with different seeds. If
the certificate holds on the blind half, the repair generalised. If it does not, that is the
honest result and it goes in the paper as one.

**What you have to do.** Nothing but run the cells in order, then re-run the pump cell if
the runtime is taken away. @@HANDS@@ The notebook carries its own copy of the code and of
the published suites, so there is no GitHub sign-in, no token, nothing to upload and no
Drive mount. Every finished arm is handed to your browser inside a zip; keep every zip and
tell me where they land.
"""

S0 = """## 0. Nothing to set up

The cell titled *The bundle* holds a base64 zip of the four modules the arms run and every
published root suite, with the sha256 of each file beside it. The setup cell re-computes
those hashes and unpacks them into a working copy under `/content`. A hash that does not
match - because the bundle was edited inside the notebook, or because it is older than the
code - is refused, and the cell falls back to cloning the repository instead. Either way
the cells below need no further input.
"""

S1 = """## 1. How much machine do we actually have

Colab gives a free runtime two usable cores and about twelve gigabytes. An arm is one
single-threaded recursion over a 6x7 endgame plus a small GRU refit, so the queue holds two
arms and gives each of them one thread. This is the same capacity cell as the scaling
notebook, and nothing here is configured by hand.
"""

S2 = """## 2. Two suites: the one the repair sees, and the one it never sees

Suite A is the published one. It travels inside the bundle because every arm already in the
paper read `data/c4_endgame_roots_<E>_40_s0.json`, and seeds are only comparable when they
share it; a runtime that sampled its own suite would be answering a different question. At
fourteen empty cells no suite exists yet, so this one samples it and the sampled file
travels back in the results zip. That is the slow part of a new depth, and it is not stuck:
the cell prints one line per wave with the games it spent, and because the sample seeds are
fixed the positions it finds are the same ones the laptop would find.

Suite B is the holdout: sampled with seeds starting at 500 + (empty cells), so it is the
same suite on every runtime and for anyone who re-runs this notebook, and it is never given
to the repair. Both suites are grown in waves and checkpointed to a `.part` file, so a
runtime taken away in the middle of sampling costs the waves already printed and nothing
else.

Two properties are checked by the cell rather than claimed in prose:

* every position has exactly the depth's number of empty cells, is unfinished and has value
  0, which is what makes it an endgame of that depth rather than something else;
* the two suites share no position. That check carries the whole argument, and it is cheap:
  a repair trains on positions *below* its roots, which have strictly fewer empty cells, so
  a suite-B position of the same depth can only be inside the training data by being
  identical to a suite-A position. Board-level disjointness therefore means suite B was
  never shown to the learner.
"""

S3 = """## 3. The queue

Each seed is three arms: repair against suite A, then verify the saved weights on suite A,
then verify them on suite B. The two verifications wait for the repair's own JSON and its
checkpoint, so they can never read a policy that has not finished; the queue launches
whatever is unblocked, up to two arms at a time. The budgets are the published ones for
each depth - 200 epochs, 60 replay repeats, and the same round cap the seed-0 arm used - so
a new seed is comparable with the seeds already in the paper.
"""

S4 = """## 4. Launch

`tick()` starts what it can@@PUSH_CLAUSE@@. Run it once to see the first wave, or go straight
to the pump below it and come back later.
"""

SM = """### Monitor

One line per arm: a running arm prints the last line of its own log, a waiting arm prints
the file it is blocked by. A `ready` line means a slot is free and `tick()` has not been
called since.
"""

SP = """The pump is the cell that makes this hands-free: running a cell is activity, so the
runtime is not idle-disconnected while it works, and it returns by itself after a cap so the
session is released before Colab takes it away at the hard limit. Interrupt it whenever you
like - the arms that finished are on disk@@SPARE_CLAUSE@@, and re-running the cell resumes
the queue where it stopped. Only the arms in flight restart from zero, which is why the slow
depths are split across two notebooks.
"""

S5 = """## 5. Read the result

Per seed: the certificate the repair ended with, then that certificate re-derived from the
checkpoint by `src/c4_endgame_verify.py`, once per suite. `W` is the worst of the teacher's
optimal tie-breaks over the suite, so `W = 1.00` is a certification - play is a draw however
the teacher breaks ties. The block after each depth counts the certifications and puts a
Wilson 95% interval on each count: the repair's own claim, the claim re-derived on suite A,
and the claim made on blind suite B, which is the one that answers the objection.
"""

S6 = """## 6. Ship what exists

The queue above only hands a depth to your browser once every arm it plans has landed, which
is what keeps the training quiet. This cell does not wait: it zips what is on disk right now,
arms from an unfinished depth included, and sends it over. Run it as the last cell of a
session, and run it *before* you close the tab if Colab warns that the runtime is ending -
without it, whatever finished after the last complete depth dies with that runtime.

Re-running it is safe: a zip is only rebuilt when its contents changed, so a depth already
delivered does not download twice.
"""

SHIP_CODE = '''suelta = ship()
print('%d zip%s al navegador%s' % (len(suelta), '' if len(suelta) == 1 else 's',
      '' if suelta else ': nada nuevo, cada profundidad ya está en los anteriores'))
for a in suelta:
    print('  ', a.name)
'''

S7 = """## 7. When the queue is empty

Every JSON, checkpoint, log and both suites are in the zips your browser took, one per
depth. Nothing else has to be collected: the suites are inside them, so arms a reviewer
re-runs read the same forty positions this notebook did. Send me the zips and I will do the
rest. Please do not re-run the queue afterwards - it overwrites nothing, and it would spend
hours that the paper does not need.
"""

# ---------------------------------------------------------------------------- code ------
ARTIFACTS = '''import os, pathlib, zipfile

try:
    from google.colab import files as _browser     # descarga directa en el navegador
except ImportError:
    _browser = None

CONTENT = pathlib.Path(os.environ.get('COLAB_CONTENT', '/content'))
SLUG = '@@SLUG@@'
_pushed = {}    # profundidad -> firma de lo ya descargado, para no repetir descargas

# True (the finish queue): nothing is written next to the training while arms run, because
# zipping and handing a file to the browser costs the arm wall time. A depth ships when its
# last arm lands, and the ship cell at the end of the document sweeps whatever is left.
# False (the fourteen-cell queue): ship as each arm finishes, the older behaviour.
PUSH_AT_END = @@PUSH_AT_END@@

# (vacías, semillas): todo brazo que corre este notebook, y nada más. Ninguna de estas
# semillas tiene resultado archivado en el bundle que el notebook trae, así que no repite
# trabajo: las semillas ya publicadas de cada profundidad se quedan donde están.
PLAN = @@PLAN@@


def arm_paths(E, s):
    """Los tres JSON de un brazo y sus logs, con los nombres exactos que escribe la queue."""
    js = [pathlib.Path('data', 'c4_end%d_s%d.json' % (E, s)),
          pathlib.Path('data', 'c4_end%d_verify_s%d.json' % (E, s)),
          pathlib.Path('data', 'c4_end%d_holdoutB_s%d.json' % (E, s))]
    return js + [pathlib.Path('runs', '_colab_' + p.name[:-5] + '.log') for p in js]


def artefacts(E):
    """Lo que este notebook produce para una profundidad, dentro del repo."""
    seeds = {s for e, ss in PLAN if e == E for s in ss}
    found = sum((arm_paths(E, s) for s in sorted(seeds)), [])
    found += [pathlib.Path('data', 'c4_endgame_roots_%d_40_s0.json' % E),
              pathlib.Path('data', 'c4_endgame_roots_%d_40_holdoutB.json' % E)]
    found += sum((sorted(pathlib.Path('runs').glob('c4_end%d_s%d/c4_gru*.pt' % (E, s)))
                  for s in sorted(seeds)), [])
    return sorted({p for p in found if p.is_file()})


def _done(E):
    """True cuando cada brazo planeado de la profundidad ya dejó sus tres JSON y sus logs."""
    return all(p.exists() for _e, seeds in PLAN if _e == E for s in seeds
               for p in arm_paths(E, s))


def snapshot():
    """Empaqueta y descarga las profundidades que ya están enteras; a medias, no.

    Con PUSH_AT_END (la escalera que queda) un zip se construye sólo cuando el último brazo
    de su profundidad aterrizó: mientras hay brazos en vuelo esta función no toca el disco,
    así que el entrenamiento no paga ni un zip ni una ventana de descarga. Con
    PUSH_AT_END=False (la cola de catorce vacías) cada pasada entrega lo que haya, que es
    como se corrió la primera escalera. La firma ignora los logs: crecen a cada minuto y
    descargarían en cada pasada.
    """
    return [a for depth, _seeds in PLAN
            if not PUSH_AT_END or _done(depth)
            for a in [_push(depth)] if a]


def ship():
    """La salida de emergencia: entrega lo que haya, aunque un brazo esté a medias.

    Corre esta celda antes de cerrar la pestaña si Colab te avisa de que el tiempo se acaba;
    es la única forma de que lo ya calculado salga del runtime.
    """
    return [a for depth, _ in PLAN for a in [_push(depth)] if a]


def _push(E):
    found = artefacts(E)
    sig = {p.as_posix(): p.stat().st_size for p in found if p.suffix != '.log'}
    if not found or sig == _pushed.get(E):
        return None
    CONTENT.mkdir(parents=True, exist_ok=True)
    archive = CONTENT / ('c4_%s_end%d_results.zip' % (SLUG, E))
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in found:
            z.write(p, arcname=p.as_posix())
    _pushed[E] = sig
    size = archive.stat().st_size / 2 ** 20
    if _browser is None:
        print('%d artefactos -> %s (%.2f MB), sin Colab' % (len(found), archive, size))
    else:
        print('%d artefactos -> el navegador descarga %s (%.2f MB)'
              % (len(found), archive.name, size), flush=True)
        try:
            _browser.download(str(archive))
        except Exception as exc:
            # la descarga es el canal de salida, no el experimento: si el navegador no
            # está aquí, el queue sigue vivo y el zip espera en /content
            print('  AVISO: no se pudo enviar al navegador (%s); el zip está en %s'
                  % (exc, archive))
    return archive


print('sin Drive: los resultados llegan al navegador en c4_%s_end<vacías>_results.zip, '
      '%s' % (SLUG, 'cuando una profundidad termina entera (o con la celda ship()).'
              if PUSH_AT_END else 'uno por brazo terminado.'))
'''

SUITES = '''try:
    REPO = bootstrap()
except NameError:
    raise SystemExit('run the section 0 setup cell first - it unpacks the bundle the cell '
                     'above carries and puts the repository on sys.path')
import json, pathlib, time

import c4_game as g
from c4_endgame import random_games

g.configure(6, 7, 4)
B_SIZE = 40              # the holdout is the same size as the suite the repair reads
B_WAVES = 6              # seeds 500+E .. 505+E, until forty unseen positions are found
A_WAVES = 1              # a published suite is one stream; a fresh depth may need more
oracle = g.Oracle()
SUITE_A, SUITE_B = {}, {}


def _boards(roots):
    return {tuple(r['board']) for r in roots}


def _check(roots, E, label):
    assert len(roots) == 40, '%s holds %d positions, not 40' % (label, len(roots))
    for r in roots:
        assert len(r['board']) == g.N, '%s: not a full 6x7 board' % label
        empties = r['board'].count(0)      # una celda vacía es 0; g.EMPTY es el tablero vacío
        assert empties == E, '%s: a position with %d empty cells is not a %d-empty endgame' \\
            % (label, empties, E)
        assert not g.terminal(tuple(r['board'])), \\
            '%s: a finished position cannot be repaired' % label
        assert r['value'] == 0, '%s: a root the learner cannot draw is not a draw suite' \\
            % label


def _grow(path, E, need, exclude, seed0, waves, per_games, label):
    """Sample draw-valued roots in waves, checkpointed, so an interrupted suite resumes.

    `random_games` stops at the first `need` roots of one stream, so the first wave of a
    depth that publishes here for the first time is exactly the suite that call would have
    returned; later waves only fire when the stream ran out of unseen draw roots.
    """
    part = pathlib.Path(str(path) + '.part')
    have = json.loads(part.read_text()) if part.exists() else []
    seen = set(exclude) | _boards(have)
    wave = 0
    while len(have) < need and wave < waves:
        t0 = time.time()
        fresh, games = random_games(oracle, E, need, seed0 + wave, 0, max_games=per_games)
        added = 0
        for r in fresh:
            b = tuple(r['board'])
            if len(have) >= need or b in seen:
                continue
            seen.add(b)
            have.append(r)
            added += 1
        wave += 1
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps(have))
        print('  %s wave %d: %d new draw roots from %d games -> %d/%d (%.0f s)'
              % (label, wave, added, games, len(have), need, time.time() - t0))
    if len(have) < need:
        raise SystemExit('%s: only %d of %d unseen draw roots found in %d waves at %d empty '
                         'cells. Raise per_games or the waves for %s rather than shrinking '
                         'the suite or reusing a position from the other one.'
                         % (label, len(have), need, waves, E, label))
    path.write_text(json.dumps(have))
    part.unlink(missing_ok=True)
    return have


for E, seeds in PLAN:
    a_path = pathlib.Path('data', 'c4_endgame_roots_%d_40_s0.json' % E)
    if not a_path.exists():
        print('%d vacias: suite A does not exist yet, sampling it (this is the slow part '
              'of a new depth)' % E)
        _grow(a_path, E, 40, set(), 0, A_WAVES, @@MAX_GAMES_A@@, 'suite A')
    A = json.loads(a_path.read_text())
    _check(A, E, 'suite A of %d empty cells' % E)
    SUITE_A[E] = str(a_path)

    b_path = pathlib.Path('data', 'c4_endgame_roots_%d_40_holdoutB.json' % E)
    if not b_path.exists():
        print('%d vacias: sampling the blind holdout, never shown to the repair' % E)
        _grow(b_path, E, B_SIZE, _boards(A), 500 + E, B_WAVES, @@MAX_GAMES@@, 'suite B')
    B = json.loads(b_path.read_text())
    _check(B, E, 'suite B of %d empty cells' % E)
    SUITE_B[E] = str(b_path)

    shared = _boards(A) & _boards(B)
    assert not shared, ('suite B is not a holdout: %d positions are also in suite A'
                        % len(shared))
    print('%d vacias | A %s | B ciega %s | posiciones compartidas 0 | semillas %s'
          % (E, a_path.name, b_path.name, str(seeds)))

print('the repair reads only the A column; the verifier reads both, from the saved weights')
'''

QUEUE = '''import json, os, pathlib, shutil, subprocess

BUDGET = @@BUDGET@@   # el presupuesto con que se publicó cada profundidad

def build(empties, seeds):
    \"\"\"Repair against suite A, then read those weights again on A and on the blind B.\"\"\"
    E = empties
    common = ['--rows', '6', '--cols', '7', '--win', '4', '--empties', str(E),
              '--roots', '40', '--root-value', '0', '--branch', '3',
              '--hidden', '128', '--emb', '16', '--epochs', '200', '--lr', '2e-3',
              '--agg-repeat', str(BUDGET[E]['agg_repeat']), '--stop', BUDGET[E]['stop'],
              '--max-iters', str(BUDGET[E]['max_iters']),
              '--roots-file', SUITE_A[E], '--threads', str(THREADS)]

    def repair(s):
        out = 'data/c4_end%d_s%d.json' % (E, s)
        return dict(name='repair s%d' % s, out=out, needs=[],
                    cmd=['python', 'src/c4_endgame.py'] + common +
                         ['--seed', str(s), '--no-controls', '--ckpt-out',
                          'runs/c4_end%d_s%d' % (E, s), '--out', out])

    def verify(s, suite, tag):
        src = 'data/c4_end%d_s%d.json' % (E, s)
        ckpt = 'runs/c4_end%d_s%d/c4_gru.pt' % (E, s)
        out = ('data/c4_end%d_verify_s%d.json' % (E, s) if tag == 'A'
               else 'data/c4_end%d_holdoutB_s%d.json' % (E, s))
        return dict(name='verify %s s%d' % (tag, s), out=out, needs=[src, ckpt],
                    cmd=['python', 'src/c4_endgame_verify.py', '--ckpt', ckpt,
                         '--roots', suite, '--hidden', '128', '--emb', '16',
                         '--orders', '120', '--games', '40', '--out', out])

    jobs = []
    for s in seeds:
        jobs.append(repair(s))
        jobs.append(verify(s, SUITE_A[E], 'A'))
        jobs.append(verify(s, SUITE_B[E], 'B'))
    for j in jobs:
        j['E'] = E
        j['log'] = 'runs/_colab_' + j['out'].split('/')[-1][:-5] + '.log'
    return jobs

JOBS = [j for E, seeds in PLAN for j in build(E, seeds)]
for E, seeds in PLAN:
    print('%2d vacias | semillas %s | %d brazos (repair en A, verify en A y en B)'
          % (E, str(seeds), sum(1 for j in JOBS if j['E'] == E)))
print('en total %d brazos, %d corriendo a la vez' % (len(JOBS), MAX_PAR))
'''

RESULTS = '''import json, math, os

def wilson(k, n, z=1.96):
    \"\"\"95% interval for a rate measured on n arms; (nan, nan) while there are none.\"\"\"
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

print('W = peor desempate del maestro sobre el suite; W 1.00 = certificado de tablas')
print()
for E, seeds in PLAN:
    got = [s for s in seeds if os.path.exists('data/c4_end%d_s%d.json' % (E, s))]
    print('=== %d vacias: %d de %d semillas con brazo en disco ==='
          % (E, len(got), len(seeds)))
    if not got:
        print('  (todavia nada: deja corriendo el pump)')
        continue
    tally = {'repair': [0, 0], 'A': [0, 0], 'B': [0, 0]}
    for s in got:
        d = json.load(open('data/c4_end%d_s%d.json' % (E, s)))
        fin = d['repair']['final']
        tally['repair'][1] += 1
        tally['repair'][0] += fin['worst_draw_rate'] == 1.0
        print('seed %d: W %.2f B %.2f D %.3f | rondas %d | historias agregadas %d | %d s'
              % (s, fin['worst_draw_rate'], fin['best_draw_rate'],
                 fin['random_draw_rate_exact'], len(d['repair']['log']) - 1,
                 len(d['repair']['aggregate']), d['repair']['elapsed_s']))
        for tag in ('A', 'B'):
            f = ('data/c4_end%d_verify_s%d.json' % (E, s) if tag == 'A'
                 else 'data/c4_end%d_holdoutB_s%d.json' % (E, s))
            if not os.path.exists(f):
                continue
            v = json.load(open(f))
            tally[tag][1] += 1
            tally[tag][0] += v['worst_draw_rate'] == 1.0
            print('   verify %s: W %.2f B %.2f D %.3f | raices forzadas a tablas %d/%d '
                  '| %d ordenes %.3f-%.3f | adaptativo %.3f-%.3f'
                  % (tag, v['worst_draw_rate'], v['best_draw_rate'],
                     v['random_draw_rate_exact'], v['n_roots_drawn_worst'], v['roots'],
                     v['deterministic_orders'], v['order_min'], v['order_max'],
                     v['adaptive_min'], v['adaptive_mean']))
    for tag, label in (('repair', 'certificado que afirma el repair (suite A)'),
                       ('A', 'verificado sobre la suite A'),
                       ('B', 'verificado sobre la suite B CIEGA')):
        k, n = tally[tag]
        if n:
            lo, hi = wilson(k, n)
            print('  %-40s %2d/%d = %.2f  [%.2f, %.2f]' % (label, k, n, k / n, lo, hi))
        else:
            print('  %-40s sin brazos todavia' % label)
    print()

print('descarga final:')
for a in snapshot() or []:
    print('  ', a.name)
'''

# ------------------------------------------------------------------------ the two docs --
# The budget each depth was published with, read out of the archived arm JSONs in data/:
# eight empty cells ran with 20 aggregated repeats and the 'agg' stopping rule, ten and
# twelve with 60 and 'both'. A new seed is only comparable with its own depth if it repeats
# those numbers, so they are a table here rather than one recipe for every depth, and
# tests/test_followup_notebooks.py re-reads them from the archives on every run.
BUDGET = """{
    8: {'agg_repeat': 20, 'stop': 'agg', 'max_iters': 12},
    10: {'agg_repeat': 60, 'stop': 'both', 'max_iters': 24},
    12: {'agg_repeat': 60, 'stop': 'both', 'max_iters': 24},
    14: {'agg_repeat': 60, 'stop': 'both', 'max_iters': 24},
}"""

LADDER = dict(
    name='c4_ladder_finish.ipynb',
    slug='finish',
    plan='((12, (7, 8, 9)), (10, (3, 4, 5)), (8, (3, 4, 5)))',
    max_games_a='500000',
    max_games='400000',
    minutes=600,
    push_at_end=True,
    title='the arms the seed ladder did not finish: twelve, ten and eight empty cells',
    what='Seeds 3 to 6 at twelve empty cells arrived from the first ladder run and are '
         'already archived, so this queue starts where that one stopped. Seeds 7 and 8 were '
         'cut just after their base fit and seed 9 never started, and the ten-empty and '
         'eight-empty bundles did not begin, because the runtime was taken away before the '
         'pump got to them. Nine repair arms and eighteen verifications, each under the '
         'budget its depth was published with, so twelve reaches ten seeds and ten and eight '
         'reach six each - and twelve is the depth the paper needs error bars for.',
    hands='Nothing is written next to the training while the arms run: no zip is built and no '
          'download window opens until a depth is complete, because packaging cost the arm '
          'wall time in the last run. So the browser stays quiet for hours, then takes one '
          'zip per depth. If Colab warns that the session is ending, run the ship cell at the '
          'bottom of this notebook first - it hands over whatever exists even from an '
          'unfinished depth, which is the only way to keep the arms already computed. Expect '
          'the session to be cut at least once; re-running the pump cell resumes the queue '
          'where it stopped, and only the arms in flight restart.',
)

NOTEBOOKS = [LADDER]


def cells_for(spec):
    """The document, in the order a reviewer runs it."""
    setup = scaling_cell('setup').replace('c4_endgame_scaling.ipynb', spec['name'])
    tick = scaling_cell('tick').replace('    snapshot(E)\n', '    snapshot()\n')
    minutes = spec['minutes']
    pump = scaling_cell('pump')
    pump = "\n".join(l for l in pump.split("\n") if not l.strip().startswith('MAX_MINUTES'))
    pump = pump.replace('t_start = time.time()',
                        'MAX_MINUTES = %d\nt_start = time.time()' % minutes, 1)
    monitor, resource = scaling_cell('monitor'), scaling_cell('resource')
    assert 'snapshot(E)' not in tick, 'the tick cell changed shape; the substitution missed'
    assert 'MAX_MINUTES = %d' % minutes in pump, 'the pump cell has no budget line to set'

    def sub(text):
        return (text.replace('@@SLUG@@', spec['slug']).replace('@@PLAN@@', spec['plan'])
                .replace('@@BUDGET@@', BUDGET)
                .replace('@@MAX_GAMES_A@@', spec['max_games_a'])
                .replace('@@MAX_GAMES@@', spec['max_games'])
                .replace('@@PUSH_AT_END@@', 'True' if spec['push_at_end'] else 'False'))

    # Two shapes of the same queue: the fourteen-cell one ships as each arm finishes, the
    # finisher ships a depth whole, so the prose a reviewer reads matches the cell they run.
    end = spec['push_at_end']
    prose = {'@@PUSH_CLAUSE@@':
             (', and it leaves the artefacts alone while any arm of a depth is still '
              'training: that depth reaches your browser whole, when its last arm lands'
              if end else ' and hands every finished arm to the browser'),
             '@@SPARE_CLAUSE@@':
             (' (they reach the browser when the depth is complete, and the ship cell below '
              'takes anything partial before the runtime dies)' if end else ' and in your '
              'browser')}

    def say(text):
        for k, v in prose.items():
            text = text.replace(k, v)
        assert '@@' not in text, 'the %s prose has a placeholder nobody fills' % spec['name']
        return text

    return [
        ('markdown', TITLE.replace('@@TITLE@@', spec['title']).replace('@@WHAT@@',
                                                                       spec['what'])
                     .replace('@@HANDS@@', spec['hands'])),
        ('code', sub(ARTIFACTS)),
        ('markdown', S0),
        ('code', setup),
        ('markdown', S1),
        ('code', resource),
        ('markdown', S2),
        ('code', sub(SUITES)),
        ('markdown', S3),
        ('code', sub(QUEUE)),
        ('markdown', say(S4)),
        ('code', tick),
        ('markdown', SM),
        ('code', monitor),
        ('code', pump),
        ('markdown', say(SP)),
        ('markdown', S5),
        ('code', RESULTS),
        ('markdown', S6),
        ('code', SHIP_CODE),
        ('markdown', S7),
    ]


def render(cells):
    out = []
    for kind, text in cells:
        lines = text.rstrip("\n").split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "execution_count": None, "metadata": {},
                "source": source}
        out.append(cell if kind == "markdown" else dict(cell, outputs=[]))
    return {"cells": out, "metadata": {
        "colab": {"provenance": None},
        "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
        "language_info": {"name": "python", "version": "3.10.12"}},
        "nbformat": 4, "nbformat_minor": 0}


def _runnable(src):
    """A code cell as Python: the IPython magics become blank statements.

    Same rule as tests/test_colab_notebook.py: `!python` and `%cd` are kernel sugar, so the
    generator can check every cell parses without pretending they are Python.
    """
    return "\n".join("" if l.startswith(("%", "!")) else l for l in src.split("\n"))


def write(spec, check_only=False):
    path = OUT_DIR / spec["name"]
    nb = render(cells_for(spec))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile(_runnable("".join(c["source"])), "%s cell %d" % (path.name, i), "exec")
    if check_only:
        if not path.exists():
            raise SystemExit("%s is missing; run python colab/build_followup_notebooks.py"
                             % path)
        on_disk = [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
                   if "BUNDLE_SHA256 = " not in "".join(c["source"])]
        # the bundle cell is injected by colab/build_bundle.py after these cells are
        # written, so it is compared by `build_bundle.py --check`, not here
        same = on_disk == nb["cells"]
        print("%s: %s" % (path.name, "matches its generator" if same
                          else "DIFFERS from colab/build_followup_notebooks.py"))
        return 0 if same else 1
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("%s: %d cells written, every code cell compiles" % (path.name, len(nb["cells"])))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if a notebook no longer matches this generator")
    args = ap.parse_args()
    rc = 0
    for spec in NOTEBOOKS:
        rc |= write(spec, args.check)
    if not args.check:
        for spec in NOTEBOOKS:
            rc |= build_bundle.build_for(OUT_DIR / spec["name"])
        print("verify with: python -m pytest tests/test_colab_notebook.py -q")
    return rc


if __name__ == "__main__":
    sys.exit(main())
