"""Write the one notebook that finishes everything this repository still owes.

Why it exists: the outstanding work is of two kinds. The first is verifier reads - calls of
`src/c4_endgame_verify.py` on checkpoints that already exist, which is what the paired base rows of
`tab:blind` were missing. One of those is five minutes of single-threaded recursion, so the laptop does
them by itself and they are here only until it has: the six twelve-cell pairing reads ran locally on 5
November, and a build made after an archive lands simply stops queueing them. The second is three
repair arms at fourteen
empty cells, the next point on the scaling curve, which have never run anywhere: only two of their
base fits came back from a runtime that was taken away mid-depth. One document queues both, because
a reviewer should not have to know which of several notebooks holds which half of a debt. The list
is re-derived from `data/` at every build, so a document built after some of it has run is shorter,
not stale: it opened with forty-eight jobs, thirty-nine came back from a runtime, and the last six
were measured at home.

Checkpoints are the wrinkle. `.gitignore` excludes `*.pt`, so a runtime that rebuilds the code from
this notebook's bundle has the suites and the modules and nothing to read. Any weight a read needs
therefore travels inside the document, as base64 with a sha256 each, listed by `colab/reads_plan.py`
and recomputed before a byte is trusted - one cell per checkpoint, so no single cell is the four
megabytes that would make the editor unusable. A queue of nothing but arms needs none, and says so in
section 2 instead of printing an empty table.

What the queue holds, in the order it holds it:

* the sixteen reads of `sec:union` - the three union arms on suite A, on suite B and on the eighty
  positions of both, repaired and unrepaired - the measurement that separates the wider certificate
  from the doubled training sample, and the re-read that puts `sec:union`'s crossing on the
  independent verifier instead of the repair loop's own recursion. These have all landed.
* the five reads that make the published tables complete: the blind suite-B read of the three
  twelve-empty seeds that ran before the holdout was cut, and the memo-free suite-A read of the two
  eight-empty seeds `tab:blind` used to mark with a dagger. These have all landed too, which is why
  `tab:blind` reports twenty-two repaired arms and no dagger.
* last of the reads: the base reads that pair an arm's pre-repair checkpoint on both suites. These
  appear one depth at a time, because `reads_plan` emits a pairing row only for an arm that already
  has a blind read to compare with - so a build made right after an archive lands can owe reads that
  the same document could not have asked for when it was written. They are also the reads the laptop
  can do by itself, since the checkpoints are on this disk; when they are all archived, the queue this
  generator writes holds only the arms that need a machine with more than two cores.
* last, because it is the only thing here that trains: the three fourteen-empty arms, each a repair
  against suite A followed by its read on A and on the blind B, under the budget the twelve-empty
  arms were published with - read off those archives rather than typed, by `reads_plan.arm_budget`.

Ordering is not cosmetic. `tick()` launches what is unblocked, first come first served, so the reads
land before any repair starts; and a zip is built only when its own gate is complete, so while there
were reads owed every read gate had fired and been handed to the browser before the first
fourteen-empty arm was in flight. Nothing packages while anything trains, which is the one thing the
last ladder run paid wall time for. Once the reads are archived the same rule gives a document with
one gate and nothing to package mid-training at all, which is why `gates_literal` drops a gate the
plan no longer queues instead of declaring an empty one.

The prose is derived from the queue, not written next to it. `prose_for` reads the same plan the cells
were built from - counts of reads, of arms, of checkpoints, which suites are pointed at, how many arms
the paired control now covers, whether a union read is still owed - and fills the markdown from it, so
a document whose reads have all landed opens by saying it trains and carries no weights, and the
numbers in its headings cannot drift away from the jobs below them. Two sentences are worth naming
because they are the ones a stale document would get wrong: `blind_state` counts the blind
certifications out of the archived read JSONs rather than repeating that nothing certified, and a
cell that describes a check the results cell would skip - the union identity, with no `U` read queued
- is left out instead of printed as a promise. `say()` then asserts that no placeholder survived, so a
token nobody fills fails the build rather than shipping as "@@OWED@@".

Every read arrives with an expectation: the certificate the arm that made those weights logged,
copied out of its archived JSON by the generator, or - for an arm this notebook itself trains - read
off the repair JSON the runtime just wrote. The results cell refuses a read that disagrees, because
a memo-free read of the wrong weights is otherwise indistinguishable from a surprising result. It
also enforces the identity the union section rests on: over two board-disjoint suites, `W` of the
whole is the minimum of the two halves, the drawn counts add, and the mean is the mean.

The cells already pinned down in `colab/c4_endgame_scaling.ipynb` - the boot, the capacity cell, the
monitor and the hands-free pump - are copied from that notebook rather than rewritten, so a fix there
lands here too. `tests/test_colab_notebook.py` runs its structural promises over every notebook and
`tests/test_reads_notebooks.py` the ones specific to this one: that what it queues is exactly what
`reads_plan` says is missing, that every byte it carries hashes to a file on this disk, that no read
launches before its checkpoint and its suite exist, and that no zip opens over an unfinished gate.

    python colab/build_reads_notebooks.py              # write the notebook, then bundle it
    python colab/build_reads_notebooks.py --check      # exit 1 if the file no longer matches
    python runs/_reads_audit.py                        # the same plan, as a report
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_bundle                                     # noqa: E402
import build_followup_notebooks as bf                   # noqa: E402
import reads_plan as rp                                 # noqa: E402

REPO = bf.REPO
OUT_DIR = REPO / "colab"
B64_WIDTH = 100
MINUTES = 700
NAME = 'c4_finish_all.ipynb'
SLUG = 'finishall'

# What the shared pump cannot say on its own. A fourteen-empty arm writes its JSON only at the
# last round, so the completed-jobs counter does not move for hours while the arm is healthy; the
# pump prints the live round line from the log so a frozen counter and a stalled run are two
# visibly different outputs.
PROGRESS = ("    for j in JOBS:\n"
            "        if pids('out ' + j['out']) and os.path.exists(j['log']):\n"
            "            _l = [x for x in open(j['log']).read().splitlines() if x.strip()]\n"
            "            print('  RUN   %s | %s' % (j['name'], _l[-1][:104] if _l else ''))\n")

# The download gates, in the order the queue empties them: the three groups of reads that
# `reads_plan` says are owed, then the depth that has to be trained. `zip` is the fragment the
# browser sees in the filename, so four runs of this document produce four distinguishable zips.
GATES = [('union', 'reads_union'), ('closure', 'reads_closure'),
         ('pairing', 'reads_pairing'), ('end14', 'end14')]


# --------------------------------------------------------------------------- prose ----
TITLE = """# Connect Four endgames: everything the repository still owes, in one queue (@@TITLE@@)

**What this notebook does.** It drains a list. @@READS_SENTENCE@@ @@ARMS_SENTENCE@@ @@CK_SENTENCE@@

@@OWED@@

**What you have to do.** Run the cells in order, and re-run the pump whenever the runtime is taken
away. @@FETCH_SENTENCE@@ Your browser gets up to @@ZIPS@@ - one per section of the list, when
that section's last item lands - and nothing before, so a session that dies three quarters of the way
through still delivered what it did. Keep every zip and tell me where they landed.
"""

SCALING_SENTENCE = ("Fourteen empty cells is the deepest point of the ladder, and the arms it still "
                    "owes cost hours each: a free CPU runtime takes them more cheaply than a laptop "
                    "that is also doing something else.")


def blind_state():
    """Repaired arms read on the blind suite, and how many of those reads certify: from the JSONs."""
    n = cert = 0
    for r in rp.reads():
        if r['kind'] != 'repaired-on-B' or not rp.exists(r['out']):
            continue
        n += 1
        if json.loads((rp.REPO / r['out']).read_text(encoding='utf-8'))['worst_draw_rate'] == 1.0:
            cert += 1
    return n, cert


def union_sentence():
    """What `sec:union` is worth right now, counted from the plan rather than remembered.

    The scope argument rests on the union arms being read on all three suites; if one of those reads
    is missing the sentence has to say so, because a notebook that describes a closed result while
    the result is still owed is the failure this generator was written to stop.
    """
    rows = [r for r in rp.reads() if r['kind'].startswith('union-')]
    done = [r for r in rows if rp.exists(r['out'])]
    arms = len({r['arm'] for r in rows})
    if len(done) != len(rows):
        return ("`sec:union` is NOT closed: %d of its %d reads over `A + B` are archived, and until "
                "the rest land the eighty-position certificate is a claim about %d arms, not %d. "
                % (len(done), len(rows), len(done), len(rows)))
    return ("`sec:union` is fully measured - all %d arms re-read on A, on B and over the eighty "
            "positions, their bases the same way - and a doubled sample certified nothing on its own. "
            % arms)


S0 = """## 0. Nothing to set up

The cell titled *The bundle* holds a base64 zip of the four modules this queue runs and every root
suite in the repository, with a sha256 beside each file. The setup cell re-computes those hashes and
unpacks them into a working copy under `/content`. A bundle that does not match - edited inside the
notebook, or older than the code - is refused, and the cell falls back to cloning the repository.
"""

S1 = """## 1. How much machine do we actually have

A read is one single-threaded recursion and an arm is one recursion plus a small GRU refit, so the
queue holds two jobs and gives each a thread. This is the same capacity cell the training notebooks
run, and nothing here is configured by hand.
"""

S2 = """## 2. The weights this notebook carries

A read needs a checkpoint, and `*.pt` files are not in git: the repository a runtime rebuilds from the
bundle above has the code and the suites and nothing to read. So the cell below lists the @@NCK@@
checkpoints this queue loads - each with its sha256, its size and the repair JSON whose certificate it
is supposed to reproduce - and the cells after it carry the bytes, one checkpoint per cell so the
editor stays usable. Before a file is written its hash is recomputed; a mismatch stops the notebook,
because the alternative is a read of some other policy printed as a result of this one. A file already
in the working copy with the right hash is left alone, so re-running these cells after a runtime
restart costs nothing. The last cell of the section refuses the document if a checkpoint arrived
without its bytes.
"""

S3 = """## 3. The suites the queue is pointed at

@@SUITE_SENTENCE@@ All of them travel in the bundle, and this cell does not just open them: it refuses
a suite whose positions are not unfinished draws at the depth it claims, refuses A and B of a depth if
they share a board (a holdout that overlaps the training set is not a holdout).@@UNION_CHECK@@
"""

S2_EMPTY = """## 2. The weights this notebook carries

None, and the section says so rather than printing an empty table. A read of weights that already
exist is what needs a checkpoint to travel inside as base64, because `*.pt` files are not in git - and
every read of that kind the repository owed is archived now. What this document holds is training: each
arm fits its own base, saves `c4_gru_base.pt` next to `c4_gru.pt`, and is then re-read by its own two
verifications inside this same queue, on the weights it just wrote. The cells of this section are the
machinery that would lay the weights down, kept in place so the document has one shape whether or not
it carries bytes, and the last cell of the section still refuses the document if a checkpoint is listed
without the cell of bytes that belongs to it.
"""

S4 = """## 4. The queue

@@QUEUE_PARAGRAPH@@ Nothing here re-runs a result the repository already has: the plan below was
generated from the JSONs on disk, and every one of these output files is absent from it.

@@QUEUE_ORDER@@
"""

SM = """### Monitor

One line per job: running, done, or waiting on the file that has to exist first. A `ready` line means
a slot is free and `tick()` has not been called since.
"""

SP = """@@PUMP_OPEN@@

Every pass it also prints a `RUN` line per job still in flight, with that job's last log line. This
is the only way to tell a working repair from a stalled one: an arm writes its JSON on the final
round, so the number of finished jobs stays exactly where it was for hours while the rounds keep
coming. A counter that does not move and a `RUN` line whose round number does not move are different
problems; a counter that does not move and a round number that does is just what fourteen empty
cells look like.
"""

S5 = """## 5. Read the result, and check it against the arm that made these weights

@@VERIFY@@
"""

S6 = """## 6. Ship what exists

The queue hands a zip to your browser when the last item of its section lands. This cell does not
wait: it zips whatever is on disk right now, section by section, and sends it over. Run it as the last
cell of a session, and run it *before* you close the tab if Colab warns that the runtime is ending -
without it, whatever finished after the last download dies with that runtime. Re-running is safe: a
zip is only rebuilt when its contents changed.
"""

S7 = """## 7. When the queue is empty

@@SHIP_LIST@@ Send them to me and I will do the rest: the numbers go into
`tab:blind`, `sec:union` and the two generators that print them, and the weights stay where they are.
Please do not re-run the queue afterwards - it overwrites nothing, and it would spend hours the paper
does not need.
"""

# --------------------------------------------------------------------------- code ----
PACKER = '''import os, pathlib, zipfile

try:
    from google.colab import files as _browser     # descarga directa en el navegador
except ImportError:
    _browser = None

CONTENT = pathlib.Path(os.environ.get('COLAB_CONTENT', '/content'))
SLUG = '@@SLUG@@'
_pushed = {}      # puerta -> firma de lo ya descargado, para no repetir la descarga

# las puertas de este notebook: los JSON que cada una espera enteros, y lo extra que lleva su
# zip. Un zip se construye SOLO cuando su puerta esta completa; mientras algo esta en vuelo
# esta celda no toca el disco, que es lo que le costo tiempo de reloj a la escalera pasada.
# globs: lo que un brazo produce fuera de data/ y que no existe hasta que el corre - y lo
# unico lugar del mundo donde va a existir.
GATES = @@GATES@@


def artefacts(gate):
    g = GATES[gate]
    found = [pathlib.Path(p) for p in g['outs'] if pathlib.Path(p).is_file()]
    found += [pathlib.Path('runs', '_colab_' + pathlib.Path(p).name[:-5] + '.log')
              for p in g['outs'] if pathlib.Path('runs', '_colab_' +
              pathlib.Path(p).name[:-5] + '.log').is_file()]
    found += [pathlib.Path(p) for p in g['extra'] if pathlib.Path(p).is_file()]
    for pat in g['globs']:
        found += [p for p in sorted(pathlib.Path('.').glob(pat)) if p.is_file()]
    return sorted(set(found))


def _complete(gate):
    return all(pathlib.Path(p).is_file() for p in GATES[gate]['outs'])


def snapshot():
    """Entregar una puerta cuando su ultimo job aterrizo; a medias, nunca.

    @@GATE_TIMING@@
    """
    return [a for a in [_push(g) if _complete(g) else None for g in GATES] if a]


def ship():
    """La salida de emergencia: entrega lo que haya, aunque una puerta este a medias."""
    return [a for a in [_push(g) for g in GATES] if a]


def _push(gate):
    found = artefacts(gate)
    sig = {p.as_posix(): p.stat().st_size for p in found if p.suffix != '.log'}
    if not found or sig == _pushed.get(gate):
        return None
    CONTENT.mkdir(parents=True, exist_ok=True)
    archive = CONTENT / ('c4_%s_%s_results.zip' % (SLUG, GATES[gate]['zip']))
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in found:
            z.write(p, arcname=p.as_posix())
    _pushed[gate] = sig
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
            # está acá, la cola sigue viva y el zip espera en /content
            print('  AVISO: no se pudo enviar al navegador (%s); el zip está en %s'
                  % (exc, archive))
    return archive


print('sin Drive: hasta %d %s al navegador, uno por seccion de la cola vaciada (o con la '
      'celda ship())' % (len(GATES), 'zip' if len(GATES) == 1 else 'zips'))
'''

WEIGHTS = '''try:
    REPO
except NameError:
    raise SystemExit('run the section 0 setup cell first: it unpacks the bundle and makes '
                     'the directory the weights are written into')
import base64, hashlib, pathlib

# checkpoint -> su sha256, su tamano y el brazo cuyo certificado debe reproducir. Los bytes
# vienen despues, una celda por checkpoint, para que ningun celda pese megabytes.
WEIGHTS = @@WEIGHTS@@
_seen = []


def _put(rel, parts):
    """Escribir un checkpoint del notebook en el disco del runtime, despues de comprobarlo."""
    if rel not in WEIGHTS:
        raise SystemExit('%s: este notebook no cola ninguna lectura de ese checkpoint, asi que '
                         'no hay certificado con el comprobarlo; no se escribe' % rel)
    meta = WEIGHTS[rel]
    blob = base64.b64decode(''.join(parts))
    got = hashlib.sha256(blob).hexdigest()
    if got != meta['sha256']:
        raise SystemExit('%s: el base64 de esta celda no hash a lo que el notebook anuncia (%s), '
                         'así que no es el checkpoint del brazo %s; no se lee nada'
                         % (rel, got[:12], meta['arm']))
    if len(blob) != meta['bytes']:
        raise SystemExit('%s: %d bytes esperados, %d en la celda' %
                         (rel, meta['bytes'], len(blob)))
    p = pathlib.Path(rel)
    if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == got:
        print('  ya está   %s (%s del brazo %s)' % (rel, got[:12], meta['arm']))
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
        print('  escrito   %s | %s | brazo %s, %d vacías, certificado en %s'
              % (rel, got[:12], meta['arm'], meta['E'], meta['arm_json']))
    _seen.append(rel)
'''

WEIGHT_DONE = '''faltan = sorted(set(WEIGHTS) - set(_seen))
if faltan or len(_seen) != len(WEIGHTS):
    raise SystemExit('faltan las celdas de bytes de %d checkpoints (%s): sin ellos las lecturas '
                     'que los necesitan esperan para siempre, y un zip parcial no paga la deuda'
                     % (len(faltan), ', '.join(faltan[:4])))
if not WEIGHTS:
    print('0 checkpoints dentro: ninguna lectura de esta cola pesa pesos ya entrenados; los '
          'unicos archivos .pt que existiran aqui son los que entrenen los brazos de catorce '
          'vacias, y salen en el zip end14')
else:
    print('%d checkpoints en el runtime | %.2f MB en total | %d celdas de pesos corre'
          % (len(WEIGHTS), sum(m['bytes'] for m in WEIGHTS.values()) / 2 ** 20, len(_seen)))
'''

SUITES = '''import json, pathlib

import c4_game as g
g.configure(6, 7, 4)

# las suites que esta cola punta, tal como las trae el bundle, con su profundidad: la
# publicatoria A, la ciega B y - para los brazos certificados sobre las dos - la union U
SUITES = @@SUITES@@
SUITES_E = @@SUITES_E@@


def _boards(roots):
    return {tuple(r['board']) for r in roots}


_cache = {}
for key, path in sorted(SUITES.items()):
    p = pathlib.Path(path)
    if not p.is_file():
        raise SystemExit('%s is not in the runtime: the suite a read is pointed at has to be '
                         'the committed one, so this notebook stops rather than sampling a '
                         'different set of positions' % path)
    roots = json.loads(p.read_text(encoding='utf-8'))
    _cache[key] = roots
    E = SUITES_E[key]
    want = 80 if key[0] == 'U' else 40
    assert len(roots) == want, 'suite %s holds %d positions, not %d' % (key, len(roots), want)
    for r in roots:
        assert len(r['board']) == g.N, 'suite %s: not a full 6x7 board' % key
        empties = r['board'].count(0)      # una celda vacía es 0; g.EMPTY es el tablero vacío
        assert empties == E, ('suite %s: a position with %d empty cells is not a %d-empty '
                              'endgame' % (key, empties, E))
        assert not g.terminal(tuple(r['board'])), 'suite %s: a finished position' % key
        assert r['value'] == 0, 'suite %s: a root the learner cannot draw' % key
    print('suite %-3s | %2d posiciones | todas unfinished y de valor 0 | %2d vacías | %s'
          % (key, len(roots), E, path))

for _E in sorted(set(SUITES_E.values())):
    a, b = 'A%d' % _E, 'B%d' % _E
    if a in _cache and b in _cache:
        shared = _boards(_cache[a]) & _boards(_cache[b])
        assert not shared, ('suite B of %d empties is not a holdout: %d positions are also in '
                            'suite A, so a blind read of them measures nothing' % (_E, len(shared)))
        print('A%d y B%d comparten 0 posiciones' % (_E, _E))
    u = 'U%d' % _E
    if u in _cache:
        assert _cache[u] == _cache[a] + _cache[b], (
            'suite U of %d empties is not suite A followed by suite B, so a certificate over it '
            'is not a certificate over both halves' % _E)
        print('U%d es exactamente A seguida de B: %d + %d = %d'
              % (_E, len(_cache[a]), len(_cache[b]), len(_cache[u])))

print('las suites están validadas; cada lectura espera a la suya antes de arrancar')
'''

QUEUE = '''import json, os, pathlib, shutil, subprocess

# una lectura = un checkpoint, una suite, un JSON de salida; un brazo = un repair y sus dos
# lecturas. Generado por colab/build_reads_notebooks.py a partir de colab/reads_plan.py, sobre
# los archivos que faltaban en data/ en el momento de escribir este notebook.
READS = @@READS@@
ARMS = @@ARMS@@
BUDGET = @@BUDGET@@   # el presupuesto con que se publico la profundidad de la que heredan


def _suite(row):
    """La suite que la seccion 3 valido, y el chequeo de que la cola apunta exactamente a ella."""
    p = SUITES[row['key']]
    assert p == row['suite_file'], ('%s: la cola apunta a %s pero la celda de suites valido %s, '
                                    'o sea que esta lectura correria sobre posiciones que nadie '
                                    'comprobo' % (row['out'], row['suite_file'], p))
    return p


def repair(a):
    """El unico comando de src/c4_endgame.py del notebook: el brazo nuevo, con su repair."""
    E, s = a['E'], a['seed']
    return dict(name='repair %d vacias s%d' % (E, s), out=a['out'], needs=[_suite(a)],
                gate=a['gate'],
                cmd=['python', 'src/c4_endgame.py', '--rows', '6', '--cols', '7',
                     '--win', '4', '--empties', str(E), '--roots', str(BUDGET['roots']),
                     '--root-value', str(BUDGET['root_value']),
                     '--branch', str(BUDGET['branch']), '--hidden', str(BUDGET['hidden']),
                     '--emb', str(BUDGET['emb']), '--epochs', str(BUDGET['epochs']),
                     '--lr', str(BUDGET['lr']), '--agg-repeat', str(BUDGET['agg_repeat']),
                     '--stop', str(BUDGET['stop']), '--max-iters', str(BUDGET['max_iters']),
                     '--roots-file', _suite(a), '--threads', str(THREADS), '--seed', str(s),
                     '--no-controls', '--ckpt-out', 'runs/c4_end%d_s%d' % (E, s),
                     '--out', a['out']])


def verify(src_row):
    """Una lectura memo-free: src/c4_endgame_verify.py sobre un checkpoint y una suite.

    Espera el checkpoint y la suite siempre, y en los brazos nuevos tambien el JSON del repair
    que todavia no existe: una lectura de unos pesos a medio escribir no es una lectura.
    """
    r = src_row
    waits = list(r['needs'])
    p = _suite(r)
    if p not in waits:
        waits.append(p)
    if r['kind'].startswith('arm-'):
        name = '%s nuevo' % r['arm']
    else:
        name = '%s %s' % (r['arm'], 'repaired' if 'repaired' in r['kind'] else 'base')
    return dict(name='%s on %s' % (name, r['which']), out=r['out'], needs=waits,
                gate=r['gate'],
                cmd=['python', 'src/c4_endgame_verify.py', '--ckpt', r['ckpt'],
                     '--roots', p, '--hidden', '128', '--emb', '16',
                     '--orders', '120', '--games', '40', '--out', r['out']])


JOBS = ([verify(r) for r in READS]
        + [repair(a) if a['kind'] == 'arm-repair' else verify(a) for a in ARMS])
for j in JOBS:
    j['log'] = 'runs/_colab_' + j['out'].split('/')[-1][:-5] + '.log'

_suites = sorted({row['suite_file'] for row in READS + ARMS})
print('%d lecturas + %d brazos nuevos = %d jobs, %d checkpoints dentro del notebook, %d suites, '
      '%d corriendo a la vez' % (len(READS), len({a['arm'] for a in ARMS}), len(JOBS),
                                 len(WEIGHTS), len(_suites), MAX_PAR))
for _s in _suites:
    print('   suite', _s)
for a in ARMS:
    if a['kind'] == 'arm-repair':
        print('   repair %s: --agg-repeat %s --stop %s --max-iters %s sobre %s'
              % (a['arm'], BUDGET['agg_repeat'], BUDGET['stop'], BUDGET['max_iters'],
                 a['suite_file']))
'''

RESULTS = '''import json, math, os, pathlib

# Cada linea de abajo es una lectura memo-free de src/c4_endgame_verify.py, y lo que cada brazo
# ya afirmó sobre la suite que certificó, copiado de su propio JSON por
# colab/build_reads_notebooks.py. Una lectura memo-free de esos mismos pesos tiene que dar lo
# mismo; si no coincide, el notebook leyó otros pesos. Los brazos que este notebook entrena no
# tienen copia: la expectation se lee del JSON que el propio runtime acaba de escribir.
EXPECTED = @@EXPECTED@@


def wilson(k, n, z=1.96):
    """95% interval for a rate measured on n arms; (nan, nan) while there are none."""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


KEYS = ('worst_draw_rate', 'best_draw_rate', 'random_draw_rate_exact', 'n_roots_drawn_worst')
ALL = READS + [a for a in ARMS if a['kind'] != 'arm-repair']


def expectation(row):
    """Lo que el brazo ya afirmo de si mismo, en la suite que el archivo dice que certifico."""
    repaired = row['kind'].startswith(('arm-read-on-', 'repaired-on-', 'union-repaired-'))
    e = EXPECTED.get(row['arm'], {}).get('repaired' if repaired else 'base')
    if e:
        return e
    p = pathlib.Path(row.get('arm_json', ''))
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding='utf-8'))
    roots_file = d['args']['roots_file'].replace("\\\\", "/")
    return {'cert': d['repair']['final'] if repaired else d['base']['certificate'],
            'suite': ('U' if 'AplusB' in roots_file else
                      'B' if 'holdoutB' in roots_file else 'A')}


read = {}
for r in ALL:
    if os.path.exists(r['out']):
        read[(r['arm'], r['kind'], r['which'])] = json.loads(pathlib.Path(r['out'])
                                                             .read_text(encoding='utf-8'))

left = [r for r in ALL if not os.path.exists(r['out'])]
print('=== %d de %d jobs en disco, %d por hacer ===' % (len(read), len(ALL), len(left)))

fail = []
for r in ALL:
    v = read.get((r['arm'], r['kind'], r['which']))
    if v is None:
        continue
    exp = expectation(r)
    tag = '%-16s %-22s on %s' % (r['arm'], r['kind'], r['which'])
    line = ('W %.3f B %.3f D %.4f | raices a tablas %d/%d | ordenes %d (%.3f-%.3f) | '
            'adaptativo %.3f-%.3f'
            % (v['worst_draw_rate'], v['best_draw_rate'], v['random_draw_rate_exact'],
               v['n_roots_drawn_worst'], v['roots'], v['deterministic_orders'],
               v['order_min'], v['order_max'], v['adaptive_min'], v['adaptive_mean']))
    if exp and exp['suite'] == r['which']:
        c = exp['cert']
        same = all(abs(c[k] - v[k]) < 1e-9 for k in KEYS)
        line += ' | el loop decia %s -> %s' % (
            ', '.join('%.3f' % c[k] if isinstance(c[k], float) else str(c[k]) for k in KEYS),
            'COINCIDE' if same else 'NO COINCIDE')
        if not same:
            fail.append(tag + ': estos pesos no son los del brazo ' + r['arm'])
    print('  %s %s' % (tag, line))

# la identidad sobre la que se apoya sec:union: dos suites desjuntas, W del todo = min de las
# mitades, las raices suman y el promedio es el promedio de los promedios
for arm in sorted({r['arm'] for r in READS}):
    got = {}
    for (a, kind, w), v in read.items():
        if a == arm and kind.startswith('union-'):
            tier = 'repaired' if 'repaired' in kind else 'base'
            got.setdefault(tier, {})[w] = v
    for tier, half in sorted(got.items()):
        if not all(t in half for t in ('A', 'B', 'U')):
            continue
        u, a, b = half['U'], half['A'], half['B']
        ok = (abs(u['worst_draw_rate'] - min(a['worst_draw_rate'], b['worst_draw_rate'])) < 1e-12
              and abs(u['random_draw_rate_exact'] -
                      (a['random_draw_rate_exact'] + b['random_draw_rate_exact']) / 2) < 1e-12
              and u['n_roots_drawn_worst'] ==
                      a['n_roots_drawn_worst'] + b['n_roots_drawn_worst']
              and u['roots'] == a['roots'] + b['roots'])
        print('  %-16s %s identidad W_U = min(W_A, W_B): %s  (W %.2f/%.2f/%.2f, raices '
              '%d+%d=%d)' % (arm, tier, 'OK' if ok else 'ROTO', a['worst_draw_rate'],
                              b['worst_draw_rate'], u['worst_draw_rate'],
                              a['n_roots_drawn_worst'], b['n_roots_drawn_worst'],
                              u['n_roots_drawn_worst']))
        if not ok:
            fail.append('%s %s: la union no es el minimo de sus dos mitades' % (arm, tier))

# lo que el propio loop certifico en los brazos que este notebook entrena, junto a sus lecturas
for a in [x for x in ARMS if x['kind'] == 'arm-repair']:
    if not os.path.exists(a['out']):
        continue
    d = json.loads(pathlib.Path(a['out']).read_text(encoding='utf-8'))
    fin = d['repair']['final']
    print('  %-16s repair: W %.2f B %.2f D %.3f | rondas %d | historias %d | %.0f s'
          % (a['arm'], fin['worst_draw_rate'], fin['best_draw_rate'],
             fin['random_draw_rate_exact'], len(d['repair']['log']) - 1,
             len(d['repair']['aggregate']), d['repair']['elapsed_s']))

# el recuento que las tablas publican: certificados sobre la suite ciega, separando las tres
# poblaciones - reparados contra A (tab:blind), contra la union (sec:union) y los de catorce
for label, pref in (('tab:blind: reparados contra la suite A', 'repaired-on-B'),
                    ('sec:union: reparados contra la union A+B', 'union-repaired-on-B'),
                    ('nuevos a catorce vacias: reparados contra A', 'arm-read-on-B')):
    rows = [(arm, v) for (arm, kind, w), v in sorted(read.items()) if kind == pref]
    if not rows:
        continue
    k = sum(1 for _a, v in rows if v['worst_draw_rate'] == 1.0)
    lo, hi = wilson(k, len(rows))
    print('  %s: %d de %d llegan a W=1 ciego [%.2f, %.2f]' % (label, k, len(rows), lo, hi))
    for arm, v in rows:
        print('     %-16s W %.2f | B %.3f | D %.3f | %d/%d raices a tablas'
              % (arm, v['worst_draw_rate'], v['best_draw_rate'],
                 v['random_draw_rate_exact'], v['n_roots_drawn_worst'], v['roots']))

if fail:
    print()
    for f in fail:
        print('  NO VALE: ' + f)
    raise SystemExit('una lectura no reproduce a la que el brazo certifico: los pesos de este '
                     'notebook no son los de ese brazo, o las suites no son las dos mitades '
                     'que se afirma que son')
if left:
    print()
    print('  faltan %d jobs; corre la celda pump otra vez' % len(left))
print('descarga final:')
for a in snapshot() or []:
    print('  ', a.name)
'''

SHIP_CODE = '''suelta = ship()
print('%d zip%s al navegador%s' % (len(suelta), '' if len(suelta) == 1 else 's',
      '' if suelta else ': nada nuevo, cada seccion ya fue descargada'))
for a in suelta:
    print('  ', a.name)
'''


# --------------------------------------------------------------------- the literals ----
def _sha(rel):
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def weights_table(ckpts):
    """The WEIGHTS dict, with the arm each checkpoint belongs to beside its hash.

    An arm records the path it saved its own weights to, and that record is the only thing in the
    repository tying a `.pt` on this disk to a run whose certificate is in `data/`. So the tie is
    checked here rather than assumed: a checkpoint no arm names, or one that names a different file,
    stops the build, because a notebook of reads of the wrong weights is worse than no notebook at
    all - it produces a number that looks like a result.
    """
    table = {}
    for rel in ckpts:
        arm = pathlib.Path(rel).parent.name
        base = rel.endswith("c4_gru_base.pt")
        p = REPO / rp.arm_json(arm)
        if not p.is_file():
            raise SystemExit("%s: no archived arm JSON, so nothing says these weights belong "
                             "to a run the paper cites" % arm)
        rec = json.loads(p.read_text(encoding="utf-8"))["repair"]
        named = rec.get("base_checkpoint" if base else "checkpoint", "")
        if named.replace("\\", "/") != rel:
            raise SystemExit("%s: the arm %s records its %s weights at %r, not at %r"
                             % (rel, arm, "base" if base else "repaired", named, rel))
        table[rel] = {
            "arm": arm,
            "arm_json": rp.arm_json(arm),
            "E": rp.DEPTH_OF[arm],
            "tier": "base" if base else "repaired",
            "bytes": (REPO / rel).stat().st_size,
            "sha256": _sha(rel),
        }
    return table


def weights_literal(ckpts):
    return json.dumps(weights_table(ckpts), indent=4)


def weight_chunk(rel):
    """One checkpoint as one cell: a comment that says what it is, then its base64.

    One cell per file, rather than one dict of four megabytes, because the whole document has to be
    loadable in a browser editor - and because a cell is the unit Colab re-runs, so re-sending one
    checkpoint after a runtime restart costs that checkpoint and not the queue.
    """
    blob = base64.b64encode((REPO / rel).read_bytes()).decode()
    lines = ",\n    ".join("'%s'" % blob[i:i + B64_WIDTH] for i in range(0, len(blob), B64_WIDTH))
    meta = weights_table([rel])[rel]
    return ("# %s | brazo %s, %s, %.0f KB | sha256 %s\n_put(%r,\n    [%s,\n    ])"
            % (rel, meta['arm'], meta['tier'], meta['bytes'] / 1024, meta['sha256'][:16], rel,
               lines))


def reads_literal(todo):
    return json.dumps([dict(arm=r['arm'], kind=r['kind'], which=r['suite'], E=r['E'],
                            key=r['key'], gate=r['gate'], ckpt=r['ckpt'],
                            suite_file=r['suite_file'], out=r['out'], needs=[r['ckpt']])
                       for r in todo], indent=4)


def arms_literal(rows):
    return json.dumps([dict(arm=a['arm'], kind=a['kind'], which=a['suite'], E=a['E'],
                            seed=a['seed'], key=a['key'], gate=a['gate'], ckpt=a['ckpt'],
                            suite_file=a['suite_file'], out=a['out'], needs=a['needs'],
                            arm_json=a['arm_json']) for a in rows], indent=4)


def suites_literal(rows):
    """key -> committed suite file, only for the suites this queue actually points at."""
    tags = {}
    for r in rows:
        tags.setdefault(r['key'], r['suite_file'])
    return json.dumps(tags, indent=4)


def suites_e_literal(rows):
    tags = {}
    for r in rows:
        tags.setdefault(r['key'], r['E'])
    return json.dumps(tags, indent=4)


def expected_literal(todo):
    """The certificate each arm logged, so a read of the wrong weights cannot pass for a result."""
    out = {}
    for r in todo:
        exp = rp.expected(r)
        if not exp:
            continue
        tier = 'repaired' if 'repaired' in r['kind'] else 'base'
        out.setdefault(r['arm'], {})[tier] = {
            'suite': exp['suite'],
            'cert': {k: exp['cert'][k] for k in ('worst_draw_rate', 'best_draw_rate',
                                                 'random_draw_rate_exact',
                                                 'n_roots_drawn_worst')},
        }
    return json.dumps(out, indent=4)


def gates_literal(reads, arms):
    """gate -> what its zip holds and what has to land before it opens.

    The fourteen-empty gate carries the produced checkpoints and the two fourteen-cell suites:
    `.gitignore` keeps the weights off GitHub, so that zip is the only place they will ever exist,
    and a suite a runtime sampled has to travel back or the next account reads different positions.
    """
    rows = {}
    for r in reads:
        rows.setdefault(r['gate'], []).append(r['out'])
    for a in arms:
        rows.setdefault(a['gate'], []).append(a['out'])
    out = {}
    for gate, zname in GATES:
        if gate not in rows:
            continue
        depth = gate.replace('end', '')
        extra, globs = [], []
        if gate.startswith('end'):
            extra = [rp.suite('A', int(depth)), rp.suite('B', int(depth))]
            globs = ['runs/c4_end%s_s*/c4_gru*.pt' % depth]
        out[gate] = {'zip': zname, 'outs': sorted(set(rows[gate])), 'extra': extra,
                     'globs': globs}
    return json.dumps(out, indent=4)


def gate_timing(reads, arms):
    """The packer's own comment: which zips this document can open, and when it opens them.

    The honest sentence about packaging depends on whether the queue trains anything, so it is
    written after the queue is: with reads in it they drain before the first repair starts, and with
    none there is no training run for a zip to compete with at all.
    """
    outs = {}
    for row in reads + arms:
        outs.setdefault(row['gate'], set()).add(row['out'])
    names = [g for g, _zip in GATES if g in outs]
    read_gates = [g for g in names if not g.startswith('end')]
    arm_gates = [g for g in names if g.startswith('end')]

    def per(gates):
        return ', '.join('%s con %d salidas' % (g, len(outs[g])) for g in gates)

    emergency = " La celda ship() del final es la salida de emergencia cuando la sesion se acaba."
    if read_gates and arm_gates:
        lead = ("La puerta de lecturas (%s) se vacia" % per(read_gates) if len(read_gates) == 1
                else "Las %d puertas de lecturas (%s) se vacian" % (len(read_gates), per(read_gates)))
        return (("%s antes de que arranque el primer repair, asi que el unico zip que puede abrirse "
                 "durante un entrenamiento es el de las puertas de brazos (%s), y ese espera todas "
                 "sus salidas.") % (lead, per(arm_gates)) + emergency)
    if arm_gates:
        return ("La unica puerta de este notebook es %s: no se abre hasta que su ultima salida esta "
                "en disco - los JSON de cada repair y sus dos lecturas, los pesos que producen y las "
                "dos suites - asi que ningun zip se construye mientras un brazo entrena."
                % per(arm_gates) + emergency)
    return ("Las puertas de este notebook son solo de lecturas (%s): ninguna cae dentro de un "
            "entrenamiento, porque esta cola no entrena nada." % per(read_gates) + emergency)


# ------------------------------------------------------------------------ the document ----
def pairing_state():
    """How many arms the paired control covers, and how many it can never cover, right now.

    Read off the plan rather than remembered: `reads_plan` emits a base row for an arm only once that
    arm has a blind read to pair against, and that row is carryable only if the arm saved its
    pre-repair checkpoint at all. Both numbers move when an archive lands, so prose written from them
    cannot go stale the way a remembered count does.
    """
    by_arm = {}
    for r in rp.reads():
        if r['kind'].startswith('base-on-'):
            by_arm.setdefault(r['arm'], []).append(r)
    paired = unpairable = arms = 0
    for arm, rows in by_arm.items():
        arms += 1
        if all(rp.exists(x['out']) for x in rows):
            paired += 1
        elif not all(rp.exists(x['ckpt']) for x in rows):
            unpairable += 1
    return paired, unpairable, arms


def _depths(rows):
    """The empty-cell depths this queue actually points at, read off the suite keys."""
    return sorted({int(r['key'][1:]) for r in rows})


def _verify_paragraph(n_read, has_union):
    """Section 5, counted from the checks the results cell will really run.

    The union identity block only fires for an arm whose three reads on A, B and U are in the queue,
    so a document that carries no union read must not tell the reviewer to expect it - a check that
    silently prints nothing is indistinguishable from a check that passed.
    """
    items = [("a read on the suite its arm certified over has to equal the certificate in that arm's "
              "own JSON - worst case, best case, mean, roots drawn - because that is the only way to "
              "tell \"the memo-free recursion agrees\" from \"the notebook read somebody else's "
              "checkpoint\"" +
              ("; for the arms this notebook trains, the comparison runs against the repair JSON the "
               "runtime itself just wrote." if not n_read else "."))]
    if has_union:
        items.append("the identity `sec:union` rests on: for two board-disjoint suites the whole's "
                     "worst case is the minimum of the halves, its mean is the mean of their means, "
                     "and its drawn roots add up.")
    items.append("the counts the tables publish, with Wilson intervals: certifications on the blind "
                 "suite, reported separately for %s."
                 % ('the arms repaired against suite A, the arms repaired against the union, and the '
                    'new fourteen-empty ones' if has_union else
                    'the arms repaired against suite A and the new fourteen-empty ones - the union '
                    'population prints nothing here, because no arm certified over `A + B` is left '
                    'in this queue'))
    lead = ['First', 'Second', 'Third']
    text = "%d things are verified rather than printed. %s" % (
        len(items), ' '.join('%s, %s' % (lead[i], t) for i, t in enumerate(items)))
    return text + (" The cell refuses the first two failures and names the read that broke them."
                   if has_union else
                   " The cell refuses the first failure and names the read that broke it.")


def _pl(n, sing, plur=None):
    """'read' or 'reads' - a queue of one job must not read as a queue of five."""
    return sing if n == 1 else (plur if plur is not None else sing + 's')


def prose_for(reads, arms, ckpts):
    """The sentences that can only be written once the queue is known.

    A document that queues nine training jobs and no reads must not open by promising reads, and one
    that carries no weights must not tell the reader to look for base64. Everything here is derived
    from the plan the queue was built from, so the prose shrinks with the debt instead of lying about
    it.
    """
    n_read, n_arm = len(reads), len(arms)
    n_rep = len([a for a in arms if a['kind'] == 'arm-repair'])
    n_verify = n_arm - n_rep
    paired, unpairable, pairable = pairing_state()
    owed_base = len([r for r in reads if r['kind'].startswith('base-on-')])
    blind, blind_cert = blind_state()
    gate_keys = {r['gate'] for r in reads} | {a['gate'] for a in arms}
    rows = reads + arms
    letters = sorted({r['key'][0] for r in rows})
    has_union = 'U' in letters
    scope = (
        "`tab:blind` reports %d repaired arms and not one of them holds at `W=1` on a "
        "board-disjoint blind suite; that is the paper's scope result, and every one of them is read "
        "by this verifier rather than by the loop that certified it. " % blind
        if blind_cert == 0 else
        "`tab:blind` reports %d repaired arms and %d of them hold at `W=1` on a board-disjoint blind "
        "suite, so the scope result is a rate rather than a wall, and every one of them is read by "
        "this verifier rather than by the loop that certified it. " % (blind, blind_cert))
    closing = union_sentence() + SCALING_SENTENCE
    p = {}
    p['READS_SENTENCE'] = (
        "%d of the list's items %s %s: one call of `src/c4_endgame_verify.py` each, on a "
        "checkpoint that already exists, recomputing that policy's worst-case, best-case and mean draw "
        "rate over a suite of root positions with a recursion that shares no memo with the one that "
        "certified it, then simulating 120 deterministic optimal tie-break orders and 40 adaptive "
        "optimal games per root." % (n_read, 'is a' if n_read == 1 else 'are',
                                     'read' if n_read == 1 else 'reads') if n_read else
        "Nothing on it re-reads weights that already exist: every memo-free read the repository owed "
        "is archived, so the list is training.")
    p['ARMS_SENTENCE'] = (
        "The %s %d items are the arms at fourteen empty cells that no runtime has ever trained - %d "
        "repairs against the published suite, each followed by its own read on that suite and on the "
        "blind one." % ('other' if n_read else 'only', n_arm, n_rep) if n_arm else
        "The other %d items are reads of checkpoints already on this disk." % n_read)
    p['CK_SENTENCE'] = (
        "%d trained %s %s inside this document."
        % (len(ckpts), _pl(len(ckpts), 'checkpoint'), 'rides' if len(ckpts) == 1 else 'ride')
        if ckpts else
        "No checkpoint has to travel inside: each arm fits its own base and its own repair from the "
        "code in the bundle.")
    p['OWED'] = (
        "**Why they are owed.** %s%d of them carry "
        "the paired control - the same seed's *pre-repair* checkpoint, read once per suite - and this "
        "queue adds %d more of those reads, which decide for those seeds too whether the blind fall "
        "was already in the policy or bought by the refit. %s"
        % (scope, paired, owed_base, closing) if owed_base else
        "**Why they are owed.** %s%d of them now "
        "carry the paired control - the same seed's *pre-repair* checkpoint, read once per suite - "
        "and that is every arm whose base weights survive on this disk: the %d that ran before "
        "`c4_gru_base.pt` was ever saved could only be paired by fitting a base again, which is a "
        "training run and not a read. %s"
        % (scope, paired, unpairable, closing))
    p['FETCH_SENTENCE'] = (
        "The weights are inside this notebook as base64 with their hashes, so there is no GitHub "
        "sign-in, no token, nothing to upload and no Drive mount." if ckpts else
        "The code and every root suite ride inside as a hashed bundle, so there is no GitHub sign-in, "
        "no token, nothing to upload and no Drive mount.")
    p['ZIPS'] = "%d %s" % (len(gate_keys), _pl(len(gate_keys), 'zip'))
    p['QUEUE_ORDER'] = (
        "The reads come first because the paper is waiting on them and because packaging competes "
        "with training: by the time the first fourteen-empty repair starts, every read gate has fired "
        "and its zip is in your browser, so the only zip that can open during a training run is the "
        "one that waits for all %d of its outputs."
        % n_arm if n_read else
        "The only zip this document opens is the fourteen-empty one, and it opens when all %d of its "
        "outputs are on disk - %d %s and their %d verifications - so nothing is packaged while an "
        "arm trains, which is the one thing the last ladder run paid wall time for."
        % (n_arm, n_rep, _pl(n_rep, 'repair'), n_verify))
    p['SUITE_SENTENCE'] = (
        "The published suite A, the blind suite B and - for the arms certified over both - the "
        "eighty-position union U, at %s empty cells. " % ' / '.join(str(e) for e in _depths(rows))
        if has_union else
        "The published suite A and the blind suite B, at %s empty cells. "
        % ' / '.join(str(e) for e in _depths(rows)))
    p['UNION_CHECK'] = (
        " It also refuses U unless it is exactly that depth's suite A followed by its suite B, and "
        "that check is what the union certificate means: `U` is not a third sample of endgames, it is "
        "the other two concatenated, so a `W=1` over `U` is a `W=1` over each half." if has_union else
        " No union suite is queued, because every arm certified over `A + B` has already been read "
        "that way, so this cell has no `U` to decompose.")
    p['QUEUE_PARAGRAPH'] = (
        "%d reads, then %d repairs each with its two reads, two jobs at a time. Every job waits for "
        "its checkpoint and for the suite it reads - and a read of a new arm waits for that arm's own "
        "repair JSON and its saved weights - so a queue launched on a runtime where the weights cells "
        "were skipped waits forever instead of quietly reading a file that is not there. "
        % (n_read, n_rep) if n_read else
        "%d jobs, two at a time: %d repairs at fourteen empty cells, each followed by its own two "
        "reads - the repaired policy re-read on the suite it certified over, and on the blind one. "
        "Every job waits for the file before it: a read of a new arm waits for that arm's repair JSON "
        "and for the weights it just trained, so a queue launched on a runtime where the pump was "
        "interrupted waits instead of quietly reading a file that is not there. " % (n_arm, n_rep))
    p['PUMP_OPEN'] = (
        "The pump keeps the runtime alive and returns by itself after @@MINUTES@@ minutes, so the "
        "session is released before Colab takes it away. Interrupt it whenever you like: a job that "
        "finished is on disk, and re-running the cell resumes where it stopped. Only the jobs in "
        "flight restart - minutes for a read, hours for a fourteen-empty repair, which is why the "
        "repairs are at the end of the list rather than interleaved with it." if n_read else
        "The pump keeps the runtime alive and returns by itself after @@MINUTES@@ minutes, so the "
        "session is released before Colab takes it away. Interrupt it whenever you like: a job that "
        "finished is on disk, and re-running the cell resumes where it stopped. Only the job in flight "
        "restarts, and at this depth every job in the list is a repair that takes hours, so a session "
        "that ends mid-repair loses that repair and nothing else.")
    p['VERIFY'] = _verify_paragraph(n_read, has_union)
    p['SHIP_LIST'] = (
        "Every read JSON, every arm JSON, their logs and the fourteen-empty checkpoints are in the "
        "zips your browser took - the checkpoints are in the `end14` one, and they are the only copy "
        "that exists, so that zip matters more than the others." if n_read else
        "The %d arm jobs - %d repairs, %d verifications - their logs and the fourteen-empty "
        "checkpoints they trained are in the `end14` zip your browser took, and those weights are the "
        "only copy that will ever exist, because `*.pt` is not in git, so that zip matters more than "
        "the numbers in it." % (n_arm, n_rep, n_verify))
    return p


def cells_for(reads, arms, ckpts):
    """The document, in the order a reviewer runs it."""
    name = NAME
    setup = bf.scaling_cell('setup').replace('c4_endgame_scaling.ipynb', name)
    tick = bf.scaling_cell('tick').replace('    snapshot(E)\n', '    snapshot()\n')
    pump = bf.scaling_cell('pump')
    pump = "\n".join(l for l in pump.split("\n") if not l.strip().startswith('MAX_MINUTES'))
    pump = pump.replace('t_start = time.time()',
                        'MAX_MINUTES = %d\nt_start = time.time()' % MINUTES, 1)
    pump = pump.replace('    tick()\n', '    tick()\n' + PROGRESS, 1)
    monitor, resource = bf.scaling_cell('monitor'), bf.scaling_cell('resource')
    assert 'snapshot(E)' not in tick, 'the tick cell changed shape; the substitution missed'
    assert 'MAX_MINUTES = %d' % MINUTES in pump, 'the pump cell has no budget line to set'
    assert PROGRESS in pump, 'the pump lost its per-round progress block'

    rows = reads + arms
    budget = rp.arm_budget(rp.ARMS[0][0]) if arms else None
    prose = prose_for(reads, arms, ckpts)
    sub = (lambda t: t.replace('@@SLUG@@', SLUG)
           .replace('@@GATES@@', gates_literal(reads, arms))
           .replace('@@READS@@', reads_literal(reads))
           .replace('@@ARMS@@', arms_literal(arms))
           .replace('@@BUDGET@@', json.dumps(budget, indent=4) if budget else '{}')
           .replace('@@SUITES_E@@', suites_e_literal(rows))
           .replace('@@SUITES@@', suites_literal(rows))
           .replace('@@EXPECTED@@', expected_literal(reads))
           .replace('@@GATE_TIMING@@', gate_timing(reads, arms))
           .replace('@@WEIGHTS@@', weights_literal(ckpts)))

    def say(text):
        for key, sentence in prose.items():
            text = text.replace('@@%s@@' % key, sentence)
        out = (text.replace('@@NREADS@@', str(len(reads)))
               .replace('@@NARMS@@', str(len(arms)))
               .replace('@@NREPAIR@@', str(len([a for a in arms if a['kind'] == 'arm-repair'])))
               .replace('@@NCK@@', str(len(ckpts)))
               .replace('@@MINUTES@@', str(MINUTES)))
        assert '@@' not in out, 'the %s prose has a placeholder nobody fills' % name
        return out

    packer_src, weights_src = sub(PACKER), sub(WEIGHTS)
    for name_, src in (('the packer', packer_src), ('the weights', weights_src)):
        assert '@@' not in src, '%s cell of %s still carries a placeholder' % (name_, NAME)
    cells = [
        ('markdown', say(TITLE.replace('@@TITLE@@', 'una sola cola'))),
        ('code', packer_src),
        ('markdown', S0),
        ('code', setup),
        ('markdown', S1),
        ('code', resource),
        ('markdown', say(S2 if ckpts else S2_EMPTY)),
        ('code', weights_src),
    ]
    for rel in ckpts:
        cells.append(('code', weight_chunk(rel)))
    cells += [
        ('code', WEIGHT_DONE),
        ('markdown', say(S3)),
        ('code', sub(SUITES)),
        ('markdown', say(S4)),
        ('code', sub(QUEUE)),
        ('markdown', SM),
        ('code', tick),
        ('code', monitor),
        ('code', pump),
        ('markdown', say(SP)),
        ('markdown', say(S5)),
        ('code', sub(RESULTS)),
        ('markdown', S6),
        ('code', SHIP_CODE),
        ('markdown', say(S7)),
    ]
    return cells


def plan():
    """What the repository owes right now: every carryable read, then every untrained arm.

    Reads whose checkpoint was never saved are not here, because a notebook cannot carry a file that
    does not exist - the audit prints them as the debt a training run would have to pay.
    """
    order = [gate for gate, _zip in GATES]
    reads = [r for r in rp.pending() if rp.exists(r["ckpt"])]
    reads.sort(key=lambda r: order.index(r["gate"]))
    return reads, rp.arm_rows()


def write(check_only=False):
    path = OUT_DIR / NAME
    reads, arms = plan()
    if not reads and not arms:
        raise SystemExit("nothing is pending: every read is archived and every planned arm has "
                         "run, so the notebook has no queue to write")
    ckpts = rp.checkpoints(reads)
    nb = bf.render(cells_for(reads, arms, ckpts))
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        assert "@@" not in src, "%s cell %d still holds a placeholder" % (path.name, i)
        if c["cell_type"] == "code":
            compile(bf._runnable(src), "%s cell %d" % (path.name, i), "exec")
    if check_only:
        if not path.exists():
            raise SystemExit("%s is missing; run python colab/build_reads_notebooks.py" % path)
        on_disk = [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
                   if "BUNDLE_SHA256 = " not in "".join(c["source"])]
        # the bundle cell is injected by colab/build_bundle.py after these cells are written
        same = on_disk == nb["cells"]
        print("%s: %s" % (path.name, "matches its generator" if same
                          else "DIFFERS from colab/build_reads_notebooks.py"))
        return 0 if same else 1
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("%s: %d cells, %d reads + %d arm jobs, %d checkpoints (%.2f MB de pesos)"
          % (path.name, len(nb["cells"]), len(reads), len(arms), len(ckpts),
             sum((REPO / c).stat().st_size for c in ckpts) / 2 ** 20))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the notebook no longer matches this generator")
    args = ap.parse_args()
    rc = write(args.check)
    if not args.check:
        rc |= build_bundle.build_for(OUT_DIR / NAME)
        print("verify with: python -m pytest tests/test_reads_notebooks.py -q")
    return rc


if __name__ == "__main__":
    sys.exit(main())
