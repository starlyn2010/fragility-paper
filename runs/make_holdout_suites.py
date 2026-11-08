"""Sample the endgame suites the follow-up notebooks read, once, on the laptop.

Why here instead of inside the runtime: `random_games` is deterministic in its sample seed,
so a suite sampled on Colab is the same suite sampled here - but sampling twelve- and
fourteen-empty positions costs the runtime the hour an arm would rather spend training, and
two accounts would each pay it. Sampling once and committing the files makes the holdout
part of the repository: a reviewer re-running a notebook reads the same forty positions the
published arms did, and `colab/build_bundle.py` ships them inside the notebook.

Suite A is the published one at each depth (sample seed 0, the seed every arm already in the
paper used). Suite B is the blind holdout: sample seeds 500 + (empty cells), wave by wave,
keeping only positions that suite A does not already hold. That disjointness is the whole of
the holdout argument, because a repair trains on positions *below* its roots, which have
strictly fewer empty cells.

    python runs/make_holdout_suites.py                 # every depth, nothing overwritten
    python runs/make_holdout_suites.py --depths 8 10   # only these
    python runs/make_holdout_suites.py --check         # exit 1 if a committed suite is wrong
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import c4_game as g                                              # noqa: E402
from c4_endgame import random_games                              # noqa: E402

SIZE = 40                 # both suites are the published size, so arms stay comparable
B_OFFSET = 500            # suite B's first sample seed is 500 + empty cells
B_WAVES = 6               # 500+E .. 505+E, until forty unseen positions are found
PER_GAMES = 400000        # a wave that cannot find forty draw roots is a depth problem,
                          # not a seed problem, so it is reported rather than extended
DEPTHS = (8, 10, 12, 14)


def a_name(E):
    return pathlib.Path("data", "c4_endgame_roots_%d_40_s0.json" % E)


def b_name(E):
    return pathlib.Path("data", "c4_endgame_roots_%d_40_holdoutB.json" % E)


def boards(roots):
    return {tuple(r["board"]) for r in roots}


def check(roots, E, label):
    assert len(roots) == SIZE, "%s holds %d positions, not %d" % (label, len(roots), SIZE)
    for r in roots:
        assert len(r["board"]) == g.N, "%s: not a full 6x7 board" % label
        empties = r["board"].count(0)      # an empty cell is 0; g.EMPTY is the empty *board*
        assert empties == E, "%s: %d empty cells is not depth %d" % (label, empties, E)
        assert not g.terminal(tuple(r["board"])), "%s: already finished" % label
        assert r["value"] == 0, "%s: a root the learner loses is not a draw suite" % label


def grow(path, E, need, exclude, seed0, waves, oracle, label):
    """Sample in waves, checkpointed to a .part file, so an interrupted run resumes."""
    part = pathlib.Path(str(path) + ".part")
    have = json.loads(part.read_text()) if part.exists() else []
    seen = set(exclude) | boards(have)
    wave = 0
    while len(have) < need and wave < waves:
        t0 = time.time()
        fresh, games = random_games(oracle, E, need, seed0 + wave, 0, max_games=PER_GAMES)
        added = 0
        for r in fresh:
            b = tuple(r["board"])
            if len(have) >= need or b in seen:
                continue
            seen.add(b)
            have.append(r)
            added += 1
        wave += 1
        part.write_text(json.dumps(have))
        print("  %s wave %d: %d new draw roots from %d games -> %d/%d [%.0fs]"
              % (label, wave, added, games, len(have), need, time.time() - t0), flush=True)
    if len(have) < need:
        raise SystemExit("%s: only %d of %d unseen draw roots in %d waves at %d empty cells"
                         % (label, len(have), need, waves, E))
    path.write_text(json.dumps(have))
    part.unlink(missing_ok=True)
    return have


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    ap.add_argument("--check", action="store_true",
                    help="exit 1 unless every suite of these depths is on disk and valid")
    args = ap.parse_args()
    g.configure(6, 7, 4)
    os.chdir(ROOT)
    oracle = g.Oracle()
    bad = 0

    for E in args.depths:
        for name, make in ((a_name(E), lambda: grow(a_name(E), E, SIZE, set(), 0, 1,
                                                    oracle, "suite A")),
                           (b_name(E), lambda: grow(b_name(E), E, SIZE, boards(
                               json.loads(a_name(E).read_text())), B_OFFSET + E, B_WAVES,
                               oracle, "suite B"))):
            if not name.exists():
                if args.check:
                    # --check is a question about the repository, not an instruction to
                    # spend an hour sampling: a missing suite is the answer.
                    print("MISSING %s" % name.name)
                    bad += 1
                    continue
                print("%s missing -> sampling" % name)
                make()
            if not name.exists():
                bad += 1
                continue
            roots = json.loads(name.read_text())
            try:
                check(roots, E, name.name)
            except AssertionError as exc:
                print("BAD %s" % exc)
                bad += 1
                continue
            if name == b_name(E):
                shared = boards(roots) & boards(json.loads(a_name(E).read_text()))
                if shared:
                    print("BAD %s shares %d positions with suite A" % (name.name,
                                                                       len(shared)))
                    bad += 1
                    continue
            print("ok %s: %d draw roots of %d empty cells" % (name.name, len(roots), E))
    if bad:
        print("%d suites are missing or wrong" % bad)
        return 1
    if args.check:
        print("every requested suite is on disk, valid, and disjoint from its A counterpart")
    return 0


if __name__ == "__main__":
    sys.exit(main())
