"""Merge suite A and suite B into one certified set, so the certificate can cover both.

`tab:blind` says a repair certified against a 40-root suite does not reach $W=1$ on a
board-disjoint one. The claim on the table is about the *scope of the certificate*, so the
matching test is to move the blind positions inside that scope: certify against the union of
the two suites and ask whether the loop still closes. It has to, by construction, if the two
suites really are disjoint: `certify` takes a minimum over roots of a per-root worst case, so
on two suites sharing no board $W_{A\\cup B}=\\min(W_A,W_B)$ and $\\bar D_{A\\cup B}$ is exactly
the mean of the two $\\bar D$---a union certificate at $W=1$ is a certificate at $W=1$ on the
blind suite by definition, not by transfer---and the experiment is not about whether that
identity holds but about what it costs. The published arms already sit at the round cap at
twelve empty cells, so doubling the certified set may simply be out of budget, and the
interesting number is rounds and histories per certified root.

The base fit is affected too: it imitates from positions below the roots, so a union arm
starts from twice the training positions. That is part of what "put the positions in the
certified set" means, and the paired base reads on each suite separately keep it visible
rather than hidden.

    python runs/make_union_suites.py            # writes the two union files
    python runs/make_union_suites.py --check    # exit 1 if a committed union is wrong
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEPTH = (8, 10)


def a_name(E):
    return pathlib.Path("data", "c4_endgame_roots_%d_40_s0.json" % E)


def b_name(E):
    return pathlib.Path("data", "c4_endgame_roots_%d_40_holdoutB.json" % E)


def u_name(E):
    return pathlib.Path("data", "c4_endgame_roots_%d_80_AplusB.json" % E)


def boards(roots):
    return {tuple(r["board"]) for r in roots}


def build(E):
    a = json.loads((ROOT / a_name(E)).read_text(encoding="utf-8"))
    b = json.loads((ROOT / b_name(E)).read_text(encoding="utf-8"))
    overlap = boards(a) & boards(b)
    if overlap:
        raise SystemExit("suites A and B at %d empties share %d boards" % (E, len(overlap)))
    union = a + b
    values = [r["value"] for r in union]
    out = ROOT / u_name(E)
    out.write_text(json.dumps(union, indent=1), encoding="utf-8")
    print("wrote %s: %d roots, value 0 in %d, 1 in %d, -1 in %d"
          % (out.name, len(union), values.count(0), values.count(1), values.count(-1)))
    return 0


def check(E):
    union = json.loads((ROOT / u_name(E)).read_text(encoding="utf-8"))
    a = json.loads((ROOT / a_name(E)).read_text(encoding="utf-8"))
    b = json.loads((ROOT / b_name(E)).read_text(encoding="utf-8"))
    ok = (len(union) == 80 and [tuple(r["board"]) for r in union] ==
          [tuple(r["board"]) for r in a + b])
    print("%s %s" % ("ok  " if ok else "FAIL", u_name(E).as_posix()))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.exit(sum(check(E) if args.check else build(E) for E in DEPTH))
