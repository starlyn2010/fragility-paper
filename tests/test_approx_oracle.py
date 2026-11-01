"""Contract tests for the tunable-tolerance oracle in src/c4_approx_oracle.py.

The paper's approximate-certificate proposition is only testable once epsilon is an
input rather than a side effect, and the two claims the measurements rest on are
properties of the oracle, not of training:

1. grading the payoff (winner takes gamma ** plies) changes the values but never the
   win/draw/loss outcome, so the graded arm measures the same game;
2. the teacher's tolerance class B_eps is exactly the optimal class while eps stays
   under the smallest payoff gap Delta, and opens the moment eps reaches it --- this
   is the quantisation that makes the integer-payoff arm a step function and the
   graded arm a ramp;
3. the per-board perturbation is the same function in every process, because the
   default hash() is salted per interpreter and an oracle that moves between runs is
   not reproducible.

Run with pytest, or directly: python tests/test_approx_oracle.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import c4_game as g
import c4_approx_oracle as ao

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMMA = 0.9


def _setup():
    g.configure(4, 3, 3)
    exact = g.Oracle()
    exact.value(g.EMPTY)
    graded = ao.GradedOracle(GAMMA)
    graded.value(g.EMPTY)
    return exact, graded, list(graded._value.keys())


def test_grading_preserves_the_outcome():
    """Same sign at every board, only a different magnitude.

    A board that is a draw under -1/0/1 payoffs has no winning line for either
    player, so it also has graded value zero; a board that is a win has a positive
    graded value whose size is not 1, which is exactly what makes epsilon a dial.
    """
    exact, graded, boards = _setup()
    for b in boards:
        ve, vg = exact.value(b), graded.value(b)
        assert (vg > 1e-12) == (ve == 1) and (vg < -1e-12) == (ve == -1)
        assert abs(vg) <= 1.0 + 1e-12
        w = g.winner(b)
        if w != 0:
            plies = b.count(1) + b.count(2)
            assert abs(abs(vg) - GAMMA ** plies) < 1e-12
    assert abs(graded.value(g.EMPTY)) < 1e-12


def test_tolerance_class_is_exact_below_the_gap_and_open_above_it():
    """B_eps = A* for eps < Delta, and strictly larger once eps >= Delta."""
    exact, graded, boards = _setup()
    delta = graded.min_gap()
    assert 0 < delta < 1
    for eps in (0.0, delta / 4.0, delta * 0.999):
        teacher = ao.ToleranceTeacher(graded, eps)
        assert all(teacher.optimal(b) == graded.optimal(b) for b in boards
                   if not g.terminal(b) and g.legal(b))
    teacher = ao.ToleranceTeacher(graded, delta)
    opened = [b for b in boards if not g.terminal(b) and g.legal(b)
              and len(teacher.optimal(b)) > len(graded.optimal(b))]
    assert opened, "the class must open at the smallest gap"


def test_integer_payoffs_have_no_middle_regime():
    """With values in {-1,0,1} the gap is 1, so eps has to reach 1 to move anything."""
    exact, _graded, boards = _setup()
    assert min(abs(exact.value(g.drop(b, c, g.turn(b))) - exact.value(b))
               for b in boards if not g.terminal(b) and g.legal(b) for c in g.legal(b)
               if abs(exact.value(g.drop(b, c, g.turn(b))) - exact.value(b)) > 1e-9) == 1.0
    for eps in (0.1, 0.5, 0.99):
        teacher = ao.ToleranceTeacher(exact, eps)
        assert all(teacher.optimal(b) == exact.optimal(b) for b in boards
                   if not g.terminal(b) and g.legal(b))
    teacher = ao.ToleranceTeacher(exact, 1.0)
    widened = [b for b in boards if not g.terminal(b) and g.turn(b) == 2 and g.legal(b)
               and len(teacher.optimal(b)) > len(exact.optimal(b))]
    assert widened, "at eps = 1 every legal move is admitted"


def test_noise_dial_equals_the_measured_epsilon():
    exact, graded, boards = _setup()
    for dial in (0.0, 0.25, 1.0):
        approx = ao.NoiseOracle(exact, dial)
        err = ao.measure_error(exact, approx, boards)
        assert err["epsilon"] <= dial + 1e-9
        if dial:
            assert err["epsilon"] > 0.9 * dial
        else:
            assert err["misrank_rate"] == 0.0


def _child_perturbations():
    code = (
        "import sys, json; sys.path.insert(0, r'%s');"
        "import c4_game as g; g.configure(4, 3, 3);"
        "import c4_approx_oracle as ao;"
        "print(json.dumps([ao._xi(20261101, b) for b in [(0,)*12, (1,0,0,0,2,0,0,0,0,0,0,0)]]))"
    ) % os.path.join(REPO, "src")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=REPO, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_perturbation_is_reproducible_across_processes():
    """hash() is salted per interpreter; the sha256 oracle must not be."""
    sys.path.insert(0, os.path.join(REPO, "src"))
    mine = [ao._xi(20261101, b) for b in [(0,) * 12, (1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0)]]
    theirs = _child_perturbations()
    assert len(mine) == len(theirs)
    for a, b in zip(mine, theirs):
        assert abs(a - b) < 1e-15 and -1.0 <= a <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
