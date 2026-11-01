#!/usr/bin/env python3
"""c4_approx_oracle.py — does the approximate-certificate bound actually bite?

The paper's Proposition (Approximate certificates) says: with an
epsilon-accurate value estimate, repairing the frontier reached against the
epsilon-optimal teachers guarantees no loss as long as 2*epsilon*H < 1, and the
guarantee degrades from "never loses" to "loses at most 2*epsilon*H". Every
published arm uses an exact oracle, so epsilon = 0 and the statement is empty.
This script measures the other side of it.

The approximate oracle is a depth-limited minimax whose leaves are scored by
the exact expectation of uniform random play (a real, cheap surrogate for a
truncated solver or a learned value function). For each search depth we
measure epsilon exactly against the true game values, then run the same
certificate-guided repair with the approximate oracle as both the teacher and
the label source, and finally certify the result twice: against the
epsilon-optimal teacher class (what the proposition promises) and against the
truly optimal class (what you actually wanted).

Usage:
  python src/c4_approx_oracle.py --rows 4 --cols 3 --win 3 --depths 0,1,2,3,4,5,6 \
      --seed 0 --out data/c4_approx_s0.json

The depth mode above is the probe: it shows that horizon truncation does not tune
epsilon smoothly, so it cannot test the threshold 2*epsilon*H = 1. --mode noise and
--mode blend make epsilon the dial instead of a side effect, over a grid that crosses
1/(2H); for each one the run reports the measured epsilon, whether the bound holds,
the size of the teacher class it opens, what repair costs against it, and whether the
policy repaired against the approximate teacher still does not lose exactly.
  python src/c4_approx_oracle.py --mode noise --seed 0 \
      --out data/c4_eps_noise_s0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c4_game as g
from c4_repair import GRUPolicy, items_from, fit, n_optimal_mistakes

ROOT = pathlib.Path(__file__).resolve().parent.parent


class RandomPlayout:
    """Exact expected terminal outcome under uniform random play, in the reference's
    payoff scale (gamma = 1 gives the integer -1/0/1 payoffs)."""

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma
        self._memo: dict[tuple, float] = {}

    def _terminal(self, board: tuple, w: int) -> float:
        if w == 0:
            return 0.0
        g0 = self.gamma ** (board.count(1) + board.count(2))
        return g0 if w == 1 else -g0

    def value(self, board: tuple) -> float:
        hit = self._memo.get(board)
        if hit is not None:
            return hit
        w = g.winner(board)
        if w != 0 or not g.legal(board):
            out = self._terminal(board, w)
        else:
            p = g.turn(board)
            vals = [self.value(g.drop(board, c, p)) for c in g.legal(board)]
            out = sum(vals) / len(vals)
        self._memo[board] = out
        return out


class DepthOracle:
    """Minimax to a fixed depth with random-playout leaves; interface of Oracle."""

    def __init__(self, leaves: RandomPlayout, depth: int) -> None:
        self.leaves = leaves
        self.depth = depth
        self._memo: dict[tuple, float] = {}

    def _v(self, board: tuple, d: int) -> float:
        key = (board, d)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        w = g.winner(board)
        if w != 0 or not g.legal(board):
            out = self.leaves._terminal(board, w)
        elif d <= 0:
            out = self.leaves.value(board)
        else:
            p = g.turn(board)
            vals = [self._v(g.drop(board, c, p), d - 1) for c in g.legal(board)]
            out = max(vals) if p == 1 else min(vals)
        self._memo[key] = out
        return out

    def value(self, board: tuple) -> float:
        return self._v(board, self.depth)

    def q(self, board: tuple, col: int) -> float:
        return self._v(g.drop(board, col, g.turn(board)), self.depth - 1)

    def optimal(self, board: tuple) -> tuple:
        acts = list(g.legal(board))
        vals = [(c, self.q(board, c)) for c in acts]
        best = max(v for _, v in vals) if g.turn(board) == 1 else \
            min(v for _, v in vals)
        return tuple(c for c, v in vals if abs(v - best) <= 1e-12)


class ToleranceTeacher:
    """B_eps(b) = actions within eps of the exact teacher value at b."""

    def __init__(self, exact: g.Oracle, eps: float) -> None:
        self.exact = exact
        self.eps = eps

    def optimal(self, board: tuple) -> tuple:
        v = self.exact.value(board)
        p = g.turn(board)
        out = []
        for c in g.legal(board):
            q = self.exact.value(g.drop(board, c, p))
            if (q >= v - self.eps - 1e-12) if p == 1 else (q <= v + self.eps + 1e-12):
                out.append(c)
        return tuple(out)


class GradedOracle:
    """Exact minimax over a depth-graded payoff: the winner takes gamma ** plies.

    With integer payoffs every exact value is an integer, so the gap between "best
    action" and "second best" is at least 1. A tolerance below 1 then admits nothing
    and a tolerance of 1 admits every legal move: the approximate certificate has no
    middle regime and the bound's safe region is just the exact case. Grading the
    payoff keeps the game's win/draw/loss outcome identical while making the value set
    fine-grained, so epsilon really moves the teacher class and the threshold becomes
    measurable instead of vacuous.
    """

    def __init__(self, gamma: float) -> None:
        self.gamma = gamma
        self._value: dict[tuple, float] = {}
        self._rec = {}

    def _v(self, board: tuple) -> float:
        hit = self._value.get(board)
        if hit is not None:
            return hit
        w = g.winner(board)
        if w != 0 or not g.legal(board):
            out = 0.0
            if w == 1:
                out = self.gamma ** (board.count(1) + board.count(2))
            elif w == 2:
                out = -(self.gamma ** (board.count(1) + board.count(2)))
        else:
            p = g.turn(board)
            vals = [self._v(g.drop(board, c, p)) for c in g.legal(board)]
            out = max(vals) if p == 1 else min(vals)
        self._value[board] = out
        return out

    def value(self, board: tuple) -> float:
        return self._v(board)

    def q(self, board: tuple, col: int) -> float:
        return self._v(g.drop(board, col, g.turn(board)))

    def optimal(self, board: tuple) -> tuple:
        acts = list(g.legal(board))
        vals = [(c, self.q(board, c)) for c in acts]
        best = max(v for _, v in vals) if g.turn(board) == 1 else \
            min(v for _, v in vals)
        return tuple(c for c, v in vals if abs(v - best) <= 1e-12)

    def min_gap(self) -> float:
        """Delta: the smallest nonzero difference between a value and a successor value.

        This is the quantity the proposition's integer step really needs: no loss is
        guaranteed while 2*epsilon*H stays under Delta, not under 1.
        """
        gaps = []
        for b, v in self._value.items():
            if g.terminal(b) or not g.legal(b):
                continue
            p = g.turn(b)
            for c in g.legal(b):
                d = abs(self._v(g.drop(b, c, p)) - v)
                if d > 1e-9:
                    gaps.append(d)
        return min(gaps) if gaps else 0.0


def _xi(salt: int, board: tuple) -> float:
    """A fixed uniform perturbation per board, reproducible across processes.

    Python's builtin hash() is salted per interpreter run, so it cannot be used to
    pin an oracle: two runs of the same configuration must be the same oracle.
    """
    digest = hashlib.sha256(f"{salt}|{board}".encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2 ** 63 - 1.0


class PerturbedOracle:
    """V~(b) = V*(b) + scale * f(b) on interior boards, exact on terminals.

    Endgame payoffs stay exact because the proposition's integer step (D >= 0 at the
    terminal) needs them to; this oracle is a solver that plays the endgame perfectly
    and evaluates the middle approximately, which is what a truncated search or a
    learned critic actually does. `scale` is the dial and the resulting epsilon is
    measured, not assumed.
    """

    def __init__(self, exact: g.Oracle, scale: float) -> None:
        self.exact = exact
        self.scale = scale
        self._memo: dict[tuple, float] = {}

    def value(self, board: tuple) -> float:
        hit = self._memo.get(board)
        if hit is not None:
            return hit
        w = g.winner(board)
        if w != 0 or not g.legal(board):
            out = float(self.exact.value(board))
        else:
            out = self._mix(board)
        self._memo[board] = out
        return out

    def _mix(self, board: tuple) -> float:
        return float(self.exact.value(board))

    def q(self, board: tuple, col: int) -> float:
        return self.value(g.drop(board, col, g.turn(board)))

    def optimal(self, board: tuple) -> tuple:
        acts = list(g.legal(board))
        vals = [(c, self.q(board, c)) for c in acts]
        best = max(v for _, v in vals) if g.turn(board) == 1 else \
            min(v for _, v in vals)
        return tuple(c for c, v in vals if abs(v - best) <= 1e-12)


class NoiseOracle(PerturbedOracle):
    """V~ = V* + alpha * xi(b): a per-board bias of known size, dial = alpha."""

    def __init__(self, exact: g.Oracle, alpha: float, salt: int = 20261101) -> None:
        super().__init__(exact, alpha)
        self.salt = salt

    def _mix(self, board: tuple) -> float:
        return float(self.exact.value(board)) + self.scale * _xi(self.salt, board)


class BlendOracle(PerturbedOracle):
    """V~ = (1-a) V* + a V_random: regression of a critic toward the random-play mean."""

    def __init__(self, exact: g.Oracle, alpha: float, leaves: RandomPlayout) -> None:
        super().__init__(exact, alpha)
        self.leaves = leaves

    def _mix(self, board: tuple) -> float:
        star = float(self.exact.value(board))
        return (1.0 - self.scale) * star + self.scale * self.leaves.value(board)


def measure_error(exact: g.Oracle, approx: DepthOracle, boards):
    """Exact epsilon = max |Qhat - Qstar|, plus the learner misranking rate."""
    eps, abs_errs = 0.0, []
    misrank, learner_states = 0, 0
    worst = None
    for b in boards:
        if g.terminal(b) or not g.legal(b):
            continue
        p = g.turn(b)
        for c in g.legal(b):
            e = abs(approx.q(b, c) - exact.value(g.drop(b, c, p)))
            abs_errs.append(e)
            if e > eps:
                eps, worst = e, {"board": list(b), "action": c,
                                 "qhat": approx.q(b, c),
                                 "qstar": exact.value(g.drop(b, c, p))}
        if p == 1:
            learner_states += 1
            if not set(approx.optimal(b)) & set(exact.optimal(b)):
                misrank += 1
    return {"epsilon": eps,
            "mean_abs_error": statistics.fmean(abs_errs) if abs_errs else 0.0,
            "misrank_rate": misrank / max(learner_states, 1),
            "learner_states": learner_states,
            "worst_state": worst}


def certify_eps(act, learner_oracle, teacher_oracle):
    """Worst/best/uniform draw rate over the teacher class the certificate uses."""
    memo, violations, frontier = {}, {}, set()

    def rec(board, history):
        key = (board, history)
        hit = memo.get(key)
        if hit is not None:
            return hit
        w = g.winner(board)
        if w != 0 or not g.legal(board):
            out = (1.0 if w != 2 else 0.0,) * 3
            memo[key] = out
            return out
        p = g.turn(board)
        if p == 1:
            frontier.add(key)
            a = act(history)
            opts = learner_oracle.optimal(board)
            if a not in opts:
                violations[key] = {"history": list(history), "action": a,
                                   "optimal": list(opts)}
            out = rec(g.drop(board, a, 1), history + (a,))
            memo[key] = out
            return out
        opts = teacher_oracle.optimal(board)
        children = [rec(g.drop(board, c, 2), history + (c,)) for c in opts]
        out = (min(ch[0] for ch in children), max(ch[1] for ch in children),
               sum(ch[2] for ch in children) / len(children))
        memo[key] = out
        return out

    w, b, mean = rec(g.EMPTY, ())
    return {"worst_draw_rate": w, "best_draw_rate": b,
            "random_draw_rate_exact": mean, "n_states": len(memo),
            "n_frontier": len(frontier), "n_violations": len(violations),
            "violations": list(violations.values())}


def certify_value(act, teacher_oracle, ref) -> dict:
    """Worst-case graded value over the epsilon-optimal teacher class.

    The proposition bounds the *loss*, not only its sign: along any play the deficit
    D(s) = V^{pi,tau}(s) - V*(s) never falls under -2*epsilon*H, so with V*(root) = 0
    the worst teacher line must end at value at least -2*epsilon*H. certify_eps reports
    the no-loss rate; this reports the number the bound actually predicts, which is what
    makes the sweep a test rather than an illustration.
    """
    memo = {}

    def rec(board, history):
        key = (board, history)
        hit = memo.get(key)
        if hit is not None:
            return hit
        w = g.winner(board)
        if w != 0 or not g.legal(board):
            out = ref.value(board)
            memo[key] = out
            return out
        if g.turn(board) == 1:
            a = act(history)
            out = rec(g.drop(board, a, 1), history + (a,))
        else:
            out = min(rec(g.drop(board, c, 2), history + (c,))
                      for c in teacher_oracle.optimal(board))
        memo[key] = out
        return out

    worst = rec(g.EMPTY, ())
    return {"worst_value": worst, "n_states": len(memo)}


def teacher_class_growth(exact, teacher, boards):
    """How many teacher states the tolerance opens up beyond the exact ties."""
    widened, total = 0, 0
    for b in boards:
        if g.terminal(b) or g.turn(b) != 2 or not g.legal(b):
            continue
        total += 1
        if len(teacher.optimal(b)) > len(exact.optimal(b)):
            widened += 1
    return {"teacher_states": total, "widened_states": widened}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--win", type=int, default=3)
    ap.add_argument("--depths", default="0,1,2,3,4,5,6")
    ap.add_argument("--mode", default="depth", choices=("depth", "noise", "blend"),
                    help="what the dial means: search depth (a truncated solver), the "
                         "amplitude of a fixed per-board value error, or the weight on "
                         "random-playout values")
    ap.add_argument("--dials", default="",
                    help="dial values; in depth mode this defaults to --depths, and in "
                         "noise/blend mode to a grid that crosses the bound's 1/(2H)")
    ap.add_argument("--ref", default="integer", choices=("integer", "graded"),
                    help="the reference game the tolerance is measured against: the "
                         "published integer payoffs, or payoffs graded by search depth "
                         "so that epsilon has a non-degenerate regime")
    ap.add_argument("--gamma", type=float, default=0.9,
                    help="depth discount of the graded payoff (ref=graded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--emb", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--agg-repeat", type=int, default=60)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "data" / "c4_approx_s0.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    g.configure(args.rows, args.cols, args.win)
    order = tuple(range(g.COLS))
    horizon = g.N
    t_start = time.time()

    exact = g.Oracle()
    exact.value(g.EMPTY)
    if args.ref == "graded":
        ref = GradedOracle(args.gamma)
        ref.value(g.EMPTY)
        delta = ref.min_gap()
    else:
        ref, delta = exact, 1.0
    leaves = RandomPlayout(args.gamma if args.ref == "graded" else 1.0)
    boards = list(ref._value.keys())
    print(f"board {args.rows}x{args.cols} win {args.win}: horizon {horizon}, "
          f"{len(boards)} reachable boards, "
          f"{len(g.tree_histories(ref, order))} teacher-tree histories",
          flush=True)
    print(f"reference {args.ref}: root value {ref.value(g.EMPTY):.4f} | "
          f"smallest value gap {delta:.6f} | critical epsilon "
          f"{delta / (2 * horizon):.6f}", flush=True)

    if args.mode == "depth":
        dials = [int(x) for x in (args.dials or args.depths).split(",") if x.strip()]
        grid_default = False
    else:
        raw = args.dials or "0,0.01,0.02,0.03,0.035,0.04,0.0417,0.045,0.05,0.06,0.08,0.12,0.2"
        dials = [float(x) for x in raw.split(",") if x.strip()]
        grid_default = not args.dials
    print(f"mode {args.mode}: {len(dials)} dials"
          + (" (default grid around the bound's 1/(2H) = "
             f"{1.0 / (2 * horizon):.4f})" if grid_default else ""), flush=True)

    def make(dial):
        if args.mode == "depth":
            return DepthOracle(leaves, dial)
        if args.mode == "noise":
            return NoiseOracle(ref, dial)
        return BlendOracle(ref, dial, leaves)

    rows = []
    for d in dials:
        t0 = time.time()
        approx = make(d)
        err = measure_error(ref, approx, boards)
        eps = err["epsilon"]
        bound = 2 * eps * horizon
        teacher = ToleranceTeacher(ref, eps)

        torch.manual_seed(args.seed)
        policy = GRUPolicy(args.hidden, args.emb)
        replay = items_from(g.tree_histories(approx, order), approx)
        _, fit_eps = fit(policy, replay, args.epochs, args.lr, args.seed,
                         stop_items=replay)

        def act(h, _p=policy):
            return _p.act(h)

        cert = certify_eps(act, approx, teacher)

        log, agg, agg_set = [], [], set()
        for it in range(args.max_iters):
            log.append({"round": it, "worst_draw_rate_eps": cert["worst_draw_rate"],
                        "n_violations": cert["n_violations"],
                        "aggregate": len(agg),
                        "worst_draw_rate_exact": g.certify(act, exact)["worst_draw_rate"]})
            print(f"[{args.mode} dial {d} it {it:2d}] W_eps {cert['worst_draw_rate']:.0f} "
                  f"viol {cert['n_violations']:3d} agg {len(agg):3d} "
                  f"W_exact {log[-1]['worst_draw_rate_exact']:.0f}", flush=True)
            if cert["worst_draw_rate"] == 1.0:
                break
            new = [tuple(v["history"]) for v in cert["violations"]
                   if tuple(v["history"]) not in agg_set]
            if not new:
                break
            for h in new:
                agg.append(h)
                agg_set.add(h)
            agg_items = items_from(agg, approx)
            fit(policy, replay + agg_items * args.agg_repeat, args.epochs,
                args.lr, args.seed + it + 1, stop_items=replay + agg_items)
            cert = certify_eps(act, approx, teacher)

        final_exact = g.certify(act, exact)
        ref_exact = certify_eps(act, ref, ToleranceTeacher(ref, 0.0))
        valued = certify_value(act, teacher, ref)
        row = {"mode": args.mode, "dial": d, "ref": args.ref, "gamma": args.gamma,
               "depth": d if args.mode == "depth" else None,
               "horizon": horizon, "epsilon": eps, "min_gap": delta,
               "two_eps_H": bound, "eps_crit": delta / (2 * horizon),
               "bound_holds": bound < delta,
               "mean_abs_error": err["mean_abs_error"],
               "misrank_rate": err["misrank_rate"],
               "replay_histories": len(replay), "base_fit_epochs": fit_eps,
               "rounds": len(log), "added": len(agg),
               "certified_eps": cert["worst_draw_rate"] == 1.0,
               "W_eps": cert["worst_draw_rate"], "D_eps": cert["random_draw_rate_exact"],
               "W_exact": final_exact["worst_draw_rate"],
               "B_exact": final_exact["best_draw_rate"],
               "D_exact": final_exact["random_draw_rate_exact"],
               "exact_violations": final_exact["n_violations"],
               "W_ref_exact": ref_exact["worst_draw_rate"],
               "ref_violations": ref_exact["n_violations"],
               "worst_value": valued["worst_value"],
               "bound_value": -2 * eps * horizon,
               "bound_respected": valued["worst_value"] >= -2 * eps * horizon - 1e-9,
               "teacher_class": teacher_class_growth(ref, teacher, boards),
               "log": log, "elapsed_s": time.time() - t0}
        rows.append(row)
        print(f"[{args.mode} dial {d}] eps {eps:.4f} 2epsH {bound:.3f} "
              f"({'bound holds' if bound < delta else 'bound silent'}) | "
              f"misrank {err['misrank_rate']:.3f} | rounds {row['rounds']} "
              f"added {row['added']} | W_eps {row['W_eps']:.0f} "
              f"W_ref {row['W_ref_exact']:.0f} viol_ref {row['ref_violations']:3d} "
              f"worstV {row['worst_value']:+.3f} (bound {row['bound_value']:+.3f}) "
              f"{'ok' if row['bound_respected'] else 'VIOLATES BOUND'} | "
              f"W_exact {row['W_exact']:.0f} D_exact {row['D_exact']:.4f} "
              f"[{row['elapsed_s']:.0f}s]", flush=True)

    out = {"args": vars(args),
           "environment": {"python": __import__("platform").python_version(),
                           "torch": torch.__version__},
           "game": {"rows": g.ROWS, "cols": g.COLS, "win": g.K, "horizon": horizon,
                    "reachable_boards": len(boards), "reference": args.ref,
                    "min_gap": delta},
           "leaf_value": "exact expectation of uniform random play",
           "rows": rows, "elapsed_s": time.time() - t_start}
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p} [{time.time()-t_start:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
