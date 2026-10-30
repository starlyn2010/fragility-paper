#!/usr/bin/env python3
"""rps_game.py — hidden-type repeated Rock-Paper-Scissors with an exact
Bayes oracle and certificate.

Six deterministic teacher types are possible:

    0 always Rock        1 always Paper       2 always Scissors
    3 copy last L move   4 counter last L     5 win-stay-lose-shift

The learner observes only the actions actually played, so its information
state is the history of (learner, teacher) pairs and its posterior is uniform
over the types consistent with that history.  The oracle computes, by exact
backward induction, the Bayes-optimal value V* of every history and the value
V_min of the worst adaptive policy.  A deterministic learner pi is certified
by walking its own reachable frontier:

    W = (V(pi) - V_min) / (V* - V_min)

is 1 exactly when pi is Bayes-optimal.  The horizon is finite (default 8
rounds).  Payoffs are from the learner's side: +1 win, 0 tie, -1 loss.
"""
from __future__ import annotations

from functools import lru_cache

N_ACTIONS = 3
N_TYPES = 6

ALWAYS_ROCK, ALWAYS_PAPER, ALWAYS_SCISSORS = 0, 1, 2
COPY, COUNTER, WSLS = 3, 4, 5

TYPE_NAMES = ("always-R", "always-P", "always-S", "copy", "counter", "wsls")


def beats(a: int, b: int) -> bool:
    """True iff action a beats action b."""
    return (a - b) % 3 == 1


def counter(a: int) -> int:
    """The action that beats a."""
    return (a + 1) % 3


def payoff(a: int, b: int) -> int:
    if a == b:
        return 0
    return 1 if beats(a, b) else -1


def teacher_action(t: int, h: tuple) -> int:
    """Deterministic action of hidden type t after history h."""
    if t <= 2:
        return t
    if not h:
        return 0  # every memory type opens with Rock
    last_l, last_t = h[-1]
    if t == COPY:
        return last_l
    if t == COUNTER:
        return counter(last_l)
    # win-stay-lose-shift on the teacher's own action
    if beats(last_t, last_l):
        return last_t
    return (last_t + 1) % 3


@lru_cache(maxsize=None)
def consistent(t: int, h: tuple) -> bool:
    """True iff type t reproduces every teacher action recorded in h."""
    for k in range(len(h)):
        if teacher_action(t, h[:k]) != h[k][1]:
            return False
    return True


@lru_cache(maxsize=None)
def posterior(h: tuple) -> tuple:
    """Types consistent with h, in increasing order."""
    return tuple(t for t in range(N_TYPES) if consistent(t, h))


@lru_cache(maxsize=None)
def extend(h: tuple, a: int) -> tuple:
    """Observed next histories after learner action a, with posterior mass."""
    types = posterior(h)
    acc: dict[tuple, float] = {}
    w = 1.0 / len(types)
    for t in types:
        h2 = h + ((a, teacher_action(t, h)),)
        acc[h2] = acc.get(h2, 0.0) + w
    return tuple(sorted(acc.items(), key=lambda kv: kv[0]))


class Oracle:
    """Exact Bayes-adaptive value over every history reachable from empty."""

    def __init__(self, horizon: int = 8) -> None:
        self.horizon = horizon
        self._v: dict[tuple, float] = {}
        self._q: dict[tuple, tuple] = {}
        self._vmin: dict[tuple, float] = {}

    def q_values(self, h: tuple) -> tuple:
        hit = self._q.get(h)
        if hit is not None:
            return hit
        if len(h) >= self.horizon:
            self._q[h] = (0.0, 0.0, 0.0)
            return self._q[h]
        qs = []
        for a in range(N_ACTIONS):
            v = 0.0
            for h2, w in extend(h, a):
                v += w * (payoff(a, h2[-1][1]) + self.value(h2))
            qs.append(v)
        self._q[h] = tuple(qs)
        return self._q[h]

    def value(self, h: tuple) -> float:
        hit = self._v.get(h)
        if hit is not None:
            return hit
        v = max(self.q_values(h))
        self._v[h] = v
        return v

    def optimal(self, h: tuple, tol: float = 1e-9) -> tuple:
        qs = self.q_values(h)
        m = max(qs)
        return tuple(a for a, q in enumerate(qs) if q >= m - tol)

    def value_min(self, h: tuple) -> float:
        hit = self._vmin.get(h)
        if hit is not None:
            return hit
        if len(h) >= self.horizon:
            self._vmin[h] = 0.0
            return 0.0
        v = min(sum(w * (payoff(a, h2[-1][1]) + self.value_min(h2))
                    for h2, w in extend(h, a))
                for a in range(N_ACTIONS))
        self._vmin[h] = v
        return v

    def span(self) -> float:
        return self.value(()) - self.value_min(())


class KnownType:
    """Exact optimal value when the type is observed (an upper bound)."""

    def __init__(self, t: int, horizon: int) -> None:
        self.t = t
        self.horizon = horizon
        self._v: dict[tuple, float] = {}

    def value(self, h: tuple) -> float:
        hit = self._v.get(h)
        if hit is not None:
            return hit
        if len(h) >= self.horizon:
            self._v[h] = 0.0
            return 0.0
        ta = teacher_action(self.t, h)
        best = max(payoff(a, ta) + self.value(h + ((a, ta),))
                   for a in range(N_ACTIONS))
        self._v[h] = best
        return best

    def optimal(self, h: tuple, tol: float = 1e-9) -> tuple:
        ta = teacher_action(self.t, h)
        vals = [payoff(a, ta) + self.value(h + ((a, ta),))
                for a in range(N_ACTIONS)]
        m = max(vals)
        return tuple(a for a, v in enumerate(vals) if v >= m - tol)


def bayes_policy(oracle: Oracle):
    """A canonical deterministic Bayes-optimal policy (smallest argmax)."""
    return lambda h: oracle.optimal(h)[0]


def rollout(policy, t: int, horizon: int) -> int:
    """Total payoff of a deterministic learner against a fixed type."""
    h, total = (), 0
    while len(h) < horizon:
        a = int(policy(h))
        ta = teacher_action(t, h)
        total += payoff(a, ta)
        h = h + ((a, ta),)
    return total


def per_type(policy, horizon: int) -> list[int]:
    return [rollout(policy, t, horizon) for t in range(N_TYPES)]


def certificate(policy, oracle: Oracle, actions: dict | None = None) -> dict:
    """Exact value of a deterministic learner and its frontier violations.

    `actions` may map every reachable history to the policy's action, computed
    in batches; the recursion then avoids one network call per frontier state,
    which matters when a bad policy keeps the belief broad and its frontier
    grows to the thousands.
    """
    horizon = oracle.horizon
    memo: dict[tuple, float] = {}
    violations: list[dict] = []
    frontier: set[tuple] = set()
    act = (lambda h: actions[h]) if actions is not None else policy

    def rec(h: tuple) -> float:
        hit = memo.get(h)
        if hit is not None:
            return hit
        if len(h) >= horizon:
            memo[h] = 0.0
            return 0.0
        frontier.add(h)
        a = int(act(h))
        opts = oracle.optimal(h)
        if a not in opts:
            qs = oracle.q_values(h)
            violations.append({
                "history": h,
                "action": a,
                "optimal": opts,
                "gap": max(qs) - qs[a],
            })
        v = 0.0
        for h2, w in extend(h, a):
            v += w * (payoff(a, h2[-1][1]) + rec(h2))
        memo[h] = v
        return v

    value = rec(())
    vstar = oracle.value(())
    vmin = oracle.value_min(())
    span = vstar - vmin
    normalized = (value - vmin) / span if span > 0 else 1.0
    return {
        "value": value,
        "bayes_value": vstar,
        "min_value": vmin,
        "normalized": normalized,
        "n_frontier": len(frontier),
        "n_violations": len(violations),
        "violations": violations,
        "per_type": per_type(act, horizon),
    }


def all_histories(horizon: int) -> list[tuple]:
    """Every history reachable from empty under some learner action."""
    seen: set[tuple] = set()
    stack = [()]
    while stack:
        h = stack.pop()
        if h in seen or len(h) >= horizon:
            continue
        seen.add(h)
        for a in range(N_ACTIONS):
            for h2, _ in extend(h, a):
                if h2 not in seen:
                    stack.append(h2)
    return sorted(seen, key=lambda h: (len(h), h))


def copy_tree(horizon: int) -> list[tuple]:
    """Learner-turn histories when every learner action is tried against type COPY."""
    out: list[tuple] = []
    stack = [()]
    while stack:
        h = stack.pop()
        out.append(h)
        if len(h) >= horizon:
            continue
        for a in range(N_ACTIONS):
            stack.append(h + ((a, teacher_action(COPY, h)),))
    return sorted(out, key=lambda h: (len(h), h))


def optimal_tree(oracle: Oracle, horizon: int) -> list[tuple]:
    """Histories reached when the learner follows a canonical optimal policy."""
    out: list[tuple] = []
    stack = [()]
    seen: set[tuple] = set()
    while stack:
        h = stack.pop()
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
        if len(h) >= horizon:
            continue
        a = oracle.optimal(h)[0]
        for h2, _ in extend(h, a):
            if h2 not in seen:
                stack.append(h2)
    return sorted(out, key=lambda h: (len(h), h))


def brute_force_value(horizon: int = 2) -> dict:
    """Max/min value over every deterministic policy mapping histories to actions.

    Only feasible for tiny horizons; used by the tests to validate the DP.
    """
    import itertools

    hist = [h for h in all_histories(horizon) if len(h) < horizon]
    idx = {h: i for i, h in enumerate(hist)}
    best, worst = -10**9, 10**9

    def evaluate(actions: tuple) -> float:
        total = 0.0
        for t in range(N_TYPES):
            h, v = (), 0
            while len(h) < horizon:
                a = actions[idx[h]]
                ta = teacher_action(t, h)
                v += payoff(a, ta)
                h = h + ((a, ta),)
            total += v
        return total / N_TYPES

    for actions in itertools.product(range(N_ACTIONS), repeat=len(hist)):
        v = evaluate(actions)
        best = max(best, v)
        worst = min(worst, v)
    return {"max": best, "min": worst, "n_histories": len(hist)}


if __name__ == "__main__":
    o = Oracle(8)
    print("V*", o.value(()), "Vmin", o.value_min(()), "span", o.span())
    print("Q* at root", [round(q, 4) for q in o.q_values(())])
    print("optimal at root", o.optimal(()))
    print("known-type optima", [KnownType(t, 8).value(()) for t in range(N_TYPES)])
    print("bayes per-type", per_type(bayes_policy(o), 8))
    print("histories", len(all_histories(8)), "copy tree", len(copy_tree(8)),
          "opt tree", len(optimal_tree(o, 8)))
