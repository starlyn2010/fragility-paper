#!/usr/bin/env python3
"""c4_game.py — Connect Four 4x4 (win 4) with an exact oracle and certificate.

The board is a tuple of 16 ints: 0 empty, 1 learner (O, first), 2 teacher (X).
Everything is exact: minimax over every reachable position, optimal-action
sets, and the worst/best/average draw rate of a deterministic learner against
*every* deterministic optimal teacher (computed by backward induction, not by
sampling tie-breaks).
"""
from __future__ import annotations

import itertools
from functools import lru_cache

ROWS = 4
COLS = 4
K = 4
N = ROWS * COLS
EMPTY = tuple([0] * N)


def configure(rows: int, cols: int, win: int) -> None:
    """Set the board geometry. Must be called before building an Oracle."""
    global ROWS, COLS, K, N, EMPTY
    ROWS, COLS, K = rows, cols, win
    N = ROWS * COLS
    EMPTY = tuple([0] * N)


def drop(board: tuple, col: int, player: int):
    for r in range(ROWS - 1, -1, -1):
        i = r * COLS + col
        if board[i] == 0:
            b = list(board)
            b[i] = player
            return tuple(b)
    return None


def winner(board: tuple) -> int:
    for r in range(ROWS):
        for c in range(COLS):
            v = board[r * COLS + c]
            if v == 0:
                continue
            if c + K <= COLS and all(board[r * COLS + c + k] == v for k in range(K)):
                return v
            if r + K <= ROWS and all(board[(r + k) * COLS + c] == v for k in range(K)):
                return v
            if r + K <= ROWS and c + K <= COLS and all(
                    board[(r + k) * COLS + c + k] == v for k in range(K)):
                return v
            if r + K <= ROWS and c - K + 1 >= 0 and all(
                    board[(r + k) * COLS + c - k] == v for k in range(K)):
                return v
    return 0


def turn(board: tuple) -> int:
    return 1 if board.count(1) <= board.count(2) else 2


def legal(board: tuple) -> tuple:
    return tuple(c for c in range(COLS) if board[c] == 0)


def terminal(board: tuple) -> bool:
    return winner(board) != 0 or not legal(board)


class Oracle:
    """Exact minimax over every reachable position (161,029 states on 4x4)."""

    def __init__(self) -> None:
        self._value: dict[tuple, int] = {}
        self._optimal: dict[tuple, tuple] = {}

    def value(self, board: tuple) -> int:
        v = self._value.get(board)
        if v is not None:
            return v
        w = winner(board)
        if w == 1:
            self._value[board] = 1
            return 1
        if w == 2:
            self._value[board] = -1
            return -1
        acts = legal(board)
        if not acts:
            self._value[board] = 0
            return 0
        p = turn(board)
        best = None
        for c in acts:
            v = self.value(drop(board, c, p))
            if best is None or (v > best if p == 1 else v < best):
                best = v
        self._value[board] = best
        return best

    def optimal(self, board: tuple) -> tuple:
        cached = self._optimal.get(board)
        if cached is not None:
            return cached
        acts = legal(board)
        if not acts:
            self._optimal[board] = ()
            return ()
        p = turn(board)
        vals = [(c, self.value(drop(board, c, p))) for c in acts]
        target = max(v for _, v in vals) if p == 1 else min(v for _, v in vals)
        out = tuple(c for c, v in vals if v == target)
        self._optimal[board] = out
        return out

    def all_boards(self) -> dict[tuple, tuple]:
        """Every reachable board with one canonical (first-found) history."""
        paths = {EMPTY: ()}
        stack = [EMPTY]
        while stack:
            b = stack.pop()
            if terminal(b):
                continue
            p = turn(b)
            for c in legal(b):
                b2 = drop(b, c, p)
                if b2 is not None and b2 not in paths:
                    paths[b2] = paths[b] + (c,)
                    stack.append(b2)
        return paths


def board_of(history: tuple) -> tuple:
    b = EMPTY
    p = 1
    for c in history:
        b = drop(b, c, p)
        p = 3 - p
    return b


def teacher_action(board: tuple, oracle: Oracle, order: tuple) -> int:
    """Deterministic optimal teacher: first optimal column in a fixed order."""
    opts = set(oracle.optimal(board))
    for c in order:
        if c in opts:
            return c
    raise AssertionError("no legal action")


def tree_histories(oracle: Oracle, order: tuple) -> list[tuple]:
    """All learner-turn histories reachable when the teacher uses `order`
    and the learner tries every legal move (the teacher-distribution tree)."""
    out = []
    stack = [(EMPTY, ())]
    seen = set()
    while stack:
        b, h = stack.pop()
        if (b, h) in seen or terminal(b):
            continue
        seen.add((b, h))
        p = turn(b)
        if p == 1:
            out.append(h)
            for c in legal(b):
                stack.append((drop(b, c, 1), h + (c,)))
        else:
            c = teacher_action(b, oracle, order)
            stack.append((drop(b, c, 2), h + (c,)))
    return out


def evaluate_teacher(policy, oracle: Oracle, order: tuple) -> int:
    """Outcome of a deterministic learner against one deterministic teacher."""
    b, h = EMPTY, ()
    while not terminal(b):
        p = turn(b)
        if p == 1:
            a = policy(h)
            if a not in legal(b):
                return 0
        else:
            a = teacher_action(b, oracle, order)
        b = drop(b, a, p)
        h = h + (a,)
    return 1 if winner(b) != 2 else 0


def certify(policy, oracle: Oracle):
    """Exact backward induction over every optimal teacher tie-break.

    Returns W (worst draw rate), B (best), Dbar (uniform tie-break at every
    teacher decision), the number of memoized states, the frontier size, and
    every learner action that is not optimal on the frontier.
    """
    memo: dict[tuple, tuple[float, float, float]] = {}
    violations: dict[tuple, dict] = {}
    frontier: set[tuple] = set()

    def rec(board: tuple, history: tuple):
        key = (board, history)
        hit = memo.get(key)
        if hit is not None:
            return hit
        w = winner(board)
        if w != 0 or not legal(board):
            out = (1.0 if w != 2 else 0.0,) * 3
            memo[key] = out
            return out
        p = turn(board)
        if p == 1:
            frontier.add(key)
            a = policy(history)
            if a not in legal(board):
                out = (0.0, 0.0, 0.0)
                violations[key] = {"history": history, "board": board,
                                   "action": a, "optimal": ()}
                memo[key] = out
                return out
            opts = oracle.optimal(board)
            if a not in opts:
                violations[key] = {"history": history, "board": board,
                                   "action": a, "optimal": opts}
            out = rec(drop(board, a, 1), history + (a,))
            memo[key] = out
            return out
        opts = oracle.optimal(board)
        children = [rec(drop(board, c, 2), history + (c,)) for c in opts]
        out = (min(ch[0] for ch in children),
               max(ch[1] for ch in children),
               sum(ch[2] for ch in children) / len(children))
        memo[key] = out
        return out

    w, b, mean = rec(EMPTY, ())
    return {
        "worst_draw_rate": w,
        "best_draw_rate": b,
        "random_draw_rate_exact": mean,
        "n_states": len(memo),
        "n_frontier": len(frontier),
        "violations": list(violations.values()),
        "n_violations": len(violations),
    }


def optimal_play_histories(oracle: Oracle, limit: int = 100000) -> list[tuple]:
    """Learner-turn histories where both players choose an optimal action."""
    out = []
    stack = [(EMPTY, ())]
    seen = set()
    while stack and len(out) < limit:
        b, h = stack.pop()
        if (b, h) in seen or terminal(b):
            continue
        seen.add((b, h))
        p = turn(b)
        if p == 1:
            out.append(h)
        for c in oracle.optimal(b):
            stack.append((drop(b, c, p), h + (c,)))
    return out


def random_play_histories(n_games: int, seed: int) -> list[tuple]:
    """Learner-turn histories from uniformly random legal play."""
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n_games):
        b, h = EMPTY, ()
        while not terminal(b):
            p = turn(b)
            if p == 1:
                out.append(h)
            a = rng.choice(legal(b))
            b = drop(b, a, p)
            h = h + (a,)
    return out


def teacher_orders() -> list[tuple]:
    return list(itertools.permutations(range(COLS)))
