"""oracle.py — Valor minimax exacto desde la perspectiva del agente.

El entorno usa AGENTE=2 y HUMANO=1. Para evaluar una decision del agente en un
tablero dado, se refleja el tablero (1<->2); asi el agente original pasa a ser
HUMANO y se puede reutilizar StochasticMinimax(profundidad 9) como oraculo.
"""
from __future__ import annotations

from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv


def env_from_board(board: tuple[int, ...]) -> TicTacToeEnv:
    e = TicTacToeEnv()
    e.grid = [list(board[0:3]), list(board[3:6]), list(board[6:9])]
    e._winner = e._check_winner()
    e.done = e._winner != e.VACIO or not any(e.VACIO in row for row in e.grid)
    return e


def mirrored(env: TicTacToeEnv) -> TicTacToeEnv:
    m = env.clone()
    flip = {0: 0, 1: 2, 2: 1}
    m.grid = [[flip[v] for v in row] for row in env.grid]
    m._winner = m._check_winner()
    return m


class Oracle:
    """Cache de valores por accion y conjunto de acciones optimas."""

    def __init__(self) -> None:
        self.mentor = StochasticMinimax(mentor_player=-1, depth=9, noise_prob=0.0, tie_break="first")
        self.cache: dict[tuple[int, ...], dict] = {}

    def lookup(self, board: tuple[int, ...]) -> dict:
        hit = self.cache.get(board)
        if hit is not None:
            return hit
        env = env_from_board(board)
        m = mirrored(env)
        values = {}
        for a in env.legal_actions():
            c = m.clone()
            c.step(a, player=c.HUMANO)
            values[a] = float(self.mentor._minimax_exact(c, 9, False))
        best = max(values.values()) if values else 0.0
        optimal = tuple(a for a, v in values.items() if v == best)
        out = {"values": values, "best": best, "optimal": optimal}
        self.cache[board] = out
        return out
