#!/usr/bin/env python3
"""
connect4_env.py — Entorno Connect 4 con la misma interfaz que TicTacToeEnv.
Tablero 7×6, victoria = 4 en línea. Incluye Minimax con alpha-beta pruning.

Uso:
  from connect4_env import Connect4Env
  env = Connect4Env()
  obs = env.reset()
  obs, reward, done, info = env.step(action, player=env.AGENTE)
"""
import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


class Connect4Env:
    VACIO = 0
    HUMANO = 1   # oponente (juega segundo)
    AGENTE = 2   # nuestro agente (juega primero)

    def __init__(self):
        self.ROWS = 6
        self.COLS = 7
        self.reset()

    def reset(self) -> np.ndarray:
        self.grid = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.done = False
        self._winner = self.VACIO
        self._obs_cache = None
        return self._obs()

    def _obs(self) -> np.ndarray:
        """Observación: tablero aplanado + turno"""
        return np.concatenate([self.grid.flatten(), [self._current_turn()]]).astype(np.float32)

    def _current_turn(self) -> int:
        """Retorna 0 si es turno del HUMANO, 1 si es turno del AGENTE"""
        n_agent = (self.grid == self.AGENTE).sum()
        n_human = (self.grid == self.HUMANO).sum()
        return 1 if n_agent <= n_human else 0

    def _col_height(self, col: int) -> int:
        for r in range(self.ROWS - 1, -1, -1):
            if self.grid[r][col] == self.VACIO:
                return r
        return -1

    def legal_actions(self) -> list[int]:
        return [c for c in range(self.COLS) if self.grid[0][c] == self.VACIO]

    def step(self, action: int, player: int | None = None) -> tuple[np.ndarray, float, bool, dict]:
        if player is None:
            player = self.AGENTE
        row = self._col_height(action)
        if row < 0:
            return self._obs(), -1.0, True, {"valid": False}
        self.grid[row][action] = player
        self._winner = self._check_winner()
        full = not any(self.grid[0][c] == self.VACIO for c in range(self.COLS))
        self.done = self._winner != self.VACIO or full
        reward = 0.0 if not self.done else self.resultado()
        return self._obs(), reward, self.done, {"valid": True}

    def _check_winner(self) -> int:
        # Horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                v = self.grid[r][c]
                if v != self.VACIO and all(self.grid[r][c+k] == v for k in range(4)):
                    return v
        # Vertical
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                v = self.grid[r][c]
                if v != self.VACIO and all(self.grid[r+k][c] == v for k in range(4)):
                    return v
        # Diagonal \
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                v = self.grid[r][c]
                if v != self.VACIO and all(self.grid[r+k][c+k] == v for k in range(4)):
                    return v
        # Diagonal /
        for r in range(3, self.ROWS):
            for c in range(self.COLS - 3):
                v = self.grid[r][c]
                if v != self.VACIO and all(self.grid[r-k][c+k] == v for k in range(4)):
                    return v
        return self.VACIO

    def clone(self) -> 'Connect4Env':
        c = Connect4Env()
        c.grid = self.grid.copy()
        c.done = self.done
        c._winner = self._winner
        return c

    def resultado(self) -> int:
        if self._winner == self.AGENTE:
            return 1
        if self._winner == self.HUMANO:
            return -1
        return 0

    def terminado(self) -> bool:
        return self.done


class MinimaxC4:
    """Minimax con alpha-beta para Connect 4."""

    def __init__(self, depth: int = 5, player: int = -1):
        self.depth = depth
        self.player = player

    def evaluate(self, env: Connect4Env) -> float:
        """Evalúa la posición desde la perspectiva del HUMANO (-1 es bueno para el agente)."""
        w = env._check_winner()
        if w == env.AGENTE:
            return -1000.0
        if w == env.HUMANO:
            return 1000.0
        return 0.0

    def _minimax(self, env: Connect4Env, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        done = env.terminado()
        legal = env.legal_actions()
        if done or depth == 0 or not legal:
            return self.evaluate(env)

        if maximizing:
            value = -float('inf')
            for a in legal:
                c = env.clone()
                c.step(a, player=env.HUMANO)
                value = max(value, self._minimax(c, depth-1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for a in legal:
                c = env.clone()
                c.step(a, player=env.AGENTE)
                value = min(value, self._minimax(c, depth-1, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def get_action(self, env: Connect4Env, legal: list[int], pity: float = 0.0, depth: int | None = None) -> int:
        if not legal:
            return -1
        if random.random() < pity:
            return random.choice(legal)
        d = depth or self.depth
        best_a = legal[0]
        best_val = -float('inf')
        for a in legal:
            c = env.clone()
            c.step(a, player=env.HUMANO)
            val = self._minimax(c, d-1, -float('inf'), float('inf'), False)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a


if __name__ == "__main__":
    # Test rápido
    env = Connect4Env()
    print(f"  Connect4 obs_dim={env._obs().shape[0]} (42 celdas + 1 turno)")
    print(f"  Action space: {env.COLS} columnas")
    env.step(3, player=env.AGENTE)
    env.step(0, player=env.HUMANO)
    env.step(3, player=env.AGENTE)
    print(f"  Tablero tras 3 movimientos:\n{env.grid}")
    print(f"  Legales: {env.legal_actions()}")

    m = MinimaxC4(depth=4)
    legal = env.legal_actions()
    a = m.get_action(env, legal, pity=0.0)
    print(f"  Minimax depth=4 juega: columna {a}")
    print("  ✓ Connect4Env listo")
