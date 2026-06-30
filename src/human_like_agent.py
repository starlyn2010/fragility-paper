#!/usr/bin/env python3
"""
human_like_agent.py — Oponente con racionalidad limitada (Bounded Rationality).
Implementa un Minimax con:
  - Profundidad asimétrica (2-3 normal, 4-5 en tensión)
  - Ruido gaussiano N(0, σ²) en la evaluación del tablero
  - σ aumenta con la profundidad de búsqueda (fatiga cognitiva)

Uso:
  from human_like_agent import BoundedRationalMinimax
  opp = BoundedRationalMinimax(depth_normal=2, depth_tension=4, noise_std=0.5)
  action = opp.get_action(env, legal)
"""
import random
import numpy as np


def _detect_tension(env, player: int) -> float:
    """Retorna 0-1: qué tan tensa está la posición para 'player'."""
    grid = env.grid if hasattr(env, 'grid') else []
    if not grid:
        return 0.0
    n_rows = len(grid)
    n_cols = len(grid[0]) if grid else 0
    total = n_rows * n_cols
    filled = sum(1 for r in range(n_rows) for c in range(n_cols) if grid[r][c] != 0)
    fill_ratio = filled / max(total, 1)

    opponent = 2 if player == 1 else 1
    threats = 0
    # Checkear amenazas inmediatas del oponente (2 en línea con espacio libre)
    for r in range(n_rows):
        for c in range(n_cols):
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                count = 0
                empty = 0
                for k in range(3):
                    rr, cc = r + dr * k, c + dc * k
                    if 0 <= rr < n_rows and 0 <= cc < n_cols:
                        if grid[rr][cc] == opponent:
                            count += 1
                        elif grid[rr][cc] == 0:
                            empty += 1
                if count == 2 and empty == 1:
                    threats += 1
    tension = min(1.0, threats / 3.0 + fill_ratio * 0.5)
    return tension


class BoundedRationalMinimax:
    """
    Minimax con racionalidad limitada (Bounded Rationality).

    Args:
        depth_normal: profundidad en turnos sin presión (default 2)
        depth_tension: profundidad en estados de tensión (default 4)
        noise_std: desviación estándar del ruido gaussiano en evaluación (default 0.5)
        depth_noise_scale: factor de escala del ruido por nivel de profundidad (default 0.2)
        tension_threshold: umbral de tensión para cambiar a depth_tension (default 0.4)
    """

    def __init__(
        self,
        depth_normal: int = 2,
        depth_tension: int = 4,
        noise_std: float = 0.5,
        depth_noise_scale: float = 0.2,
        tension_threshold: float = 0.4,
    ):
        self.depth_normal = depth_normal
        self.depth_tension = depth_tension
        self.noise_std = noise_std
        self.depth_noise_scale = depth_noise_scale
        self.tension_threshold = tension_threshold

    def get_action(self, env, legal: list[int], pity: float = 0.0, depth: int | None = None) -> int:
        if not legal:
            return -1
        if random.random() < pity:
            return random.choice(legal)

        tension = _detect_tension(env, 1)  # desde perspectiva HUMANA
        search_depth = depth or (self.depth_tension if tension > self.tension_threshold else self.depth_normal)

        best_a = legal[0]
        best_val = -float("inf")
        for a in legal:
            c = env.clone()
            c.step(a, player=env.HUMANO)
            val = self._minimax(c, search_depth - 1, -float("inf"), float("inf"), False, search_depth)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    def _minimax(self, env, depth: int, alpha: float, beta: float, maximizing: bool, max_depth: int) -> float:
        done = getattr(env, "done", getattr(env, "terminado", lambda: False)())
        if callable(done):
            done = done()
        legal = env.legal_actions() if hasattr(env, "legal_actions") else []

        if done or depth == 0 or not legal:
            base = self._evaluate(env)
            # Ruido gaussiano que escala con profundidad (fatiga cognitiva)
            noise_scale = self.noise_std * (1.0 + self.depth_noise_scale * (max_depth - depth))
            noise = np.random.normal(0, noise_scale)
            return base + noise

        if maximizing:
            val = -float("inf")
            for a in legal:
                c = env.clone()
                c.step(a, player=env.HUMANO)
                val = max(val, self._minimax(c, depth - 1, alpha, beta, False, max_depth))
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return val
        else:
            val = float("inf")
            for a in legal:
                c = env.clone()
                c.step(a, player=env.AGENTE)
                val = min(val, self._minimax(c, depth - 1, alpha, beta, True, max_depth))
                beta = min(beta, val)
                if alpha >= beta:
                    break
            return val

    def _evaluate(self, env) -> float:
        """Evalúa la posición desde perspectiva HUMANA (-1 es bueno para el agente)."""
        r = env.resultado() if hasattr(env, "resultado") else 0.0
        if r != 0:
            return r * 100.0  # Escalar para que domine sobre el ruido en estados terminales
        return 0.0


if __name__ == "__main__":
    # Test rápido
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from liquid_rl_trainer import TicTacToeEnv

    env = TicTacToeEnv()
    opp = BoundedRationalMinimax(depth_normal=2, depth_tension=4, noise_std=0.5)
    legal = env.legal_actions()
    a = opp.get_action(env, legal)
    print(f"  BoundedRationalMinimax jugó: {a}")
    print("  ✓ Test OK")
