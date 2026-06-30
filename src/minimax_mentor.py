"""
minimax_mentor.py — Agente Mentor perfecto con factor de piedad para TTT.
Proyecto Identidad Líquida — MBC
"""

import random

VACIO = 0
HUMANO = 1
AGENTE = 2

WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6],
]


class MinimaxMentor:
    """
    Mentor de Tic-Tac-Toe basado en Minimax con poda alfa-beta.

    - 80% de las veces juega perfecto (nunca pierde)
    - 20% de las veces (piedad) comete un error estratégico
      para dar oportunidad de aprendizaje al agente.
    """

    def __init__(self, piedad: float = 0.2, lado: int = HUMANO):
        self.piedad = piedad
        self.lado = lado
        self.oponente = AGENTE if lado == HUMANO else HUMANO

    def elegir_jugada(self, grid: list[list[int]]) -> int | None:
        flat = [cell for row in grid for cell in row]
        legales = [i for i, c in enumerate(flat) if c == VACIO]
        if not legales:
            return None

        if random.random() < self.piedad:
            return self._jugada_suboptima(flat, legales)
        return self._mejor_jugada(flat, legales)

    def _mejor_jugada(self, flat: list[int], legales: list[int]) -> int:
        best_score = -float("inf")
        best_move = legales[0]
        for m in legales:
            flat[m] = self.lado
            score = self._minimax(flat, 0, False, -float("inf"), float("inf"))
            flat[m] = VACIO
            if score > best_score:
                best_score = score
                best_move = m
        return best_move

    def _jugada_suboptima(self, flat: list[int], legales: list[int]) -> int:
        scored = []
        for m in legales:
            flat[m] = self.lado
            score = self._minimax(flat, 0, False, -float("inf"), float("inf"))
            flat[m] = VACIO
            scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        cutoff = max(1, len(scored) * 2 // 5)
        candidates = scored[-cutoff:]
        return random.choice(candidates)[1]

    def _minimax(
        self, flat: list[int], depth: int, is_max: bool,
        alpha: float, beta: float
    ) -> float:
        winner = self._check_winner_flat(flat)
        if winner == self.lado:
            return 10.0 - depth
        if winner == self.oponente:
            return depth - 10.0
        if all(c != VACIO for c in flat):
            return 0.0

        legales = [i for i, c in enumerate(flat) if c == VACIO]
        current = self.lado if is_max else self.oponente

        if is_max:
            max_score = -float("inf")
            for m in legales:
                flat[m] = current
                score = self._minimax(flat, depth + 1, False, alpha, beta)
                flat[m] = VACIO
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_score
        else:
            min_score = float("inf")
            for m in legales:
                flat[m] = current
                score = self._minimax(flat, depth + 1, True, alpha, beta)
                flat[m] = VACIO
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_score

    @staticmethod
    def _check_winner_flat(flat: list[int]) -> int:
        for a, b, c in WIN_LINES:
            if flat[a] == flat[b] == flat[c] != VACIO:
                return flat[a]
        return VACIO
