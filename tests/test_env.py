"""Fast CPU-only tests for TicTacToeEnv.

Run with pytest from the repository root, or directly:
    python tests/test_env.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from liquid_rl_trainer import TicTacToeEnv  # noqa: E402


def _play(env, moves):
    """Apply alternating moves as (action, player) pairs."""
    for action, player in moves:
        env.step(action, player=player)
    return env


def test_initial_state_and_legality():
    env = TicTacToeEnv()
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (9,)
    assert env.legal_actions() == list(range(9))
    assert not env.done
    assert env.resultado() == 0

    env.step(4, player=env.AGENTE)
    assert env.legal_actions() == [0, 1, 2, 3, 5, 6, 7, 8]
    before = np.array(env.grid).copy()
    _, reward, done, info = env.step(4, player=env.HUMANO)
    assert info["valid"] is False
    assert reward == -1.0 and done is True
    assert np.array_equal(np.array(env.grid), before), "illegal move must not change board"


def test_clone_is_independent():
    env = _play(TicTacToeEnv(), [(0, TicTacToeEnv.AGENTE), (4, TicTacToeEnv.HUMANO)])
    clone = env.clone()
    assert clone is not env
    assert np.array_equal(np.array(clone.grid), np.array(env.grid))
    assert clone.legal_actions() == env.legal_actions()
    clone.step(1, player=clone.HUMANO)
    assert env.grid[0][1] == TicTacToeEnv.VACIO
    assert clone.grid[0][1] == TicTacToeEnv.HUMANO

    # done / winner state is copied too
    won = _play(TicTacToeEnv(), [
        (0, TicTacToeEnv.AGENTE), (3, TicTacToeEnv.HUMANO),
        (1, TicTacToeEnv.AGENTE), (4, TicTacToeEnv.HUMANO),
        (2, TicTacToeEnv.AGENTE),
    ])
    won_clone = won.clone()
    assert won.done and won_clone.done
    assert won_clone.resultado() == won.resultado() == 1
    assert won_clone.legal_actions() == won.legal_actions()


def test_result_signs():
    agent_win = _play(TicTacToeEnv(), [
        (0, TicTacToeEnv.AGENTE), (3, TicTacToeEnv.HUMANO),
        (1, TicTacToeEnv.AGENTE), (4, TicTacToeEnv.HUMANO),
        (2, TicTacToeEnv.AGENTE),
    ])
    assert agent_win.done
    assert agent_win.resultado() == 1

    human_win = _play(TicTacToeEnv(), [
        (0, TicTacToeEnv.HUMANO), (3, TicTacToeEnv.AGENTE),
        (1, TicTacToeEnv.HUMANO), (4, TicTacToeEnv.AGENTE),
        (2, TicTacToeEnv.HUMANO),
    ])
    assert human_win.done
    assert human_win.resultado() == -1

    draw = _play(TicTacToeEnv(), [
        (0, TicTacToeEnv.HUMANO), (2, TicTacToeEnv.AGENTE),
        (1, TicTacToeEnv.HUMANO), (4, TicTacToeEnv.AGENTE),
        (5, TicTacToeEnv.HUMANO), (3, TicTacToeEnv.AGENTE),
        (6, TicTacToeEnv.HUMANO), (7, TicTacToeEnv.AGENTE),
        (8, TicTacToeEnv.HUMANO),
    ])
    assert draw.done
    assert draw.legal_actions() == []
    assert draw.resultado() == 0


def _transform_index(index, kind):
    r, c = divmod(index, 3)
    if kind == "identity":
        nr, nc = r, c
    elif kind == "rot90":
        nr, nc = c, 2 - r
    elif kind == "rot180":
        nr, nc = 2 - r, 2 - c
    elif kind == "rot270":
        nr, nc = 2 - c, r
    elif kind == "flip_h":
        nr, nc = r, 2 - c
    elif kind == "flip_v":
        nr, nc = 2 - r, c
    elif kind == "transpose":
        nr, nc = c, r
    elif kind == "anti_transpose":
        nr, nc = 2 - c, 2 - r
    else:  # pragma: no cover
        raise ValueError(kind)
    return nr * 3 + nc


def test_board_symmetries_preserve_winner_and_legality():
    kinds = [
        "identity", "rot90", "rot180", "rot270",
        "flip_h", "flip_v", "transpose", "anti_transpose",
    ]
    # Agent (2) has a diagonal win; human (1) has an unrelated pair.
    grid = [
        [2, 1, 0],
        [0, 2, 1],
        [0, 0, 2],
    ]
    env = TicTacToeEnv()
    env.grid = [row[:] for row in grid]
    env._winner = env._check_winner()
    assert env._winner == env.AGENTE

    for kind in kinds:
        transformed = TicTacToeEnv()
        transformed.grid = [[0, 0, 0] for _ in range(3)]
        for old_index in range(9):
            r, c = divmod(old_index, 3)
            nr, nc = divmod(_transform_index(old_index, kind), 3)
            transformed.grid[nr][nc] = grid[r][c]
        transformed._winner = transformed._check_winner()
        assert transformed._winner == env.AGENTE, kind
        assert transformed.resultado() == 1, kind
        assert transformed.legal_actions() == sorted(
            _transform_index(a, kind) for a in env.legal_actions()
        ), kind


if __name__ == "__main__":
    test_initial_state_and_legality()
    test_clone_is_independent()
    test_result_signs()
    test_board_symmetries_preserve_winner_and_legality()
    print("ok: env legality, clone, result signs, symmetry")
