import numpy as np

import repair
from liquid_rl_trainer import TicTacToeEnv


def _first_legal(board, hist):
    return next(i for i, v in enumerate(board) if v == 0)


def _all_legal(board):
    return [i for i, v in enumerate(board) if v == 0]


def test_dagger_visits_are_learner_turn_histories():
    rng = np.random.default_rng(0)
    visited = repair.dagger_histories(_first_legal, _all_legal, rng, 5)
    assert visited
    for key in visited:
        board = key[-1]
        done, _result, pieces = repair.board_state(board)
        assert not done
        assert pieces % 2 == 0
        assert len(key) == pieces // 2 + 1


def test_add_history_is_idempotent():
    oracle = repair.Oracle()
    aggregate = {}
    key = (repair.EMPTY,)
    assert repair.add_history(aggregate, key, oracle)
    assert not repair.add_history(aggregate, key, oracle)
    assert len(aggregate) == 1
    assert aggregate[key]["optimal"] == list(oracle.lookup(repair.EMPTY)["optimal"])
