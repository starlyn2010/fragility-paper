"""Equivalence and tie-break tests for StochasticMinimax.

Run with pytest, or directly: python tests/test_minimax.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from liquid_rl_trainer import StochasticMinimax, TicTacToeEnv


def _random_reachable_states(n=300, seed=0):
    rng = random.Random(seed)
    env = TicTacToeEnv()
    states = []
    for _ in range(n):
        env.reset()
        while not env.done:
            legal = env.legal_actions()
            if not legal:
                break
            if rng.random() < 0.35:
                states.append(env.clone())
            env.step(rng.choice(legal), player=rng.choice([env.AGENTE, env.HUMANO]))
    return states


def test_cached_value_matches_alpha_beta():
    mentor = StochasticMinimax(depth=6, noise_prob=0.0, tie_break="first")
    checked = 0
    for env in _random_reachable_states(300):
        if env.done:
            continue
        for maximizing in (True, False):
            v_ab = mentor._minimax(env.clone(), 6, -float("inf"), float("inf"), maximizing)
            v_ex = mentor._minimax_exact(env.clone(), 6, maximizing)
            assert v_ab == v_ex, (env.grid, maximizing, v_ab, v_ex)
            checked += 1
    assert checked > 100


def test_optimal_actions_are_optimal_and_tiebreak_uses_them():
    env = TicTacToeEnv()
    mentor = StochasticMinimax(depth=6, noise_prob=0.0, tie_break="random", seed=7)
    value, actions = mentor.optimal_actions(env)
    assert actions, "el tablero vacio debe tener acciones legales"
    for a in actions:
        c = env.clone()
        c.step(a, player=env.HUMANO)
        assert mentor._minimax_exact(c, 6, False) == value
    picks = {mentor.get_action(env, env.legal_actions()) for _ in range(200)}
    assert picks <= set(actions)
    assert picks == set(actions), "TBR debe cubrir todas las acciones optimas"


def test_first_tiebreak_is_deterministic():
    env = TicTacToeEnv()
    a = StochasticMinimax(depth=6, noise_prob=0.0, tie_break="first")
    picks = {a.get_action(env, env.legal_actions()) for _ in range(20)}
    assert len(picks) == 1


if __name__ == "__main__":
    test_cached_value_matches_alpha_beta()
    test_optimal_actions_are_optimal_and_tiebreak_uses_them()
    test_first_tiebreak_is_deterministic()
    print("ok: minimax cache equivalence + tie-break coverage")
