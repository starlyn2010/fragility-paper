"""Fast CPU-only tests for RandomAgent and StochasticMinimax (including TBR).

Run with pytest from the repository root, or directly:
    python tests/test_agents.py
"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from liquid_rl_trainer import RandomAgent, StochasticMinimax, TicTacToeEnv  # noqa: E402


def _random_states(n=40, seed=11):
    rng = random.Random(seed)
    env = TicTacToeEnv()
    states = []
    for _ in range(n):
        env.reset()
        while not env.done:
            legal = env.legal_actions()
            if not legal:
                break
            states.append(env.clone())
            env.step(rng.choice(legal), player=rng.choice([env.AGENTE, env.HUMANO]))
    return states


def test_random_agent_is_legal_and_uniform():
    agent = RandomAgent()
    rng = random.Random(3)
    env = TicTacToeEnv()
    for _ in range(100):
        legal = env.legal_actions()
        assert agent.get_action(legal) in legal

    counts = {a: 0 for a in range(9)}
    draws = 9000
    for _ in range(draws):
        counts[agent.get_action(env.legal_actions())] += 1
    expected = draws / 9
    # Very loose bounds (~6 sigma); catches a biased or illegal policy, not noise.
    for action, count in counts.items():
        assert 0.7 * expected < count < 1.3 * expected, (action, counts)


def test_stochastic_minimax_never_returns_illegal_action():
    states = _random_states(40)
    for tie_break, seed in (("first", None), ("random", 5)):
        for depth in (0, 2):
            mentor = StochasticMinimax(
                depth=depth, noise_prob=0.0, tie_break=tie_break, seed=seed
            )
            for env in states:
                legal = env.legal_actions()
                action = mentor.get_action(env, legal)
                assert action in legal, (depth, tie_break, env.grid, action)

    # noise_prob=1.0 takes the random branch; it must still respect legality.
    noisy = StochasticMinimax(depth=2, noise_prob=1.0, tie_break="first")
    for env in states:
        assert noisy.get_action(env, env.legal_actions()) in env.legal_actions()


def test_tbr_covers_all_optimal_actions_on_empty_board():
    env = TicTacToeEnv()
    mentor = StochasticMinimax(depth=6, noise_prob=0.0, tie_break="random", seed=1234)
    value, actions = mentor.optimal_actions(env)
    assert len(actions) == 9, actions
    for action in actions:
        child = env.clone()
        child.step(action, player=env.HUMANO)
        assert mentor._minimax_exact(child, 6, False) == value

    picks = {mentor.get_action(env, env.legal_actions()) for _ in range(300)}
    assert picks == set(actions), "TBR must sample every optimal action"


def test_fixed_tiebreak_is_deterministic_and_optimal():
    env = TicTacToeEnv()
    mentor = StochasticMinimax(depth=6, noise_prob=0.0, tie_break="first")
    _, optimal = mentor.optimal_actions(env)
    picks = {mentor.get_action(env, env.legal_actions()) for _ in range(50)}
    assert len(picks) == 1
    assert picks.pop() in optimal


if __name__ == "__main__":
    test_random_agent_is_legal_and_uniform()
    test_stochastic_minimax_never_returns_illegal_action()
    test_tbr_covers_all_optimal_actions_on_empty_board()
    test_fixed_tiebreak_is_deterministic_and_optimal()
    print("ok: random agent, minimax legality, TBR coverage, fixed tie-break")
