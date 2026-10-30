"""Exactness tests for the hidden-type RPS oracle and certificate.

Run with pytest, or directly: python tests/test_rps.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rps_game as g


def test_payoff_and_counter():
    assert g.beats(1, 0) and g.beats(2, 1) and g.beats(0, 2)
    assert not g.beats(0, 1)
    assert g.payoff(0, 0) == 0
    assert g.payoff(1, 0) == 1
    assert g.payoff(0, 1) == -1
    for a in range(3):
        assert g.beats(g.counter(a), a)


def test_constant_types():
    for t in range(3):
        h = ()
        for _ in range(4):
            a = g.teacher_action(t, h)
            assert a == t
            h = h + ((0, a),)


def test_copy_and_counter():
    h = ((2, 0),)
    assert g.teacher_action(g.COPY, h) == 2
    h = h + ((1, 2),)
    assert g.teacher_action(g.COPY, h) == 1
    h = ((0, 0),)
    assert g.teacher_action(g.COUNTER, h) == 1
    h = h + ((2, 1),)
    assert g.teacher_action(g.COUNTER, h) == 0


def test_wsls():
    assert g.teacher_action(g.WSLS, ()) == 0
    assert g.teacher_action(g.WSLS, ((2, 0),)) == 0   # teacher won, repeat
    assert g.teacher_action(g.WSLS, ((0, 0),)) == 1   # tie, shift
    assert g.teacher_action(g.WSLS, ((1, 0),)) == 1   # learner won, shift
    assert g.teacher_action(g.WSLS, ((0, 1),)) == 1   # teacher won, repeat


def test_posterior_after_first_round():
    assert g.posterior(((0, 0),)) == (0, 3, 4, 5)
    assert g.posterior(((2, 0),)) == (0, 3, 4, 5)
    assert g.posterior(((0, 1),)) == (1,)
    assert g.posterior(((0, 2),)) == (2,)


def test_types_are_pairwise_identifiable():
    import itertools
    horizon = 4
    seqs = list(itertools.product(range(3), repeat=horizon))
    for t1 in range(g.N_TYPES):
        for t2 in range(t1 + 1, g.N_TYPES):
            found = False
            for seq in seqs:
                h, differ = (), False
                for a in seq:
                    if g.teacher_action(t1, h) != g.teacher_action(t2, h):
                        differ = True
                        break
                    h = h + ((a, g.teacher_action(t1, h)),)
                if differ:
                    found = True
                    break
            assert found, f"types {t1} and {t2} are indistinguishable"


def test_extend_is_a_distribution():
    for horizon in (2, 4):
        for h in g.all_histories(horizon):
            for a in range(3):
                ext = g.extend(h, a)
                assert abs(sum(w for _, w in ext) - 1.0) < 1e-12
                for h2, _ in ext:
                    assert g.posterior(h2)


def test_dp_matches_brute_force():
    bf = g.brute_force_value(2)
    oracle = g.Oracle(2)
    assert abs(oracle.value(()) - bf["max"]) < 1e-9
    assert abs(oracle.value_min(()) - bf["min"]) < 1e-9


def test_bayes_policy_is_certified():
    oracle = g.Oracle(5)
    cert = g.certificate(g.bayes_policy(oracle), oracle)
    assert abs(cert["normalized"] - 1.0) < 1e-9
    assert cert["n_violations"] == 0
    assert abs(cert["value"] - oracle.value(())) < 1e-9


def test_constant_policy_is_not_certified():
    oracle = g.Oracle(4)
    cert = g.certificate(lambda h: 0, oracle)
    assert cert["normalized"] < 1.0 - 1e-6
    assert cert["n_violations"] > 0


def test_certificate_value_matches_simulation():
    oracle = g.Oracle(4)

    def policy(h):
        return (sum(l * 7 + t * 13 for l, t in h) + len(h)) % 3

    cert = g.certificate(policy, oracle)
    vals = g.per_type(policy, 4)
    assert cert["per_type"] == vals
    assert abs(cert["value"] - sum(vals) / g.N_TYPES) < 1e-9


def test_action_dict_certificate_matches_callable():
    oracle = g.Oracle(3)

    def policy(h):
        return (sum(l * 5 + t * 11 for l, t in h) + 2 * len(h)) % 3

    acts = {h: policy(h) for h in g.all_histories(3) if len(h) < 3}
    assert g.certificate(None, oracle, acts) == g.certificate(policy, oracle)


def test_known_type_value_is_an_upper_bound():
    oracle = g.Oracle(4)
    for t in range(g.N_TYPES):
        kt = g.KnownType(t, 4)
        assert kt.value(()) == 4
        assert g.rollout(g.bayes_policy(oracle), t, 4) <= kt.value(()) + 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
