"""Unit tests for ERL-Re² genetic operators."""

import numpy as np

from src.utils.erl_re2_ga import Er2GaConfig, b_crossover_inplace, b_mutate_inplace, erl_re2_epoch
from src.utils.individual import Individual


def _dummy_weights(action_dim=2, hidden=4, state_dim=3):
    return {
        'actor.0.weight': np.random.randn(hidden, state_dim).astype(np.float32),
        'actor.0.bias': np.random.randn(hidden).astype(np.float32),
        'actor.2.weight': np.random.randn(hidden, hidden).astype(np.float32),
        'actor.2.bias': np.random.randn(hidden).astype(np.float32),
        'actor.4.weight': np.random.randn(action_dim, hidden).astype(np.float32),
        'actor.4.bias': np.random.randn(action_dim).astype(np.float32),
    }


def test_erl_re2_epoch_runs():
    pop = [
        Individual(id=i, weights=_dummy_weights(), fitness=float(i), seed=i)
        for i in range(6)
    ]
    cfg = Er2GaConfig(num_elitists=1, mutation_prob=1.0)
    elite_idx, stats = erl_re2_epoch(pop, cfg, rng=__import__('random').Random(0))
    assert elite_idx >= 0
    assert stats['elite'] >= 1.0


def test_crossover_changes_weights():
    w1 = _dummy_weights()
    w2 = _dummy_weights()
    before = w1['actor.4.weight'].copy()
    b_crossover_inplace(w1, w2, rng=__import__('random').Random(1))
    assert not np.allclose(before, w1['actor.4.weight']) or not np.allclose(
        before, w2['actor.4.weight'])


def test_mutate_changes_weights():
    w = _dummy_weights()
    before = w['actor.4.weight'].copy()
    cfg = Er2GaConfig(mutation_alpha=1.0, mutation_beta_frac=1.0, mutation_prob=1.0)
    b_mutate_inplace(w, cfg, rng=__import__('random').Random(2))
    assert not np.allclose(before, w['actor.4.weight'])
