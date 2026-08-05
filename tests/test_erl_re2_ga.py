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


def test_layer_scaled_mutation_changes_bias_and_freezes_log_std():
    w = _dummy_weights()
    w['actor.4.bias'].fill(0.0)
    w['actor.log_std.weight'] = np.ones((2, 4), dtype=np.float32)
    w['actor.log_std.bias'] = np.ones(2, dtype=np.float32)
    before_log_std = {
        key: value.copy()
        for key, value in w.items()
        if key.startswith('actor.log_std.')
    }
    cfg = Er2GaConfig(
        mutation_alpha=2.0,
        mutation_beta_frac=1.0,
        mutation_prob=1.0,
        mutation_scale_mode='layer_rms',
        mutation_scale_floor=0.05,
        mutate_bias=True,
        actor_exclude_substrings=('actor.log_std.',),
    )
    b_mutate_inplace(w, cfg, rng=__import__('random').Random(5))
    assert not np.allclose(w['actor.4.bias'], 0.0)
    for key, value in before_log_std.items():
        assert np.array_equal(w[key], value)


def test_crossover_freezes_excluded_log_std():
    w1 = _dummy_weights()
    w2 = _dummy_weights()
    w1['actor.log_std.weight'] = np.ones((2, 4), dtype=np.float32)
    w1['actor.log_std.bias'] = np.ones(2, dtype=np.float32)
    w2['actor.log_std.weight'] = -np.ones((2, 4), dtype=np.float32)
    w2['actor.log_std.bias'] = -np.ones(2, dtype=np.float32)
    before1 = w1['actor.log_std.weight'].copy()
    before2 = w2['actor.log_std.weight'].copy()
    b_crossover_inplace(
        w1,
        w2,
        rng=__import__('random').Random(3),
        exclude_substrings=('actor.log_std.',),
    )
    assert np.array_equal(w1['actor.log_std.weight'], before1)
    assert np.array_equal(w2['actor.log_std.weight'], before2)
