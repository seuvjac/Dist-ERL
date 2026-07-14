"""Tests for federated aggregation helpers."""

import numpy as np

from src.federated import aggregate_weight_dicts, weight_entropy
from src.main import _aggregation_scores_from_rewards


def test_fitness_weighted_aggregation_prefers_better_client():
    client_weights = [
        {'actor.0.weight': np.array([0.0, 0.0], dtype=np.float32)},
        {'actor.0.weight': np.array([2.0, 2.0], dtype=np.float32)},
    ]
    aggregated = aggregate_weight_dicts(client_weights, scores=[0.0, 10.0], mode='fitness')
    assert np.allclose(aggregated['actor.0.weight'], np.array([2.0, 2.0], dtype=np.float32))


def test_uniform_aggregation_when_scores_tie():
    client_weights = [
        {'actor.0.weight': np.array([0.0, 0.0], dtype=np.float32)},
        {'actor.0.weight': np.array([2.0, 2.0], dtype=np.float32)},
    ]
    aggregated = aggregate_weight_dicts(client_weights, scores=[1.0, 1.0], mode='fitness')
    assert np.allclose(aggregated['actor.0.weight'], np.array([1.0, 1.0], dtype=np.float32))
    assert weight_entropy([1.0, 1.0], mode='fitness') > 0.0


def test_normalized_score_scale_does_not_change_raw_ablation():
    indices = [0, 1]
    rewards = [12.0, 8.0]

    def scores(mode, scale):
        return _aggregation_scores_from_rewards(
            indices,
            rewards,
            np.zeros(2, dtype=np.float64),
            np.ones(2, dtype=np.float64),
            np.ones(2, dtype=np.int64),
            mode,
            0.9,
            1.0,
            scale,
        )

    base = np.asarray(scores('relative_gain', 1.0))
    scaled = np.asarray(scores('relative_gain', 4.0))
    raw = np.asarray(scores('raw', 4.0))

    assert np.allclose(scaled, 4.0 * base)
    assert np.allclose(raw, rewards)
