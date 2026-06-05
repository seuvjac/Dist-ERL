"""Tests for federated aggregation helpers."""

import numpy as np

from src.federated import aggregate_weight_dicts, weight_entropy


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
