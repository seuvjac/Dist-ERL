"""Tests for federated aggregation helpers."""

import numpy as np
from types import SimpleNamespace

from src.federated import aggregate_weight_dicts, blend_actor_update, weight_entropy
from src.config import (
    FED_ABLATION_NO_LOCAL_RL,
    FED_ABLATION_RAW_SOFTMAX,
    FED_EVO_RL,
)
from src.main import (
    _accepted_client_uploads,
    _aggregation_scores_from_rewards,
    _apply_fed_ablation_args,
)


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


def test_softmax_preserves_explicit_score_scale():
    client_weights = [
        {'actor.0.weight': np.array([0.0], dtype=np.float32)},
        {'actor.0.weight': np.array([2.0], dtype=np.float32)},
    ]
    low_pressure = aggregate_weight_dicts(
        client_weights, scores=[0.0, 1.0], mode='softmax', temperature=1.0)
    high_pressure = aggregate_weight_dicts(
        client_weights, scores=[0.0, 4.0], mode='softmax', temperature=1.0)
    assert high_pressure['actor.0.weight'][0] > low_pressure['actor.0.weight'][0]
    assert high_pressure['actor.0.weight'][0] > 1.9


def test_delta_clip_applies_to_whole_client_update():
    client = {
        'actor.a': np.asarray([1.0], dtype=np.float32),
        'actor.b': np.asarray([1.0], dtype=np.float32),
    }
    base = {
        'actor.a': np.asarray([0.0], dtype=np.float32),
        'actor.b': np.asarray([0.0], dtype=np.float32),
    }

    aggregated = aggregate_weight_dicts(
        [client], [1.0], mode='uniform', base_weights=base, delta_clip_norm=1.0)

    expected = 1.0 / np.sqrt(2.0)
    assert np.allclose(aggregated['actor.a'], expected)
    assert np.allclose(aggregated['actor.b'], expected)
    combined_norm = np.sqrt(sum(float(value[0] ** 2) for value in aggregated.values()))
    assert np.isclose(combined_norm, 1.0)


def test_client_upload_blend_limits_local_actor_drift():
    base = {'actor.weight': np.asarray([0.0, 2.0], dtype=np.float32)}
    candidate = {'actor.weight': np.asarray([2.0, -2.0], dtype=np.float32)}

    blended = blend_actor_update(base, candidate, 0.25)

    assert np.allclose(blended['actor.weight'], [0.5, 1.0])


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


def test_rejected_local_actor_is_not_aggregated_as_a_server_update():
    indices = np.asarray([0, 1, 2])
    results = [
        {'candidate_accepted': 0, 'upload_score': 10.0, 'avg_reward': 1.0,
         'weights': {'actor.weight': np.asarray([0.0])}},
        {'candidate_accepted': 1, 'upload_score': 12.0, 'avg_reward': 2.0,
         'weights': {'actor.weight': np.asarray([1.0])}},
        {'candidate_accepted': 0, 'upload_score': 11.0, 'avg_reward': 3.0,
         'weights': {'actor.weight': np.asarray([2.0])}},
    ]

    accepted_indices, rewards, weights = _accepted_client_uploads(indices, results)

    assert accepted_indices == [1]
    assert rewards == [12.0]
    assert len(weights) == 1
    assert weights[0]['actor.weight'][0] == 1.0


def test_no_local_rl_is_a_pure_ea_ablation():
    args = SimpleNamespace(
        mode=FED_EVO_RL,
        fed_ablation=FED_ABLATION_NO_LOCAL_RL,
        client_updates=64,
        client_rollouts=2,
        client_validation_episodes=1,
        migration_copies=3,
    )
    _apply_fed_ablation_args(args)
    assert args.client_updates == 0
    assert args.client_rollouts == 0
    assert args.client_validation_episodes == 0
    assert args.migration_copies == 0


def test_raw_softmax_ablation_uses_its_own_temperature():
    args = SimpleNamespace(
        mode=FED_EVO_RL,
        fed_ablation=FED_ABLATION_RAW_SOFTMAX,
        fed_aggregation='softmax',
        fed_score_normalization='relative_gain',
        fed_aggregation_temperature=4.0,
        fed_raw_softmax_temperature=15.0,
        fed_score_warmup_rounds=2,
        fed_injection_warmup_rounds=2,
    )
    _apply_fed_ablation_args(args)
    assert args.fed_score_normalization == 'raw'
    assert args.fed_aggregation_temperature == 15.0
    assert args.fed_score_warmup_rounds == 0
