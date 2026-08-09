"""Regression tests for paper plotting helpers."""

import csv
import json

import numpy as np

from scripts.plot_fedrl_heterogeneous import (
    _align_runs_to_first_evaluation,
    _display_env,
    _smooth_nan,
    load_runs,
)


def test_smoothing_preserves_support_and_endpoints():
    values = np.array([np.nan, np.nan, 1.0, 9.0, 5.0, 7.0, np.nan])

    smoothed = _smooth_nan(values, 5)

    assert np.isnan(smoothed[:2]).all()
    assert np.isnan(smoothed[-1])
    assert smoothed[2] == values[2]
    assert smoothed[5] == values[5]
    assert smoothed[3] == np.mean(values[2:5])


def test_align_runs_only_shifts_x_axis():
    run = {
        'x': np.array([120.0, 180.0, 260.0]),
        'y': np.array([2.0, 3.0, 5.0]),
        'y_std': np.array([0.2, 0.3, 0.5]),
    }

    aligned = _align_runs_to_first_evaluation([run])

    np.testing.assert_array_equal(aligned[0]['x'], [0.0, 60.0, 140.0])
    np.testing.assert_array_equal(aligned[0]['y'], run['y'])
    np.testing.assert_array_equal(aligned[0]['y_std'], run['y_std'])
    np.testing.assert_array_equal(run['x'], [120.0, 180.0, 260.0])


def test_load_runs_deduplicates_nested_log_roots(tmp_path):
    run_dir = tmp_path / 'baselines' / 'fedavg_sac_seed0'
    run_dir.mkdir(parents=True)
    (run_dir / 'metadata.json').write_text(json.dumps({
        'env': 'Walker2d-v5',
        'mode': 'fedavg_sac',
        'seed': 0,
    }), encoding='utf-8')
    with (run_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'total_env_steps', 'eval_reward_mean', 'eval_reward_std',
        ])
        writer.writeheader()
        writer.writerow({
            'total_env_steps': 100,
            'eval_reward_mean': 12.5,
            'eval_reward_std': 1.5,
        })

    runs = load_runs([tmp_path, tmp_path / 'baselines'])

    assert len(runs) == 1
    assert runs[0]['label'] == 'FedAvg-SAC'


def test_display_env_marks_custom_walker_reward():
    assert _display_env({
        'env': 'Walker2d-v5',
        'walker_healthy_reward': 0.05,
        'walker_forward_reward_weight': 1.0,
    }) == 'Walker2d-Locomotion (healthy=0.05, forward=1)'


def test_aggregation_plot_labels_score_mode(tmp_path):
    run_dir = tmp_path / 'batch_zscore_seed0'
    run_dir.mkdir()
    (run_dir / 'metadata.json').write_text(json.dumps({
        'env': 'Walker2d-v5',
        'mode': 'fed_evo_rl',
        'algorithm': 'SAC',
        'fed_ablation': 'full',
        'fed_score_normalization': 'batch_zscore',
        'seed': 0,
    }), encoding='utf-8')
    with (run_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'total_env_steps', 'eval_reward_mean', 'eval_reward_std',
        ])
        writer.writeheader()
        writer.writerow({
            'total_env_steps': 100,
            'eval_reward_mean': 12.5,
            'eval_reward_std': 1.5,
        })

    runs = load_runs([tmp_path], plot_kind='aggregation')

    assert len(runs) == 1
    assert runs[0]['label'] == 'FedEvoSAC-batch_zscore'


def test_module_ablation_plot_excludes_aggregation_controls(tmp_path):
    for variant in ('full', 'no_local_rl', 'raw_softmax', 'uniform_aggregation'):
        run_dir = tmp_path / variant
        run_dir.mkdir()
        (run_dir / 'metadata.json').write_text(json.dumps({
            'env': 'Walker2d-v5',
            'mode': 'fed_evo_rl',
            'algorithm': 'SAC',
            'fed_ablation': variant,
            'seed': 0,
        }), encoding='utf-8')
        with (run_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'total_env_steps', 'eval_reward_mean', 'eval_reward_std',
            ])
            writer.writeheader()
            writer.writerow({
                'total_env_steps': 100,
                'eval_reward_mean': 12.5,
                'eval_reward_std': 1.5,
            })

    labels = {run['label'] for run in load_runs([tmp_path], plot_kind='ablation')}

    assert labels == {'FedEvoSAC-full', 'FedEvoSAC-no_local_rl'}
