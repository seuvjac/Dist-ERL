#!/usr/bin/env python3
"""Summarize FedRL experiment final/current and best returns."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.plot_fedrl_heterogeneous import _num
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_fedrl_heterogeneous import _num


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fed-log-dir', default='logs/logs_fedrl_hetero_mixed')
    p.add_argument('--paper-log-dir', default='logs/logs_fsac_paper_mixed')
    p.add_argument('--dqn-log-dir', default='logs/logs_dqn_fedrl_mixed')
    p.add_argument('--out-dir', default='plots/fedrl_tables_mixed')
    p.add_argument('--plot-kind', default='comparison',
                   choices=['comparison', 'ablation', 'aggregation', 'all'])
    p.add_argument('--envs', nargs='*', default=['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'LunarLander-v3'])
    return p.parse_args()


def _label(meta, run_name):
    mode = meta.get('mode', run_name)
    if mode == 'fed_evo_rl':
        prefix = 'FedEvoSAC' if meta.get('algorithm') == 'SAC' else 'FedEvoFSAC'
        return f"{prefix}-{meta.get('fed_ablation', 'n/a')}"
    if mode == 'paper_fsac':
        return 'Paper-FSAC'
    if mode == 'paper_sac':
        return 'Paper-SAC'
    if mode == 'independent_sac':
        return 'Independent-SAC'
    if mode in ('fedavg_sac', 'fedavg_fsac'):
        return 'FedAvg-SAC'
    if mode in ('fedsoftmax_sac_noea', 'fedsoftmax_fsac_noea'):
        return 'FedSoftmax-SAC-noEA'
    if mode in ('fedbest_sac', 'fedbest_fsac'):
        return 'FedBest-SAC'
    if mode in ('fedmedian_sac', 'fedmedian_fsac'):
        return 'RobustFed-SAC-Median'
    if mode in ('fedtrimmedmean_sac', 'fedtrimmedmean_fsac'):
        return 'RobustFed-SAC-TrimmedMean'
    if mode in ('attention_sac_lite', 'attention_fsac_lite'):
        return 'ContextFed-SAC-lite'
    if mode == 'fedavg_dqn':
        return 'FedAvg-DQN'
    if mode == 'standard_erl' and meta.get('algorithm') == 'FSAC':
        return 'EvoSAC-noFed'
    return mode


def _include(meta, plot_kind):
    mode = meta.get('mode', '')
    fed_abl = meta.get('fed_ablation', 'n/a')
    if str(mode).startswith('sb3_'):
        return False
    if plot_kind == 'comparison':
        if mode == 'independent_sac':
            return False
        if mode == 'fed_evo_rl' and fed_abl != 'full':
            return False
        if mode == 'standard_erl' and meta.get('algorithm') == 'FSAC':
            return False
    elif plot_kind == 'ablation':
        if mode != 'fed_evo_rl' or fed_abl in (
            'raw_softmax', 'uniform_aggregation',
        ):
            return False
    elif plot_kind == 'aggregation':
        if mode != 'fed_evo_rl' or fed_abl != 'full':
            return False
    return True


def _run_values(metrics_path):
    rows = []
    with metrics_path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            current = _num(row.get('eval_reward_mean'))
            if not math.isfinite(current):
                continue
            best_vals = [
                _num(row.get('eval_reward_mean')),
                _num(row.get('eval_ea_mean')),
                _num(row.get('best_fitness')),
                _num(row.get('archive_best')),
            ]
            best_vals = [v for v in best_vals if math.isfinite(v)]
            rows.append({
                'generation': (
                    _num(row.get('communication_round'))
                    if math.isfinite(_num(row.get('communication_round')))
                    else _num(row.get('generation'))
                ),
                'steps': _num(row.get('total_env_steps')),
                'wall_time_sec': _num(row.get('total_time')),
                'current': current,
                'best': max(best_vals) if best_vals else current,
                'forward_return': _num(row.get('eval_forward_return_mean')),
                'survive_return': _num(row.get('eval_survive_return_mean')),
                'episode_length': _num(row.get('eval_episode_length_mean')),
                'x_displacement': _num(row.get('eval_x_displacement_mean')),
                'x_velocity': _num(row.get('eval_x_velocity_mean')),
            })
    if not rows:
        return None
    return rows[-1]


def _fmt(mean, std):
    return f"{mean:.2f} +/- {std:.2f}"


def _sample_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def main():
    args = parse_args()
    roots = []
    seen_roots = set()
    for root_name in (args.fed_log_dir, args.paper_log_dir, args.dqn_log_dir):
        if not root_name:
            continue
        root = Path(root_name)
        resolved = root.resolve()
        if resolved in seen_roots:
            continue
        seen_roots.add(resolved)
        roots.append(root)
    grouped = defaultdict(list)
    seen_metrics = set()
    for root in roots:
        if not root.exists():
            continue
        for metrics in root.rglob('metrics.csv'):
            metrics_key = metrics.resolve()
            if metrics_key in seen_metrics:
                continue
            seen_metrics.add(metrics_key)
            meta_path = metrics.parent / 'metadata.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
            env = meta.get('env', 'unknown')
            if args.envs and env not in args.envs:
                continue
            if not _include(meta, args.plot_kind):
                continue
            vals = _run_values(metrics)
            if vals is None:
                continue
            label = _label(meta, metrics.parent.name)
            if args.plot_kind == 'aggregation':
                prefix = 'FedEvoSAC' if meta.get('algorithm') == 'SAC' else 'FedEvoFSAC'
                label = f"{prefix}-{meta.get('fed_score_normalization', 'unknown')}"
            grouped[(env, label)].append(vals)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f'{args.plot_kind}_summary.csv'
    fields = [
        'env', 'method', 'n',
        'final_current_mean', 'final_current_std', 'final_current',
        'final_best_mean', 'final_best_std', 'final_best',
        'final_forward_return_mean', 'final_forward_return_std',
        'final_survive_return_mean', 'final_survive_return_std',
        'final_episode_length_mean', 'final_episode_length_std',
        'final_x_displacement_mean', 'final_x_displacement_std',
        'final_x_velocity_mean', 'final_x_velocity_std',
        'max_steps', 'max_round', 'max_wall_time_sec', 'wall_time_sec',
    ]
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (env, method), vals in sorted(grouped.items()):
            currents = np.asarray([v['current'] for v in vals], dtype=float)
            bests = np.asarray([v['best'] for v in vals], dtype=float)
            diagnostics = {
                key: np.asarray([v[key] for v in vals], dtype=float)
                for key in (
                    'forward_return', 'survive_return', 'episode_length',
                    'x_displacement', 'x_velocity',
                )
            }
            steps = np.asarray([v['steps'] for v in vals], dtype=float)
            rounds = np.asarray([v['generation'] for v in vals], dtype=float)
            wall_times = np.asarray([v['wall_time_sec'] for v in vals], dtype=float)
            current_std = _sample_std(currents)
            best_std = _sample_std(bests)
            wall_time_std = _sample_std(wall_times[np.isfinite(wall_times)])
            row = {
                'env': env,
                'method': method,
                'n': len(vals),
                'final_current_mean': float(currents.mean()),
                'final_current_std': current_std,
                'final_current': _fmt(currents.mean(), current_std),
                'final_best_mean': float(bests.mean()),
                'final_best_std': best_std,
                'final_best': _fmt(bests.mean(), best_std),
                'max_steps': int(np.nanmax(steps)),
                'max_round': int(np.nanmax(rounds)),
                'max_wall_time_sec': float(np.nanmax(wall_times)),
                'wall_time_sec': _fmt(float(np.nanmean(wall_times)), wall_time_std),
            }
            for key, values in diagnostics.items():
                finite = values[np.isfinite(values)]
                row[f'final_{key}_mean'] = float(np.mean(finite)) if finite.size else float('nan')
                row[f'final_{key}_std'] = _sample_std(finite) if finite.size else float('nan')
            writer.writerow(row)
    print(out_path)


if __name__ == '__main__':
    main()
