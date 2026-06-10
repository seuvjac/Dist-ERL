#!/usr/bin/env python3
"""Plot FedEvoFSAC performance across heterogeneous federated scenarios."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import FEDRL_HETEROGENEITY_SCENARIOS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--log-root', default='logs_fedrl_scenarios')
    p.add_argument('--out-dir', default='plots/fedrl_scenarios')
    p.add_argument('--envs', nargs='*', default=['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'LunarLander-v3'])
    p.add_argument('--variant', default='full')
    return p.parse_args()


def _num(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def _run_label(meta):
    scenario = meta.get('scenario')
    if not scenario:
        name = meta.get('exp_name', '')
        if name.startswith('scenario_'):
            scenario = name.split('_', 2)[1] + '_' + name.split('_', 3)[2]
    if scenario in FEDRL_HETEROGENEITY_SCENARIOS:
        return FEDRL_HETEROGENEITY_SCENARIOS[scenario]['label']
    return scenario or 'unknown'


def load_runs(log_root: Path, variant: str):
    runs = []
    for metrics in sorted(log_root.glob('*/*/metrics.csv')):
        meta_path = metrics.parent / 'metadata.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        if meta.get('fed_ablation') != variant:
            continue
        scenario = metrics.parents[1].name
        meta['scenario'] = scenario
        xs, ys = [], []
        with metrics.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                x = _num(row.get('total_env_steps'))
                vals = [
                    _num(row.get('eval_reward_mean')),
                    _num(row.get('eval_ea_mean')),
                    _num(row.get('best_fitness')),
                    _num(row.get('archive_best')),
                ]
                finite = [v for v in vals if np.isfinite(v)]
                if np.isfinite(x) and finite:
                    xs.append(x)
                    ys.append(max(finite))
        if xs:
            runs.append({
                'env': meta.get('env', 'unknown'),
                'label': _run_label(meta),
                'scenario': scenario,
                'seed': meta.get('seed', ''),
                'x': np.asarray(xs, dtype=float),
                'y': np.asarray(ys, dtype=float),
            })
    return runs


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(args.log_root), args.variant)
    colors = {
        'dynamics_mild': '#0072B2',
        'sensor_reward': '#E69F00',
        'mixed_hard': '#D55E00',
    }
    summary_rows = []
    for env in args.envs:
        env_runs = [r for r in runs if r['env'] == env]
        if not env_runs:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for scenario in sorted({r['scenario'] for r in env_runs}):
            group = [r for r in env_runs if r['scenario'] == scenario]
            label = group[0]['label']
            max_x = max(float(r['x'][-1]) for r in group)
            xs = np.linspace(0, max_x, 120)
            mat = np.vstack([np.interp(xs, r['x'], r['y']) for r in group])
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)
            ax.plot(xs, mean, label=f'{label} (n={len(group)})',
                    color=colors.get(scenario), linewidth=2.4)
            if len(group) > 1:
                ax.fill_between(xs, mean - std, mean + std,
                                color=colors.get(scenario), alpha=0.14)
            summary_rows.append({
                'env': env,
                'scenario': scenario,
                'label': label,
                'n_runs': len(group),
                'mean_final': float(np.mean([r['y'][-1] for r in group])),
                'std_final': float(np.std([r['y'][-1] for r in group])),
                'mean_best': float(np.mean([np.max(r['y']) for r in group])),
                'std_best': float(np.std([np.max(r['y']) for r in group])),
            })
        ax.set_title(f'{env}: FedEvoFSAC heterogeneity scenarios')
        ax.set_xlabel('Environment steps')
        ax.set_ylabel('Best evaluation score')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f'{env}_scenarios.png', dpi=180, bbox_inches='tight')
        plt.close(fig)

    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'env', 'scenario', 'label', 'n_runs', 'mean_final', 'std_final', 'mean_best', 'std_best'])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(out)


if __name__ == '__main__':
    main()
