#!/usr/bin/env python3
"""Plot FedEvoRL heterogeneous variants with optional SB3 baselines."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fed-log-dir', default='logs_fedrl_hetero')
    p.add_argument('--sb3-log-dir', default='logs_sb3')
    p.add_argument('--out-dir', default='plots/fedrl_heterogeneous')
    p.add_argument('--envs', nargs='*', default=None)
    return p.parse_args()


def _num(v):
    try:
        if v == '' or v is None:
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def load_runs(log_dirs):
    runs = []
    for log_dir in log_dirs:
        root = Path(log_dir)
        if not root.exists():
            continue
        for metrics in sorted(root.glob('*/metrics.csv')):
            meta_path = metrics.parent / 'metadata.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
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
            if len(xs) < 1:
                continue
            mode = meta.get('mode', metrics.parent.name)
            fed_abl = meta.get('fed_ablation', 'n/a')
            label = mode
            if mode == 'fed_evo_rl':
                label = f"FedEvoFSAC-{fed_abl}"
            elif mode.startswith('sb3_'):
                label = mode.upper().replace('SB3_', 'SB3-')
            runs.append({
                'env': meta.get('env', 'unknown'),
                'label': label,
                'seed': meta.get('seed', ''),
                'x': np.asarray(xs, dtype=float),
                'y': np.asarray(ys, dtype=float),
            })
    return runs


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs([args.fed_log_dir, args.sb3_log_dir])
    envs = args.envs or sorted({r['env'] for r in runs})
    colors = {
        'FedEvoFSAC-full': '#D55E00',
        'FedEvoFSAC-uniform_aggregation': '#0072B2',
        'FedEvoFSAC-no_local_rl': '#009E73',
        'FedEvoFSAC-no_ea_injection': '#E69F00',
        'FedEvoFSAC-no_heterogeneity': '#CC79A7',
        'SB3-PPO': '#444444',
    }
    for env in envs:
        env_runs = [r for r in runs if r['env'] == env]
        if not env_runs:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = sorted({r['label'] for r in env_runs})
        for label in labels:
            group = [r for r in env_runs if r['label'] == label]
            max_x = max(float(r['x'][-1]) for r in group)
            xs = np.linspace(0, max_x, 100)
            mat = np.vstack([np.interp(xs, r['x'], r['y']) for r in group])
            y = mat.mean(axis=0)
            s = mat.std(axis=0)
            ax.plot(xs, y, label=f"{label} (n={len(group)})", color=colors.get(label), linewidth=2)
            if len(group) > 1:
                ax.fill_between(xs, y - s, y + s, color=colors.get(label), alpha=0.14)
        ax.set_title(f'{env}: FedRL variants vs SB3 baselines')
        ax.set_xlabel('Environment steps')
        ax.set_ylabel('Best evaluation score')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f'{env.replace("/", "_")}_comparison.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
