#!/usr/bin/env python3
"""Plot worker scaling, bandwidth, and diversity from metrics.csv logs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_runs(log_dir, prefix='scaling_'):
    runs = []
    for d in Path(log_dir).iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        meta = json.loads((d / 'metadata.json').read_text(encoding='utf-8'))
        rows = list(csv.DictReader(open(d / 'metrics.csv', encoding='utf-8')))
        if not rows:
            continue
        last = rows[-1]
        runs.append({
            'name': d.name,
            'workers': meta.get('num_workers'),
            'env': meta.get('env'),
            'final_eval': float(last.get('eval_reward_mean', 0)),
            'total_time': float(last.get('total_time', 0)),
            'upload': float(last.get('comm_upload_bytes', 0)),
            'full_traj': float(last.get('comm_full_traj_bytes', 0)),
            'diversity': [float(r.get('weight_diversity', 0)) for r in rows if r.get('weight_diversity')],
            'steps': [float(r['total_env_steps']) for r in rows],
            'evals': [float(r['eval_reward_mean']) for r in rows if r.get('eval_reward_mean')],
        })
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log-dir', default='logs')
    p.add_argument('--out-dir', default='plots')
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    runs = sorted(load_runs(args.log_dir), key=lambda x: x['workers'] or 0)
    if not runs:
        print('No scaling_* runs found.')
        return

    workers = [r['workers'] for r in runs]
    times = [r['total_time'] for r in runs]
    t1 = times[0] if times else 1
    speedup = [t1 / max(t, 1e-6) for t in times]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(workers, times, 'o-', color='#D55E00', linewidth=2.5, markersize=8, label='Wall-clock to finish')
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Dist-ERL Scalability (Wall-clock)')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(workers, speedup, 's--', color='#0072B2', linewidth=2, label='Speedup vs 1 worker')
    ax2.set_ylabel('Speedup')
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc='best')
    fig.savefig(out / 'scaling_wallclock.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    upload = [r['upload'] for r in runs]
    full = [r['full_traj'] for r in runs]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(workers))
    w = 0.35
    ax.bar(x - w / 2, np.array(upload) / 1e6, w, label='Seed+fitness upload (MB/gen)', color='#009E73')
    ax.bar(x + w / 2, np.array(full) / 1e6, w, label='Hypothetical full traj (MB/gen)', color='#E69F00')
    ax.set_xticks(x)
    ax.set_xticklabels([str(wk) for wk in workers])
    ax.set_xlabel('Workers')
    ax.set_ylabel('MB per generation')
    ax.set_title('Communication: Seed vs Full Trajectory')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.savefig(out / 'bandwidth_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    if runs[0]['diversity']:
        fig, ax = plt.subplots(figsize=(9, 5))
        for r in runs:
            if r['diversity']:
                ax.plot(r['diversity'], label=f"{r['workers']} workers", linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Weight Diversity (1 - mean cosine sim)')
        ax.set_title('Population Diversity During Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(out / 'diversity_over_generations.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f'Wrote scaling/bandwidth/diversity plots to {out}/')


if __name__ == '__main__':
    main()
