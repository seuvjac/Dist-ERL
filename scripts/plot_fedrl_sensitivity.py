#!/usr/bin/env python3
"""Plot final-return sensitivity with two-sided 95% Student-t intervals."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.plot_fedrl_heterogeneous import _ci_multiplier
from scripts.summarize_fedrl_results import _run_values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--envs', nargs='+', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    selected_envs = set(args.envs)
    grouped = defaultdict(list)
    for metrics_path in sorted(Path(args.log_dir).rglob('metrics.csv')):
        meta_path = metrics_path.parent / 'metadata.json'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('mode') != 'fed_evo_rl' or meta.get('fed_ablation') != 'full':
            continue
        env = meta.get('env', 'unknown')
        if env not in selected_envs:
            continue
        final = _run_values(metrics_path)
        if final is None:
            continue
        grouped[(env, float(meta.get('client_heterogeneity', 0.0)))].append(final['current'])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for env in args.envs:
        levels = sorted(level for candidate_env, level in grouped if candidate_env == env)
        if not levels:
            continue
        means, intervals = [], []
        for level in levels:
            values = np.asarray(grouped[(env, level)], dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            half_width = float(
                _ci_multiplier(np.asarray([values.size]), 0.95)[0]
                * std / np.sqrt(max(1, values.size))
            )
            means.append(mean)
            intervals.append(half_width)
            records.append({
                'env': env,
                'heterogeneity': level,
                'n': values.size,
                'final_return_mean': mean,
                'final_return_std': std,
                'final_return_ci95_half_width': half_width,
                'final_return_ci95_lower': mean - half_width,
                'final_return_ci95_upper': mean + half_width,
            })
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.errorbar(
            levels,
            means,
            yerr=intervals,
            color='#C94752',
            marker='o',
            linewidth=2.6,
            markersize=6,
            capsize=4,
        )
        ax.set_title(env, loc='left', weight='semibold')
        ax.set_xlabel('Client dynamics heterogeneity strength')
        ax.set_ylabel('Final evaluation return')
        ax.grid(axis='y', color='#d7dbe6', alpha=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f'{env}_heterogeneity_sensitivity.png', dpi=300, bbox_inches='tight')
        fig.savefig(out / f'{env}_heterogeneity_sensitivity.pdf', bbox_inches='tight')
        plt.close(fig)

    fields = [
        'env', 'heterogeneity', 'n', 'final_return_mean', 'final_return_std',
        'final_return_ci95_half_width', 'final_return_ci95_lower',
        'final_return_ci95_upper',
    ]
    with (out / 'heterogeneity_sensitivity.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(out)


if __name__ == '__main__':
    main()
