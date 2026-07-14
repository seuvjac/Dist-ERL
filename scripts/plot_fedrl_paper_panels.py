#!/usr/bin/env python3
"""Render a compact multi-environment FedRL figure for paper drafts."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_fedrl_heterogeneous import (
    _apply_plot_style,
    _smooth_nan,
    load_runs,
)


COLORS = {
    'FedEvoSAC-full': '#D55E00',
    'FedEvoSAC-uniform_aggregation': '#0072B2',
    'FedEvoSAC-no_local_rl': '#009E73',
    'FedEvoSAC-no_ea_injection': '#E69F00',
    'FedEvoSAC-no_heterogeneity': '#CC79A7',
    'FedEvoSAC-raw_softmax': '#882255',
    'FedAvg-SAC': '#0072B2',
    'FedBest-SAC': '#E69F00',
    'FedSoftmax-SAC-noEA': '#009E73',
    'RobustFed-SAC-Median': '#332288',
}

LINE_STYLES = {
    'FedEvoSAC-full': '-',
    'FedEvoSAC-uniform_aggregation': '--',
    'FedEvoSAC-no_local_rl': '-.',
    'FedEvoSAC-no_ea_injection': ':',
    'FedEvoSAC-no_heterogeneity': '-',
    'FedEvoSAC-raw_softmax': '-.',
    'FedAvg-SAC': ':',
    'FedBest-SAC': '-.',
    'FedSoftmax-SAC-noEA': '--',
    'RobustFed-SAC-Median': '--',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fed-log-dir', required=True)
    parser.add_argument('--paper-log-dir', default='')
    parser.add_argument('--out-file', required=True)
    parser.add_argument('--envs', nargs='+', required=True)
    parser.add_argument('--plot-kind', choices=['comparison', 'ablation'], default='comparison')
    parser.add_argument('--x-axis', choices=['steps', 'progress', 'round'], default='round')
    parser.add_argument('--metric', choices=['current', 'candidate', 'best'], default='current')
    parser.add_argument('--variance', choices=['seed', 'sem', 'ci90', 'ci95', 'none'], default='ci90')
    parser.add_argument('--smooth-window', type=int, default=7)
    parser.add_argument('--target-x', type=float, default=None)
    return parser.parse_args()


def _band(mat, mode):
    count = np.sum(np.isfinite(mat), axis=0)
    mean = np.divide(
        np.nansum(mat, axis=0), count,
        out=np.full(mat.shape[1], np.nan), where=count > 0,
    )
    centered = np.where(np.isfinite(mat), mat - mean, 0.0)
    std = np.sqrt(np.divide(
        np.sum(centered ** 2, axis=0), np.maximum(1, count - 1),
        out=np.zeros(mat.shape[1], dtype=float), where=count > 1,
    ))
    if mode == 'none':
        return mean, np.zeros_like(mean)
    if mode == 'seed':
        return mean, std
    uncertainty = std / np.sqrt(np.maximum(1, count))
    if mode == 'ci90':
        uncertainty *= 1.645
    elif mode == 'ci95':
        uncertainty *= 1.960
    return mean, uncertainty


def _x_label(axis):
    return {
        'round': 'Communication round',
        'steps': 'Environment interactions',
        'progress': 'Normalized training progress (%)',
    }[axis]


def main():
    args = parse_args()
    _apply_plot_style('reference')
    roots = [args.fed_log_dir]
    if args.paper_log_dir:
        roots.append(args.paper_log_dir)
    runs = load_runs(roots, args.plot_kind, args.x_axis, args.metric)

    fig, axes = plt.subplots(1, len(args.envs), figsize=(5.2 * len(args.envs), 4.7), squeeze=False)
    handles = {}
    for ax, env in zip(axes[0], args.envs):
        env_runs = [run for run in runs if run['env'] == env]
        if not env_runs:
            ax.set_visible(False)
            continue
        max_x = args.target_x
        if max_x is None:
            max_x = max(float(run['x'][-1]) for run in env_runs)
        xs = np.linspace(0.0, float(max_x), 240)
        for label in sorted({run['label'] for run in env_runs}):
            group = [run for run in env_runs if run['label'] == label]
            rows = []
            for run in group:
                values = np.interp(xs, run['x'], run['y'])
                values[xs < run['x'][0]] = np.nan
                rows.append(values)
            mean, uncertainty = _band(np.vstack(rows), args.variance)
            mean = _smooth_nan(mean, args.smooth_window)
            uncertainty = _smooth_nan(uncertainty, args.smooth_window)
            color = COLORS.get(label)
            if np.nanmax(uncertainty) > 0:
                ax.fill_between(
                    xs, mean - uncertainty, mean + uncertainty,
                    color=color, alpha=0.17, linewidth=0,
                )
            line, = ax.plot(
                xs, mean, color=color, linestyle=LINE_STYLES.get(label, '-'),
                linewidth=2.5, solid_capstyle='round', label=f'{label} (n={len(group)})',
            )
            handles[label] = line
        ax.set_title(env)
        ax.set_xlabel(_x_label(args.x_axis))
        ax.grid(True, color='#d7dbe6', linewidth=0.9, alpha=0.8)
        ax.margins(x=0.0)
    axes[0][0].set_ylabel('Evaluation return')
    ordered = sorted(handles)
    if ordered:
        fig.legend(
            [handles[label] for label in ordered],
            [handles[label].get_label() for label in ordered],
            loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=min(3, len(ordered)),
            fontsize=9, framealpha=0.9,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
