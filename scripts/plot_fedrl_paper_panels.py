#!/usr/bin/env python3
"""Render a compact multi-environment FedRL figure for paper drafts."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_fedrl_heterogeneous import (
    _align_runs_to_first_evaluation,
    _apply_plot_style,
    _smooth_nan,
    load_runs,
)


COLORS = {
    'FedEvoSAC-full': '#C94752',
    'FedEvoSAC-uniform_aggregation': '#0072B2',
    'FedEvoSAC-no_local_rl': '#009E73',
    'FedEvoSAC-no_ea_injection': '#E69F00',
    'FedEvoSAC-no_heterogeneity': '#CC79A7',
    'FedEvoSAC-raw_softmax': '#882255',
    'FedAvg-SAC': '#3B82B6',
    'FedBest-SAC': '#C7A439',
    'FedSoftmax-SAC-noEA': '#3A9D68',
    'RobustFed-SAC-Median': '#62528C',
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
    parser.add_argument('--align-start', action='store_true',
                        help='Shift each run so its first logged x value is plotted at zero')
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


def _x_label(axis, align_start=False):
    if axis == 'steps' and align_start:
        return 'Environment interactions since first evaluation'
    if axis == 'round' and align_start:
        return 'Communication rounds since first evaluation'
    return {
        'round': 'Communication round',
        'steps': 'Environment interactions',
        'progress': 'Normalized training progress (%)',
    }[axis]


def main():
    args = parse_args()
    _apply_plot_style('paper')
    roots = [args.fed_log_dir]
    if args.paper_log_dir:
        roots.append(args.paper_log_dir)
    runs = load_runs(roots, args.plot_kind, args.x_axis, args.metric)
    if args.align_start:
        runs = _align_runs_to_first_evaluation(runs)

    fig, axes = plt.subplots(1, len(args.envs), figsize=(5.4 * len(args.envs), 4.9), squeeze=False)
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
            is_proposed = label == 'FedEvoSAC-full'
            if np.nanmax(uncertainty) > 0:
                ax.fill_between(
                    xs, mean - uncertainty, mean + uncertainty,
                    color=color, alpha=0.13 if is_proposed else 0.055, linewidth=0,
                    zorder=2 if is_proposed else 1,
                )
            line, = ax.plot(
                xs, mean, color=color, linestyle=LINE_STYLES.get(label, '-'),
                linewidth=3.25 if is_proposed else 2.1, solid_capstyle='round',
                label=label, zorder=4 if is_proposed else 3,
            )
            handles[label] = line
        ax.set_title(env_runs[0].get('display_env', env), loc='left', pad=10)
        if args.x_axis == 'steps':
            ax.set_xlabel('Environment interactions ($10^6$)')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1e6:g}'))
        else:
            ax.set_xlabel(_x_label(args.x_axis, args.align_start))
        ax.grid(axis='y', color='#d7dbe6', linewidth=0.85, alpha=0.8)
        ax.grid(axis='x', color='#eef1f5', linewidth=0.7, alpha=0.75)
        ax.margins(x=0.0, y=0.04)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0][0].set_ylabel('Average evaluation return')
    ordered = sorted(handles)
    if ordered:
        fig.legend(
            [handles[label] for label in ordered],
            [handles[label].get_label() for label in ordered],
            loc='lower right', bbox_to_anchor=(0.99, 0.015),
            ncol=1, fontsize=9, framealpha=0.96,
            borderpad=0.55, labelspacing=0.35, handlelength=2.4,
        )
    fig.tight_layout(rect=(0.0, 0.21 if ordered else 0.0, 1.0, 1.0))
    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
