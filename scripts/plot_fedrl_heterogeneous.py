#!/usr/bin/env python3
"""Plot FedEvoFSAC variants with Paper-SAC/FSAC baselines."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fed-log-dir', default='logs/logs_fedrl_hetero')
    p.add_argument('--paper-log-dir', default='logs/logs_fsac_paper')
    p.add_argument('--dqn-log-dir', default='logs/logs_dqn_fedrl')
    p.add_argument('--out-dir', default='plots/fedrl_heterogeneous')
    p.add_argument('--envs', nargs='*', default=None)
    p.add_argument('--plot-kind', default='comparison',
                   choices=['comparison', 'ablation', 'all'],
                   help='comparison excludes FedEvoFSAC ablations and EvoSAC-noFed; ablation plots only FedEvoFSAC variants')
    p.add_argument('--x-axis', default='steps', choices=['steps', 'progress', 'round'],
                   help='Use raw env steps, per-run progress percentage, or logged generation/round as x-axis')
    p.add_argument('--max-x', type=float, default=None,
                   help='Optional x-axis cap after x-axis conversion')
    p.add_argument('--target-x', type=float, default=None,
                   help='Force every method in each plot to extend to this x value, holding the final return')
    p.add_argument('--align-start', action='store_true',
                   help='Shift each run so its first logged x value is plotted at zero')
    p.add_argument('--metric', default='current', choices=['current', 'candidate', 'best'],
                   help='current uses deployable eval; candidate uses the current training policy; best uses optimistic archive metrics')
    p.add_argument('--variance', default='seed', choices=['seed', 'eval', 'combined', 'none'],
                   help='Uncertainty band source: across seeds, within-run evaluation std, both, or none')
    p.add_argument('--smooth-window', type=int, default=1,
                   help='Moving-average window over interpolated plotting points; 1 disables smoothing')
    p.add_argument('--style', default='reference', choices=['reference', 'standard'],
                   help='reference uses a white grid, faint raw traces, and thick smoothed lines')
    p.add_argument('--raw-traces', action=argparse.BooleanOptionalAction, default=True,
                   help='Draw faint unsmoothed traces behind the smoothed mean curve')
    return p.parse_args()


def _num(v):
    try:
        if v == '' or v is None:
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def _smooth_nan(values, window):
    window = int(window)
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size < 3:
        return arr
    if window % 2 == 0:
        window += 1
    window = min(window, arr.size if arr.size % 2 == 1 else arr.size - 1)
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=float)
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    num = np.convolve(filled, kernel, mode='same')
    den = np.convolve(finite.astype(float), kernel, mode='same')
    out = np.divide(num, den, out=np.full_like(arr, np.nan), where=den > 0)
    out[~finite & (den <= 0)] = np.nan
    return out


def _apply_plot_style(style):
    if style != 'reference':
        return
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#d7dbe6',
        'axes.labelcolor': '#2f3640',
        'axes.titlecolor': '#1f2933',
        'axes.grid': True,
        'grid.color': '#d7dbe6',
        'grid.linewidth': 0.9,
        'grid.alpha': 0.8,
        'xtick.color': '#4b5563',
        'ytick.color': '#4b5563',
        'font.size': 11,
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#d7dbe6',
    })


def load_runs(log_dirs, plot_kind='comparison', x_axis='steps', metric='current'):
    runs = []
    for log_dir in log_dirs:
        root = Path(log_dir)
        if not root.exists():
            continue
        for metrics in sorted(root.glob('*/metrics.csv')):
            meta_path = metrics.parent / 'metadata.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
            xs, ys, y_stds = [], [], []
            with metrics.open(newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if x_axis == 'round':
                        x = _num(row.get('generation'))
                    else:
                        x = _num(row.get('total_env_steps'))
                    if metric == 'current':
                        pairs = [(_num(row.get('eval_reward_mean')), _num(row.get('eval_reward_std')))]
                    elif metric == 'candidate':
                        pairs = [
                            (_num(row.get('candidate_eval_mean')), _num(row.get('candidate_eval_std'))),
                            (_num(row.get('eval_ea_mean')), _num(row.get('eval_ea_std'))),
                            (_num(row.get('eval_reward_mean')), _num(row.get('eval_reward_std'))),
                        ]
                    else:
                        pairs = [
                            (_num(row.get('eval_reward_mean')), _num(row.get('eval_reward_std'))),
                            (_num(row.get('eval_ea_mean')), _num(row.get('eval_ea_std'))),
                            (_num(row.get('best_fitness')), _num(row.get('fitness_std'))),
                            (_num(row.get('archive_best')), _num(row.get('deployable_eval_std'))),
                        ]
                    finite = [(v, s) for v, s in pairs if np.isfinite(v)]
                    if np.isfinite(x) and finite:
                        value, std = max(finite, key=lambda item: item[0])
                        xs.append(x)
                        ys.append(value)
                        y_stds.append(std if np.isfinite(std) else 0.0)
            if len(xs) < 1:
                continue
            if x_axis == 'progress':
                denom = max(xs)
                if denom > 0:
                    xs = [100.0 * x / denom for x in xs]
            mode = meta.get('mode', metrics.parent.name)
            if str(mode).startswith('sb3_'):
                continue
            fed_abl = meta.get('fed_ablation', 'n/a')

            if plot_kind == 'comparison':
                if mode == 'independent_sac':
                    continue
                if mode == 'fed_evo_rl' and fed_abl != 'full':
                    continue
                if mode == 'standard_erl' and meta.get('algorithm') == 'FSAC':
                    continue
            elif plot_kind == 'ablation':
                if mode != 'fed_evo_rl':
                    continue

            label = mode
            if mode == 'fed_evo_rl':
                prefix = 'FedEvoSAC' if meta.get('algorithm') == 'SAC' else 'FedEvoFSAC'
                label = f"{prefix}-{fed_abl}"
            elif mode == 'standard_erl' and meta.get('algorithm') == 'FSAC':
                label = 'EvoSAC-noFed'
            elif mode == 'paper_fsac':
                label = 'Paper-FSAC'
            elif mode == 'paper_sac':
                label = 'Paper-SAC'
            elif mode == 'independent_sac':
                label = 'Independent-SAC'
            elif mode in ('fedavg_sac', 'fedavg_fsac'):
                label = 'FedAvg-SAC'
            elif mode in ('fedsoftmax_sac_noea', 'fedsoftmax_fsac_noea'):
                label = 'FedSoftmax-SAC-noEA'
            elif mode in ('fedbest_sac', 'fedbest_fsac'):
                label = 'FedBest-SAC'
            elif mode in ('fedmedian_sac', 'fedmedian_fsac'):
                label = 'RobustFed-SAC-Median'
            elif mode in ('fedtrimmedmean_sac', 'fedtrimmedmean_fsac'):
                label = 'RobustFed-SAC-TrimmedMean'
            elif mode in ('attention_sac_lite', 'attention_fsac_lite'):
                label = 'ContextFed-SAC-lite'
            elif mode == 'fedavg_dqn':
                label = 'FedAvg-DQN'
            runs.append({
                'env': meta.get('env', 'unknown'),
                'label': label,
                'seed': meta.get('seed', ''),
                'x': np.asarray(xs, dtype=float),
                'y': np.asarray(ys, dtype=float),
                'y_std': np.asarray(y_stds, dtype=float),
            })
    return runs


def main():
    args = parse_args()
    _apply_plot_style(args.style)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_dirs = [args.fed_log_dir]
    if args.paper_log_dir:
        log_dirs.append(args.paper_log_dir)
    if args.dqn_log_dir:
        log_dirs.append(args.dqn_log_dir)
    runs = load_runs(
        log_dirs, plot_kind=args.plot_kind, x_axis=args.x_axis, metric=args.metric)
    if args.align_start:
        shifted = []
        for run in runs:
            if len(run['x']) < 1:
                continue
            run = dict(run)
            run['x'] = run['x'] - run['x'][0]
            shifted.append(run)
        runs = shifted
    if args.max_x is not None:
        capped = []
        for run in runs:
            mask = run['x'] <= args.max_x
            if np.any(mask):
                run = dict(run)
                run['x'] = run['x'][mask]
                run['y'] = run['y'][mask]
                run['y_std'] = run['y_std'][mask]
                capped.append(run)
        runs = capped
    envs = args.envs or sorted({r['env'] for r in runs})
    colors = {
        'FedEvoFSAC-full': '#D55E00',
        'FedEvoSAC-full': '#D55E00',
        'FedEvoFSAC-uniform_aggregation': '#0072B2',
        'FedEvoFSAC-no_local_rl': '#009E73',
        'FedEvoFSAC-no_ea_injection': '#E69F00',
        'FedEvoFSAC-no_heterogeneity': '#CC79A7',
        'FedEvoFSAC-raw_softmax': '#882255',
        'FedEvoSAC-raw_softmax': '#882255',
        'Paper-FSAC': '#56B4E9',
        'Paper-SAC': '#999999',
        'Independent-SAC': '#999999',
        'FedAvg-SAC': '#0072B2',
        'FedSoftmax-SAC-noEA': '#009E73',
        'FedBest-SAC': '#E69F00',
        'EvoSAC-noFed': '#CC79A7',
        'RobustFed-SAC-Median': '#332288',
        'RobustFed-SAC-TrimmedMean': '#882255',
        'ContextFed-SAC-lite': '#44AA99',
        'FedAvg-DQN': '#117733',
    }
    line_styles = {
        'FedEvoFSAC-full': '-',
        'FedEvoSAC-full': '-',
        'FedEvoFSAC-raw_softmax': '-.',
        'FedEvoSAC-raw_softmax': '-.',
        'FedAvg-SAC': ':',
        'FedSoftmax-SAC-noEA': '--',
        'FedBest-SAC': '-.',
        'RobustFed-SAC-Median': '--',
        'RobustFed-SAC-TrimmedMean': '-.',
        'ContextFed-SAC-lite': '--',
        'FedAvg-DQN': ':',
    }
    for env in envs:
        env_runs = [r for r in runs if r['env'] == env]
        if not env_runs:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        if args.style == 'reference':
            ax.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_color('#d7dbe6')
            ax.set_axisbelow(True)
        labels = sorted({r['label'] for r in env_runs})
        if args.target_x is not None:
            plot_max_x = float(args.target_x)
        elif args.max_x is not None:
            plot_max_x = float(args.max_x)
        else:
            plot_max_x = max(float(r['x'][-1]) for r in env_runs)
        for label in labels:
            group = [r for r in env_runs if r['label'] == label]
            xs = np.linspace(0, plot_max_x, 240)
            interpolated = []
            interpolated_stds = []
            for run in group:
                vals = np.interp(xs, run['x'], run['y'])
                std_vals = np.interp(xs, run['x'], run['y_std'])
                vals[xs < run['x'][0]] = np.nan
                std_vals[xs < run['x'][0]] = np.nan
                interpolated.append(vals)
                interpolated_stds.append(std_vals)
                if args.raw_traces and args.style == 'reference':
                    ax.plot(
                        xs,
                        vals,
                        color=colors.get(label),
                        linestyle=line_styles.get(label, '-'),
                        linewidth=1.0,
                        alpha=0.14,
                        zorder=1,
                    )
            mat = np.vstack(interpolated)
            std_mat = np.vstack(interpolated_stds)
            y = np.nanmean(mat, axis=0)
            seed_s = np.nanstd(mat, axis=0)
            eval_s = np.nanmean(std_mat, axis=0)
            if args.variance == 'eval':
                s = eval_s
            elif args.variance == 'combined':
                s = np.sqrt(seed_s ** 2 + eval_s ** 2)
            elif args.variance == 'none':
                s = np.zeros_like(y)
            else:
                s = seed_s
            y = _smooth_nan(y, args.smooth_window)
            s = _smooth_nan(s, args.smooth_window)
            if np.isfinite(s).any() and np.nanmax(s) > 0:
                ax.fill_between(
                    xs,
                    y - s,
                    y + s,
                    color=colors.get(label),
                    alpha=0.18 if args.style == 'reference' else 0.14,
                    linewidth=0,
                    zorder=2,
                )
            ax.plot(
                xs,
                y,
                label=f"{label} (n={len(group)})",
                color=colors.get(label),
                linestyle=line_styles.get(label, '-'),
                linewidth=2.8 if args.style == 'reference' else 2,
                solid_capstyle='round',
                zorder=3,
            )
        if args.plot_kind == 'ablation':
            title = f'{env}: FedEvoFSAC ablations'
        else:
            title = f'{env}: FedEvoFSAC vs FedRL baselines'
        ax.set_title(title)
        if args.x_axis == 'progress':
            xlabel = 'Training progress (%)'
        elif args.x_axis == 'round':
            xlabel = 'Communication round / generation'
        elif args.align_start:
            xlabel = 'Environment steps since first logged evaluation'
        else:
            xlabel = 'Environment steps'
        ax.set_xlabel(xlabel)
        ylabel = 'Evaluation return' if args.metric == 'current' else 'Best evaluation score'
        ax.set_ylabel(ylabel)
        if args.style == 'reference':
            ax.grid(True, color='#d7dbe6', linewidth=0.9, alpha=0.8)
            ax.margins(x=0.0)
            ax.legend(fontsize=8, framealpha=0.86)
        else:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f'{env.replace("/", "_")}_comparison.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
