#!/usr/bin/env python3
"""Plot FedEvoFSAC variants with Paper-SAC/FSAC baselines."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fed-log-dir', default='logs/logs_fedrl_hetero')
    p.add_argument('--paper-log-dir', default='logs/logs_fsac_paper')
    p.add_argument('--dqn-log-dir', default='logs/logs_dqn_fedrl')
    p.add_argument('--out-dir', default='plots/fedrl_heterogeneous')
    p.add_argument('--envs', nargs='*', default=None)
    p.add_argument('--repeat-ids', nargs='*', default=None,
                   help='Optional repeat_id values to include; useful for a documented experiment snapshot')
    p.add_argument('--plot-kind', default='comparison',
                   choices=['comparison', 'ablation', 'aggregation', 'all'],
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
    p.add_argument(
        '--variance', default='seed',
        choices=['seed', 'sem', 'ci90', 'ci95', 'eval', 'combined', 'none'],
        help='Uncertainty band: seed std, standard error, 90/95%% CI, eval std, combined std, or none',
    )
    p.add_argument('--smooth-window', type=int, default=1,
                   help='Moving-average window over interpolated plotting points; 1 disables smoothing')
    p.add_argument('--style', default='reference', choices=['reference', 'paper', 'standard'],
                   help='reference is diagnostic; paper uses restrained publication-facing emphasis')
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


def _display_env(meta):
    env = meta.get('env', 'unknown')
    healthy = _num(meta.get('walker_healthy_reward'))
    forward = _num(meta.get('walker_forward_reward_weight'))
    if env == 'Walker2d-v5' and np.isfinite(healthy) and np.isfinite(forward):
        if not np.isclose(healthy, 1.0) or not np.isclose(forward, 1.0):
            return f'Walker2d-Locomotion (healthy={healthy:g}, forward={forward:g})'
    return env


def _smooth_nan(values, window):
    """Smooth finite spans without inventing values outside their support.

    The old convolution-based implementation leaked later values into leading
    NaNs and changed the first/last observed points.  That made a method whose
    first evaluation happened later appear to start at x=0 with an inflated
    return.  A shrinking, symmetric window keeps both endpoints exact and
    leaves every originally missing point missing.
    """
    window = int(window)
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size < 3:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    finite = np.isfinite(arr)
    out = arr.copy()
    half_window = window // 2
    starts = np.flatnonzero(finite & np.r_[True, ~finite[:-1]])
    ends = np.flatnonzero(finite & np.r_[~finite[1:], True])
    for start, end in zip(starts, ends):
        span = arr[start:end + 1]
        for offset in range(span.size):
            radius = min(half_window, offset, span.size - offset - 1)
            if radius > 0:
                out[start + offset] = np.mean(
                    span[offset - radius:offset + radius + 1])
    return out


def _align_runs_to_first_evaluation(runs):
    """Return copies whose first recorded x value is zero."""
    aligned = []
    for run in runs:
        if len(run['x']) < 1:
            continue
        shifted = dict(run)
        shifted['x'] = run['x'] - run['x'][0]
        aligned.append(shifted)
    return aligned


def _apply_plot_style(style):
    if style not in ('reference', 'paper'):
        return
    params = {
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
    }
    if style == 'paper':
        params.update({
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 16,
            'legend.fontsize': 9,
            'axes.titleweight': 'semibold',
        })
    plt.rcParams.update(params)


def load_runs(log_dirs, plot_kind='comparison', x_axis='steps', metric='current', repeat_ids=None):
    runs = []
    seen_metrics = set()
    selected_repeats = {str(value) for value in repeat_ids} if repeat_ids else None
    for log_dir in log_dirs:
        root = Path(log_dir)
        if not root.exists():
            continue
        for metrics in sorted(root.rglob('metrics.csv')):
            metrics_key = metrics.resolve()
            if metrics_key in seen_metrics:
                continue
            seen_metrics.add(metrics_key)
            meta_path = metrics.parent / 'metadata.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
            if selected_repeats is not None and str(meta.get('repeat_id', '')) not in selected_repeats:
                continue
            xs, ys, y_stds = [], [], []
            with metrics.open(newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if x_axis == 'round':
                        x = _num(row.get('communication_round'))
                        if not np.isfinite(x):
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
            if x_axis == 'round':
                collapsed = {}
                for x, y, y_std in zip(xs, ys, y_stds):
                    collapsed[float(x)] = (float(y), float(y_std))
                xs = sorted(collapsed)
                ys = [collapsed[x][0] for x in xs]
                y_stds = [collapsed[x][1] for x in xs]
            if x_axis == 'progress':
                start = min(xs)
                denom = max(xs) - start
                if denom > 0:
                    xs = [100.0 * (x - start) / denom for x in xs]
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
                if mode != 'fed_evo_rl' or fed_abl in (
                    'raw_softmax', 'uniform_aggregation',
                ):
                    continue
            elif plot_kind == 'aggregation':
                if mode != 'fed_evo_rl' or fed_abl != 'full':
                    continue

            label = mode
            if mode == 'fed_evo_rl':
                prefix = 'FedEvoSAC' if meta.get('algorithm') == 'SAC' else 'FedEvoFSAC'
                if plot_kind == 'aggregation':
                    score_mode = meta.get('fed_score_normalization', 'unknown')
                    label = f"{prefix}-{score_mode}"
                else:
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
                'display_env': _display_env(meta),
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
        log_dirs,
        plot_kind=args.plot_kind,
        x_axis=args.x_axis,
        metric=args.metric,
        repeat_ids=args.repeat_ids,
    )
    if args.align_start:
        runs = _align_runs_to_first_evaluation(runs)
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
        'FedEvoSAC-uniform_aggregation': '#0072B2',
        'FedEvoFSAC-no_local_rl': '#009E73',
        'FedEvoSAC-no_local_rl': '#009E73',
        'FedEvoFSAC-no_ea_injection': '#E69F00',
        'FedEvoSAC-no_ea_injection': '#E69F00',
        'FedEvoFSAC-no_heterogeneity': '#CC79A7',
        'FedEvoSAC-no_heterogeneity': '#CC79A7',
        'FedEvoFSAC-raw_softmax': '#882255',
        'FedEvoSAC-raw_softmax': '#882255',
        'FedEvoSAC-relative_gain': '#D55E00',
        'FedEvoSAC-batch_zscore': '#0072B2',
        'FedEvoSAC-raw': '#009E73',
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
    if args.style == 'paper':
        # Colorblind-friendly palette with a clear visual hierarchy: the
        # proposed method is the only saturated solid curve.
        colors.update({
            'FedEvoFSAC-full': '#C94752',
            'FedEvoSAC-full': '#C94752',
            'FedAvg-SAC': '#3B82B6',
            'FedSoftmax-SAC-noEA': '#3A9D68',
            'FedBest-SAC': '#C7A439',
            'RobustFed-SAC-Median': '#62528C',
        })
    line_styles = {
        'FedEvoFSAC-full': '-',
        'FedEvoSAC-full': '-',
        'FedEvoFSAC-uniform_aggregation': '--',
        'FedEvoSAC-uniform_aggregation': '--',
        'FedEvoFSAC-no_local_rl': '-.',
        'FedEvoSAC-no_local_rl': '-.',
        'FedEvoFSAC-no_ea_injection': ':',
        'FedEvoSAC-no_ea_injection': ':',
        'FedEvoFSAC-no_heterogeneity': '-',
        'FedEvoSAC-no_heterogeneity': '-',
        'FedEvoFSAC-raw_softmax': '-.',
        'FedEvoSAC-raw_softmax': '-.',
        'FedEvoSAC-relative_gain': '-',
        'FedEvoSAC-batch_zscore': '--',
        'FedEvoSAC-raw': '-.',
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
        display_env = env_runs[0].get('display_env', env)
        paper_style = args.style == 'paper'
        fig, ax = plt.subplots(figsize=(9.6, 5.8) if paper_style else (10, 6))
        if args.style in ('reference', 'paper'):
            ax.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_color('#d7dbe6')
            ax.set_axisbelow(True)
        if paper_style:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=10.5)
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
            finite_count = np.sum(np.isfinite(mat), axis=0)
            y = np.divide(
                np.nansum(mat, axis=0),
                finite_count,
                out=np.full(mat.shape[1], np.nan),
                where=finite_count > 0,
            )
            centered = np.where(np.isfinite(mat), mat - y, 0.0)
            seed_s = np.sqrt(np.divide(
                np.sum(centered ** 2, axis=0),
                np.maximum(1, finite_count - 1),
                out=np.zeros(mat.shape[1], dtype=float),
                where=finite_count > 1,
            ))
            eval_s = np.divide(
                np.nansum(std_mat, axis=0),
                np.maximum(1, np.sum(np.isfinite(std_mat), axis=0)),
            )
            if args.variance == 'eval':
                s = eval_s
            elif args.variance == 'combined':
                s = np.sqrt(seed_s ** 2 + eval_s ** 2)
            elif args.variance in ('sem', 'ci90', 'ci95'):
                s = seed_s / np.sqrt(np.maximum(1, finite_count))
                if args.variance == 'ci90':
                    s = 1.645 * s
                elif args.variance == 'ci95':
                    s = 1.960 * s
            elif args.variance == 'none':
                s = np.zeros_like(y)
            else:
                s = seed_s
            y = _smooth_nan(y, args.smooth_window)
            s = _smooth_nan(s, args.smooth_window)
            is_proposed = label in ('FedEvoFSAC-full', 'FedEvoSAC-full')
            if np.isfinite(s).any() and np.nanmax(s) > 0:
                ax.fill_between(
                    xs,
                    y - s,
                    y + s,
                    color=colors.get(label),
                    alpha=(0.13 if is_proposed else 0.055) if paper_style
                    else (0.18 if args.style == 'reference' else 0.14),
                    linewidth=0,
                    zorder=2 if is_proposed else 1,
                )
            ax.plot(
                xs,
                y,
                label=label if paper_style else f"{label} (n={len(group)})",
                color=colors.get(label),
                linestyle=line_styles.get(label, '-'),
                linewidth=3.35 if paper_style and is_proposed
                else (2.15 if paper_style else (2.8 if args.style == 'reference' else 2)),
                solid_capstyle='round',
                zorder=4 if is_proposed else 3,
            )
        if paper_style:
            title = display_env
        elif args.plot_kind == 'ablation':
            title = f'{display_env}: FedEvoSAC ablations'
        elif args.plot_kind == 'aggregation':
            title = f'{display_env}: aggregation screening'
        else:
            title = f'{display_env}: FedEvoSAC vs FedRL baselines'
        ax.set_title(title, loc='left' if paper_style else 'center', pad=10)
        if paper_style and args.x_axis == 'steps':
            xlabel = 'Environment interactions ($10^6$)'
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1e6:g}'))
        elif args.x_axis == 'progress':
            xlabel = 'Training progress (%)'
        elif args.x_axis == 'round':
            xlabel = (
                'Communication rounds'
                if args.align_start else 'Communication round'
            )
        elif args.align_start:
            xlabel = 'Environment steps since first logged evaluation'
        else:
            xlabel = 'Environment steps'
        ax.set_xlabel(xlabel)
        if args.metric == 'current':
            ylabel = 'Average evaluation return'
        elif args.metric == 'candidate':
            ylabel = 'Candidate evaluation return'
        else:
            ylabel = 'Best evaluation score'
        ax.set_ylabel(ylabel)
        if paper_style:
            ax.grid(axis='y', color='#d7dbe6', linewidth=0.85, alpha=0.8)
            ax.grid(axis='x', color='#eef1f5', linewidth=0.7, alpha=0.75)
            ax.margins(x=0.0, y=0.04)
            ax.legend(
                loc='lower right', fontsize=9, framealpha=0.96,
                borderpad=0.55, labelspacing=0.35, handlelength=2.4,
            )
        elif args.style == 'reference':
            ax.grid(True, color='#d7dbe6', linewidth=0.9, alpha=0.8)
            ax.margins(x=0.0)
            ax.legend(loc='lower right', fontsize=8, framealpha=0.86)
        else:
            ax.grid(alpha=0.3)
            ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        out_file = out / f'{env.replace("/", "_")}_comparison.png'
        fig.savefig(out_file, dpi=300 if paper_style else 180, bbox_inches='tight')
        if paper_style:
            fig.savefig(out_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
