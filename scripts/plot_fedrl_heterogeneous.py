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
    p.add_argument('--fed-log-dir', default='logs_fedrl_hetero')
    p.add_argument('--paper-log-dir', default='logs_fsac_paper')
    p.add_argument('--dqn-log-dir', default='logs_dqn_fedrl')
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
    return p.parse_args()


def _num(v):
    try:
        if v == '' or v is None:
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def load_runs(log_dirs, plot_kind='comparison', x_axis='steps', metric='current'):
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
                    if x_axis == 'round':
                        x = _num(row.get('generation'))
                    else:
                        x = _num(row.get('total_env_steps'))
                    if metric == 'current':
                        vals = [_num(row.get('eval_reward_mean'))]
                    elif metric == 'candidate':
                        candidate_vals = [
                            _num(row.get('candidate_eval_mean')),
                            _num(row.get('eval_ea_mean')),
                            _num(row.get('eval_reward_mean')),
                        ]
                        vals = [next((v for v in candidate_vals if np.isfinite(v)), np.nan)]
                    else:
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
            })
    return runs


def main():
    args = parse_args()
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
    for env in envs:
        env_runs = [r for r in runs if r['env'] == env]
        if not env_runs:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = sorted({r['label'] for r in env_runs})
        if args.target_x is not None:
            plot_max_x = float(args.target_x)
        elif args.max_x is not None:
            plot_max_x = float(args.max_x)
        else:
            plot_max_x = max(float(r['x'][-1]) for r in env_runs)
        for label in labels:
            group = [r for r in env_runs if r['label'] == label]
            xs = np.linspace(0, plot_max_x, 100)
            interpolated = []
            for run in group:
                vals = np.interp(xs, run['x'], run['y'])
                vals[xs < run['x'][0]] = np.nan
                interpolated.append(vals)
            mat = np.vstack(interpolated)
            y = np.nanmean(mat, axis=0)
            s = np.nanstd(mat, axis=0)
            ax.plot(xs, y, label=f"{label} (n={len(group)})", color=colors.get(label), linewidth=2)
            if len(group) > 1:
                ax.fill_between(xs, y - s, y + s, color=colors.get(label), alpha=0.14)
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
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f'{env.replace("/", "_")}_comparison.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
