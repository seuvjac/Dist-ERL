#!/usr/bin/env python3
"""Publication plots from local logs (no synthetic fallback by default)."""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9})


def _load_config():
    path = Path(__file__).resolve().parent / 'src' / 'config.py'
    spec = importlib.util.spec_from_file_location('erl_config', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cfg = _load_config()
BASELINE_MODES = _cfg.BASELINE_MODES
ERL_PROGRESSION_MODES = _cfg.ERL_PROGRESSION_MODES
MODE_LABELS = _cfg.MODE_LABELS
PLOT_STYLES = _cfg.PLOT_STYLES
RE2_ABLATION_VARIANTS = _cfg.RE2_ABLATION_VARIANTS
FED_ABLATION_VARIANTS = _cfg.FED_ABLATION_VARIANTS
FED_ABLATION_FULL = _cfg.FED_ABLATION_FULL
FED_EVO_RL = _cfg.FED_EVO_RL
BENCHMARK_ENVS = _cfg.BENCHMARK_ENVS
effective_mode_label = _cfg.effective_mode_label


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--log-dir', default='logs')
    p.add_argument('--metric', default='eval_reward_mean', choices=['eval_reward_mean', 'best_fitness'])
    p.add_argument('--env', default=None)
    p.add_argument('--modes', nargs='+', default=None)
    p.add_argument('--points', type=int, default=100)
    p.add_argument('--require-real', action='store_true', help='Error if no logs (no synthetic demo)')
    p.add_argument('--allow-synthetic', action='store_true', help='Allow demo curves when logs missing')
    p.add_argument('--show', action='store_true')
    return p.parse_args()


def _style(key):
    if key in PLOT_STYLES:
        return PLOT_STYLES[key]
    base = key.split('__')[0]
    return PLOT_STYLES.get(base, {'color': '#333', 'ls': '-', 'lw': 2.5, 'marker': 'o'})


def _read_csv(path, metric, xcol='total_env_steps'):
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
    x, y, t, div = [], [], [], []
    for row in rows:
        try:
            xv = float(row[xcol])
            yv = float(row[metric])
            if np.isfinite(xv) and np.isfinite(yv):
                x.append(xv)
                y.append(yv)
            if 'total_time' in row:
                t.append(float(row['total_time']))
            if 'weight_diversity' in row and row['weight_diversity'] != '':
                div.append(float(row['weight_diversity']))
        except (KeyError, ValueError):
            continue
    if len(x) < 2:
        return None
    o = np.argsort(x)
    return {
        'x': np.array(x)[o], 'y': np.array(y)[o],
        'time': np.array(t) if len(t) == len(x) else None,
        'diversity': div,
    }


def load_entries(log_dir, metric, env_filter=None):
    entries = []
    for d in sorted(Path(log_dir).iterdir()):
        if not d.is_dir() or not (d / 'metrics.csv').exists():
            continue
        meta = json.loads((d / 'metadata.json').read_text(encoding='utf-8')) if (d / 'metadata.json').exists() else {}
        if env_filter and meta.get('env') != env_filter:
            continue
        data = _read_csv(d / 'metrics.csv', metric)
        if data is None:
            continue
        mode = meta.get('mode', d.name.split('_')[0])
        abl = meta.get('ablation', 'n/a')
        fed_abl = meta.get('fed_ablation', 'n/a')
        if mode == 'erl_re2' and abl not in ('n/a', 'full'):
            key = f"{mode}__{abl}"
            label = effective_mode_label(mode, abl)
        elif mode == FED_EVO_RL and fed_abl not in ('n/a', FED_ABLATION_FULL):
            key = f"{mode}__{fed_abl}"
            label = effective_mode_label(mode, fed_abl)
        else:
            key = mode
            label = effective_mode_label(mode, FED_ABLATION_FULL) if mode == FED_EVO_RL else MODE_LABELS.get(mode, mode)
        seed = meta.get('seed')
        entries.append({
            'key': key, 'label': label, 'mode': mode, 'seed': seed,
            'env': meta.get('env'), 'dir': str(d), **data,
        })
    return entries


def group_by_key(entries):
    g = {}
    for e in entries:
        g.setdefault(e['key'], {'label': e['label'], 'runs': []})
        g[e['key']]['runs'].append((e['x'], e['y']))
        g[e['key']].setdefault('time_runs', []).append(
            (e['time'], e['y']) if e['time'] is not None and len(e['time']) == len(e['y']) else None
        )
    g[e['key']]['time_runs'] = [t for t in g[e['key']]['time_runs'] if t is not None]
    return g


def resample_matrix(runs, n, xcol_idx=0):
    if not runs:
        return None, None
    max_x = max(r[0][-1] for r in runs)
    xs = np.linspace(0, max_x, n)
    mat = np.vstack([np.interp(xs, r[0], r[1]) for r in runs])
    return xs, mat


def iqm(vals):
    q25, q75 = np.percentile(vals, [25, 75])
    mid = vals[(vals >= q25) & (vals <= q75)]
    return float(np.mean(mid)) if len(mid) else float(np.mean(vals))


def plot_curve(ax, xs, mat, label, key, alpha_fill=0.2):
    st = _style(key)
    mean = np.median(mat, axis=0) if mat.shape[0] >= 3 else mat.mean(axis=0)
    if mat.shape[0] >= 2:
        lo = np.percentile(mat, 25, axis=0)
        hi = np.percentile(mat, 75, axis=0)
        ax.fill_between(xs, lo, hi, color=st['color'], alpha=alpha_fill)
    markevery = max(1, len(xs) // 8)
    ax.plot(xs, mean, color=st['color'], linestyle=st['ls'], linewidth=st['lw'],
            marker=st.get('marker'), markevery=markevery, label=label)


def plot_panel(ax, groups, order, title, xlabel, use_time=False):
    plotted = 0
    for key in order:
        if key not in groups:
            continue
        runs = groups[key]['time_runs'] if use_time and groups[key].get('time_runs') else groups[key]['runs']
        if not runs:
            continue
        xs, mat = resample_matrix(runs, 100)
        if mat is None:
            continue
        plot_curve(ax, xs, mat, groups[key]['label'], key)
        plotted += 1
    if plotted == 0:
        return False
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Eval Reward Mean' if 'eval' in title.lower() or True else 'Metric')
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.35, linestyle='--')
    return True


def significance_table(entries, out_path, baseline='standard_erl', target='fed_evo_rl'):
    rows = []
    b_runs = [e['y'][-1] for e in entries if e['mode'] == baseline]
    t_runs = [e['y'][-1] for e in entries if e['mode'] == target]
    if len(b_runs) >= 2 and len(t_runs) >= 2:
        tstat, pval = stats.ttest_ind(t_runs, b_runs, equal_var=False)
        rows.append({
            'comparison': f'{target}_vs_{baseline}',
            'n_target': len(t_runs), 'n_baseline': len(b_runs),
            'iqm_target': iqm(np.array(t_runs)), 'iqm_baseline': iqm(np.array(b_runs)),
            'p_value': pval, 'significant_0.05': pval < 0.05,
        })
    if rows:
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f'Significance: {out_path}')


def main():
    args = parse_args()
    out = Path('plots')
    out.mkdir(exist_ok=True)

    entries = load_entries(args.log_dir, args.metric, args.env)
    if not entries:
        if args.require_real or not args.allow_synthetic:
            print('ERROR: No logs found. Run ./run_seeds.sh or ./run_benchmarks.sh first.', file=sys.stderr)
            sys.exit(1)
        print('WARNING: Using synthetic demo (pass --allow-synthetic explicitly).')
        sys.exit(1)

    groups = {}
    for e in entries:
        groups.setdefault(e['key'], {'label': e['label'], 'runs': [], 'time_runs': []})
        groups[e['key']]['runs'].append((e['x'], e['y']))
        if e['time'] is not None and len(e['time']) == len(e['y']):
            groups[e['key']]['time_runs'].append((e['time'], e['y']))

    if args.modes:
        groups = {k: v for k, v in groups.items() if k in args.modes or k.split('__')[0] in args.modes}

    order = [m for m in BASELINE_MODES if m in groups]
    paths = []

    if len(order) >= 3:
        for use_time, suffix, xlab in [(False, 'steps', 'Total Environment Steps'),
                                        (True, 'wallclock', 'Wall-clock Time (s)')]:
            fig, ax = plt.subplots(figsize=(12, 7))
            if plot_panel(ax, groups, order, f'Baseline Comparison ({len(order)} methods)', xlab, use_time):
                fig.savefig(out / f'sample_efficiency_comparison_{suffix}.png', dpi=300, bbox_inches='tight')
                paths.append(out / f'sample_efficiency_comparison_{suffix}.png')
            plt.close(fig)
    else:
        print(f'WARNING: Only {len(order)} baselines found (need >=3). Skipping main comparison.')

    abl_order = []
    if 'erl_re2' in groups:
        abl_order.append('erl_re2')
    for abl, _ in RE2_ABLATION_VARIANTS:
        k = f'erl_re2__{abl}'
        if k in groups:
            abl_order.append(k)
    if len(abl_order) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        if plot_panel(ax, groups, abl_order, 'ERL-Re2 Baseline Ablation',
                      'Total Environment Steps'):
            fig.savefig(out / 'ablation_re2_impact.png', dpi=300, bbox_inches='tight')
            paths.append(out / 'ablation_re2_impact.png')
        plt.close(fig)

    fed_abl_order = []
    if FED_EVO_RL in groups:
        fed_abl_order.append(FED_EVO_RL)
    for abl, _ in FED_ABLATION_VARIANTS:
        k = f'{FED_EVO_RL}__{abl}'
        if k in groups:
            fed_abl_order.append(k)
    if len(fed_abl_order) >= 4:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        if plot_panel(ax, groups, fed_abl_order, 'FedEvoRL Ablation',
                      'Total Environment Steps'):
            fig.savefig(out / 'ablation_fed_evo_rl.png', dpi=300, bbox_inches='tight')
            paths.append(out / 'ablation_fed_evo_rl.png')
        plt.close(fig)
    elif fed_abl_order:
        print(f'WARNING: Only {len(fed_abl_order)} FedEvoRL ablation curves found (need >=4).')

    by_env = {}
    for e in entries:
        by_env.setdefault(e['env'], []).append(e)
    env_ids = [c['id'] for c in BENCHMARK_ENVS if c['id'] in by_env] + [x for x in by_env if x not in [c['id'] for c in BENCHMARK_ENVS]]
    if len(env_ids) >= 1:
        n = len(env_ids)
        ncols = 2
        nrows = (n + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows))
        axes = np.atleast_2d(axes)
        compare = ['pure_rl', 'standard_erl', 'erl_re2', 'dist_erl', 'fed_evo_rl']
        for idx, env_id in enumerate(env_ids):
            r, c = divmod(idx, ncols)
            sub = {}
            for e in by_env[env_id]:
                sub.setdefault(e['key'], {'label': e['label'], 'runs': [], 'time_runs': []})
                sub[e['key']]['runs'].append((e['x'], e['y']))
            short = next((x['short'] for x in BENCHMARK_ENVS if x['id'] == env_id), env_id)
            plot_panel(axes[r, c], sub, [m for m in compare if m in sub], short, 'Total Environment Steps')
        for idx in range(len(env_ids), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)
        fig.suptitle('Cross-Environment Comparison', fontweight='bold')
        fig.savefig(out / 'multi_env_comparison.png', dpi=300, bbox_inches='tight')
        paths.append(out / 'multi_env_comparison.png')
        plt.close(fig)

    significance_table(entries, out / 'significance_table.csv')

    print('Plots:')
    for p in paths:
        print(f'  {p}')


if __name__ == '__main__':
    main()
