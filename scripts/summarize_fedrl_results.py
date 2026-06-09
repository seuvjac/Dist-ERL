#!/usr/bin/env python3
"""Summarize FedRL experiment final/current and best returns."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.plot_fedrl_heterogeneous import _num
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_fedrl_heterogeneous import _num


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fed-log-dir', default='logs_fedrl_hetero_mixed')
    p.add_argument('--paper-log-dir', default='logs_fsac_paper_mixed')
    p.add_argument('--dqn-log-dir', default='logs_dqn_fedrl_mixed')
    p.add_argument('--out-dir', default='plots/fedrl_tables_mixed')
    p.add_argument('--plot-kind', default='comparison', choices=['comparison', 'ablation', 'all'])
    p.add_argument('--envs', nargs='*', default=['CartPole-v1', 'Acrobot-v1', 'LunarLander-v3'])
    return p.parse_args()


def _label(meta, run_name):
    mode = meta.get('mode', run_name)
    if mode == 'fed_evo_rl':
        return f"FedEvoFSAC-{meta.get('fed_ablation', 'n/a')}"
    if mode == 'paper_fsac':
        return 'Paper-FSAC'
    if mode == 'paper_sac':
        return 'Paper-SAC'
    if mode in ('fedavg_sac', 'fedavg_fsac'):
        return 'FedAvg-SAC'
    if mode in ('fedsoftmax_sac_noea', 'fedsoftmax_fsac_noea'):
        return 'FedSoftmax-SAC-noEA'
    if mode in ('fedbest_sac', 'fedbest_fsac'):
        return 'FedBest-SAC'
    if mode in ('fedmedian_sac', 'fedmedian_fsac'):
        return 'FedMedian-SAC'
    if mode in ('fedtrimmedmean_sac', 'fedtrimmedmean_fsac'):
        return 'FedTrimmedMean-SAC'
    if mode in ('attention_sac_lite', 'attention_fsac_lite'):
        return 'Attention-SAC-lite'
    if mode == 'fedavg_dqn':
        return 'FedAvg-DQN'
    if mode == 'standard_erl' and meta.get('algorithm') == 'FSAC':
        return 'EvoSAC-noFed'
    return mode


def _include(meta, plot_kind):
    mode = meta.get('mode', '')
    fed_abl = meta.get('fed_ablation', 'n/a')
    if str(mode).startswith('sb3_'):
        return False
    if plot_kind == 'comparison':
        if mode == 'fed_evo_rl' and fed_abl != 'full':
            return False
        if mode == 'standard_erl' and meta.get('algorithm') == 'FSAC':
            return False
    elif plot_kind == 'ablation':
        if mode != 'fed_evo_rl':
            return False
    return True


def _run_values(metrics_path):
    rows = []
    with metrics_path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            current = _num(row.get('eval_reward_mean'))
            if not math.isfinite(current):
                continue
            best_vals = [
                _num(row.get('eval_reward_mean')),
                _num(row.get('eval_ea_mean')),
                _num(row.get('best_fitness')),
                _num(row.get('archive_best')),
            ]
            best_vals = [v for v in best_vals if math.isfinite(v)]
            rows.append({
                'generation': _num(row.get('generation')),
                'steps': _num(row.get('total_env_steps')),
                'current': current,
                'best': max(best_vals) if best_vals else current,
            })
    if not rows:
        return None
    return rows[-1]


def _fmt(mean, std):
    return f"{mean:.2f} +/- {std:.2f}"


def main():
    args = parse_args()
    roots = [Path(args.fed_log_dir), Path(args.paper_log_dir), Path(args.dqn_log_dir)]
    grouped = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for metrics in root.glob('*/metrics.csv'):
            meta_path = metrics.parent / 'metadata.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
            env = meta.get('env', 'unknown')
            if args.envs and env not in args.envs:
                continue
            if not _include(meta, args.plot_kind):
                continue
            vals = _run_values(metrics)
            if vals is None:
                continue
            grouped[(env, _label(meta, metrics.parent.name))].append(vals)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f'{args.plot_kind}_summary.csv'
    fields = [
        'env', 'method', 'n',
        'final_current_mean', 'final_current_std', 'final_current',
        'final_best_mean', 'final_best_std', 'final_best',
        'max_steps', 'max_round',
    ]
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (env, method), vals in sorted(grouped.items()):
            currents = np.asarray([v['current'] for v in vals], dtype=float)
            bests = np.asarray([v['best'] for v in vals], dtype=float)
            steps = np.asarray([v['steps'] for v in vals], dtype=float)
            rounds = np.asarray([v['generation'] for v in vals], dtype=float)
            row = {
                'env': env,
                'method': method,
                'n': len(vals),
                'final_current_mean': float(currents.mean()),
                'final_current_std': float(currents.std()),
                'final_current': _fmt(currents.mean(), currents.std()),
                'final_best_mean': float(bests.mean()),
                'final_best_std': float(bests.std()),
                'final_best': _fmt(bests.mean(), bests.std()),
                'max_steps': int(np.nanmax(steps)),
                'max_round': int(np.nanmax(rounds)),
            }
            writer.writerow(row)
    print(out_path)


if __name__ == '__main__':
    main()
