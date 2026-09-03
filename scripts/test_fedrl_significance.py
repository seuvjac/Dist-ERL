#!/usr/bin/env python3
"""Paired, seed-matched significance tests for FedRL final returns."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

from scripts.summarize_fedrl_results import _include, _label, _run_values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dirs', nargs='+', required=True)
    parser.add_argument('--out-file', required=True)
    parser.add_argument('--envs', nargs='*', default=None)
    parser.add_argument('--plot-kind', choices=['comparison', 'ablation'], default='comparison')
    parser.add_argument('--reference', default='FedEvoSAC-full')
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260903)
    return parser.parse_args()


def _rank_biserial(differences):
    diff = np.asarray(differences, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if not diff.size:
        return 0.0
    ranks = rankdata(np.abs(diff), method='average')
    positive = float(ranks[diff > 0].sum())
    negative = float(ranks[diff < 0].sum())
    return (positive - negative) / max(1e-12, positive + negative)


def _bootstrap_mean_ci(differences, samples, seed):
    diff = np.asarray(differences, dtype=float)
    if not diff.size:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diff.size, size=(max(1, samples), diff.size))
    means = diff[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def _wilcoxon(differences, alternative):
    # Rounding avoids accidental rank splits caused only by floating-point
    # subtraction, as recommended by scipy.stats.wilcoxon.
    diff = np.round(np.asarray(differences, dtype=float), decimals=10)
    if not np.any(diff != 0):
        return 0.0, 1.0
    result = wilcoxon(
        diff,
        zero_method='wilcox',
        correction=False,
        alternative=alternative,
        method='auto',
    )
    return float(result.statistic), float(result.pvalue)


def _holm_adjust(records, p_key, out_key):
    ordered = sorted(range(len(records)), key=lambda idx: records[idx][p_key])
    running = 0.0
    total = len(records)
    for rank, idx in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * records[idx][p_key])
        running = max(running, adjusted)
        records[idx][out_key] = running


def main():
    args = parse_args()
    selected_envs = set(args.envs or [])
    values = defaultdict(dict)
    seen = set()
    for root_name in args.log_dirs:
        root = Path(root_name)
        if not root.exists():
            continue
        for metrics_path in sorted(root.rglob('metrics.csv')):
            resolved = metrics_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            meta_path = metrics_path.parent / 'metadata.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
            env = meta.get('env', 'unknown')
            if selected_envs and env not in selected_envs:
                continue
            if not _include(meta, args.plot_kind):
                continue
            final = _run_values(metrics_path)
            if final is None or not math.isfinite(final['current']):
                continue
            label = _label(meta, metrics_path.parent.name)
            seed = int(meta.get('seed', -1))
            key = (env, label)
            if seed in values[key]:
                raise ValueError(f'duplicate non-independent seed {seed} for {env}/{label}')
            values[key][seed] = float(final['current'])

    output = []
    envs = args.envs or sorted({env for env, _ in values})
    for env_index, env in enumerate(envs):
        reference = values.get((env, args.reference), {})
        comparators = sorted(
            label for candidate_env, label in values
            if candidate_env == env and label != args.reference
        )
        env_rows = []
        for comparator_index, comparator in enumerate(comparators):
            baseline = values[(env, comparator)]
            seeds = sorted(set(reference) & set(baseline))
            if not seeds:
                continue
            ref = np.asarray([reference[seed] for seed in seeds], dtype=float)
            other = np.asarray([baseline[seed] for seed in seeds], dtype=float)
            diff = ref - other
            statistic, p_two_sided = _wilcoxon(diff, 'two-sided')
            _, p_greater = _wilcoxon(diff, 'greater')
            ci_low, ci_high = _bootstrap_mean_ci(
                diff,
                args.bootstrap_samples,
                args.bootstrap_seed + 1000 * env_index + comparator_index,
            )
            env_rows.append({
                'env': env,
                'reference': args.reference,
                'comparator': comparator,
                'paired_n': len(seeds),
                'seeds': ' '.join(str(seed) for seed in seeds),
                'reference_mean': float(ref.mean()),
                'comparator_mean': float(other.mean()),
                'mean_difference': float(diff.mean()),
                'mean_difference_bootstrap_ci95_lower': ci_low,
                'mean_difference_bootstrap_ci95_upper': ci_high,
                'median_difference': float(np.median(diff)),
                'wins': int(np.sum(diff > 0)),
                'ties': int(np.sum(diff == 0)),
                'losses': int(np.sum(diff < 0)),
                'wilcoxon_statistic': statistic,
                'wilcoxon_p_two_sided_raw': p_two_sided,
                'wilcoxon_p_greater_raw': p_greater,
                'rank_biserial_correlation': _rank_biserial(diff),
            })
        _holm_adjust(env_rows, 'wilcoxon_p_two_sided_raw', 'wilcoxon_p_two_sided_holm')
        _holm_adjust(env_rows, 'wilcoxon_p_greater_raw', 'wilcoxon_p_greater_holm')
        for row in env_rows:
            row['reject_two_sided_0_05'] = int(row['wilcoxon_p_two_sided_holm'] < 0.05)
            row['reject_greater_0_05'] = int(row['wilcoxon_p_greater_holm'] < 0.05)
        output.extend(env_rows)

    fields = [
        'env', 'reference', 'comparator', 'paired_n', 'seeds',
        'reference_mean', 'comparator_mean', 'mean_difference',
        'mean_difference_bootstrap_ci95_lower', 'mean_difference_bootstrap_ci95_upper',
        'median_difference', 'wins', 'ties', 'losses',
        'wilcoxon_statistic', 'wilcoxon_p_two_sided_raw',
        'wilcoxon_p_two_sided_holm', 'reject_two_sided_0_05',
        'wilcoxon_p_greater_raw', 'wilcoxon_p_greater_holm',
        'reject_greater_0_05', 'rank_biserial_correlation',
    ]
    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(f'{out}: {len(output)} paired comparisons')


if __name__ == '__main__':
    main()
