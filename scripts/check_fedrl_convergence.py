#!/usr/bin/env python3
"""Create a transparent tail-stability report for completed FedRL runs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dirs', nargs='+', required=True)
    parser.add_argument('--out-file', required=True)
    parser.add_argument('--tail-points', type=int, default=6)
    parser.add_argument('--relative-tolerance', type=float, default=0.03)
    parser.add_argument('--absolute-tolerance', type=float, default=5.0)
    return parser.parse_args()


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def main():
    args = parse_args()
    records = []
    seen = set()
    for root_name in args.log_dirs:
        root = Path(root_name)
        if not root.exists():
            continue
        for metrics_path in sorted(root.rglob('metrics.csv')):
            key = metrics_path.resolve()
            if key in seen:
                continue
            seen.add(key)
            metadata_path = metrics_path.parent / 'metadata.json'
            metadata = (
                json.loads(metadata_path.read_text(encoding='utf-8'))
                if metadata_path.exists() else {}
            )
            rows = list(csv.DictReader(metrics_path.open(newline='', encoding='utf-8')))
            values = [_float(row.get('eval_reward_mean')) for row in rows]
            steps = [_float(row.get('total_env_steps')) for row in rows]
            finite = [(step, value) for step, value in zip(steps, values) if np.isfinite(value)]
            tail_n = min(max(1, args.tail_points), len(finite))
            tail = np.asarray([value for _, value in finite[-tail_n:]], dtype=float)
            final = float(tail[-1]) if tail.size else np.nan
            tail_gain = float(tail[-1] - tail[0]) if tail.size else np.nan
            tail_range = float(np.ptp(tail)) if tail.size else np.nan
            tolerance = max(
                float(args.absolute_tolerance),
                float(args.relative_tolerance) * max(1.0, abs(final)) if np.isfinite(final) else np.inf,
            )
            converged = bool(
                len(finite) >= args.tail_points
                and abs(tail_gain) <= tolerance
                and tail_range <= 2.0 * tolerance
            )
            records.append({
                'env': metadata.get('env', 'unknown'),
                'mode': metadata.get('mode', metrics_path.parent.name),
                'ablation': metadata.get('fed_ablation', 'n/a'),
                'seed': metadata.get('seed', ''),
                'repeat_id': metadata.get('repeat_id', ''),
                'evaluations': len(finite),
                'final_steps': int(finite[-1][0]) if finite and np.isfinite(finite[-1][0]) else '',
                'final_return': final,
                'tail_points': tail_n,
                'tail_gain': tail_gain,
                'tail_range': tail_range,
                'tolerance': tolerance,
                'converged': int(converged),
            })

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'env', 'mode', 'ablation', 'seed', 'repeat_id', 'evaluations',
        'final_steps', 'final_return', 'tail_points', 'tail_gain',
        'tail_range', 'tolerance', 'converged',
    ]
    with out.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    failed = sum(not bool(row['converged']) for row in records)
    print(f'{out}: {len(records) - failed}/{len(records)} runs tail-stable')


if __name__ == '__main__':
    main()
