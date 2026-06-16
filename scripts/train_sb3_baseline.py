#!/usr/bin/env python3
"""Train SB3 baselines and export Dist-ERL-compatible metrics.csv files."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.environment import make_env

METRIC_FIELDS = [
    'generation', 'total_env_steps', 'eval_reward_mean', 'eval_reward_std',
    'eval_ea_mean', 'eval_ea_std',
    'best_fitness', 'mean_fitness', 'rl_steps', 'buffer_size',
    'gen_time', 'total_time', 'sync_applied', 'reproduced_trajectories', 'migrated',
    'weight_diversity', 'fitness_std', 'comm_upload_bytes', 'comm_full_traj_bytes',
    'stagnation_boost', 'rl_reset', 'migration_allowed', 'migration_gate',
    'eval_rl_aligned', 'ea_median_fitness', 'policy_exploration_noise',
    'federated_warm_start', 'migration_copies',
    'client_reward_mean', 'client_reward_std', 'client_fitness_mean',
    'client_fitness_std', 'selected_clients', 'aggregation_entropy',
    'fed_round_applied', 'archive_best', 'archive_size',
    'aggregation_temperature',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='Pendulum-v1')
    p.add_argument('--algo', default='PPO', choices=['PPO', 'SAC', 'TD3'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--total-timesteps', type=int, default=50000)
    p.add_argument('--eval-interval', type=int, default=5000)
    p.add_argument('--eval-episodes', type=int, default=5)
    p.add_argument('--max-episode-steps', type=int, default=1000)
    p.add_argument('--log-dir', default='logs/logs_sb3')
    p.add_argument('--exp-name', default=None)
    return p.parse_args()


def _make_sb3_model(algo_name, env, seed):
    try:
        from stable_baselines3 import PPO, SAC, TD3
    except ImportError as exc:
        raise SystemExit(
            'stable-baselines3 is not installed. Install project requirements first.'
        ) from exc
    cls = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3}[algo_name]
    return cls('MlpPolicy', env, seed=seed, verbose=0)


def _evaluate(model, env_name, max_episode_steps, seed, episodes):
    rewards = []
    for ep in range(episodes):
        env = make_env(env_name, max_episode_steps=max_episode_steps)
        obs, _ = env.reset(seed=seed + ep + 100000)
        done = False
        truncated = False
        total = 0.0
        steps = 0
        while not (done or truncated) and steps < max_episode_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
        env.close()
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    args = parse_args()
    exp = args.exp_name or f"sb3_{args.algo.lower()}_{args.env}_s{args.seed}"
    run_dir = Path(args.log_dir) / exp
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / 'metrics.csv'
    meta = {
        'mode': f"sb3_{args.algo.lower()}",
        'env': args.env,
        'algorithm': args.algo,
        'seed': args.seed,
        'total_timesteps': args.total_timesteps,
        'eval_interval': args.eval_interval,
        'eval_episodes': args.eval_episodes,
        'max_episode_steps': args.max_episode_steps,
        'source': 'DLR-RM/stable-baselines3',
    }
    (run_dir / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    with metrics_path.open('w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=METRIC_FIELDS).writeheader()

    try:
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            'stable-baselines3 is not installed. Install project requirements first.'
        ) from exc

    env = Monitor(make_env(args.env, max_episode_steps=args.max_episode_steps))
    model = _make_sb3_model(args.algo, env, args.seed)
    start = time.time()
    done_steps = 0
    generation = 0
    while done_steps < args.total_timesteps:
        chunk = min(args.eval_interval, args.total_timesteps - done_steps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        done_steps += chunk
        mean, std = _evaluate(
            model, args.env, args.max_episode_steps, args.seed, args.eval_episodes)
        row = {k: '' for k in METRIC_FIELDS}
        row.update({
            'generation': generation,
            'total_env_steps': done_steps,
            'eval_reward_mean': mean,
            'eval_reward_std': std,
            'best_fitness': mean,
            'gen_time': '',
            'total_time': time.time() - start,
            'selected_clients': 0,
        })
        with metrics_path.open('a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=METRIC_FIELDS).writerow(row)
        print(f"{exp}: steps={done_steps} eval={mean:.2f} +/- {std:.2f}", flush=True)
        generation += 1
    env.close()
    model.save(run_dir / 'model.zip')
    print(metrics_path)


if __name__ == '__main__':
    main()
