#!/usr/bin/env python3
"""Train discrete SAC/FSAC baselines for FedEvoFSAC comparisons.

These baselines keep local SAC critics on each worker and only synchronize
actor parameters. They are intended as clean comparison targets for FedEvoFSAC.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.environment import get_env_info, make_env
from src.utils.policies import FSACPolicy

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


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data = []
        self.pos = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        item = (
            np.asarray(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            float(done),
        )
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, len(self.data), size=int(batch_size))
        batch = [self.data[i] for i in idx]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return {
            'observations': np.asarray(obs, dtype=np.float32),
            'actions': np.asarray(actions, dtype=np.int64),
            'rewards': np.asarray(rewards, dtype=np.float32),
            'next_observations': np.asarray(next_obs, dtype=np.float32),
            'dones': np.asarray(dones, dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.data)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='CartPole-v1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rounds', type=int, default=120)
    p.add_argument('--target-env-steps', type=int, default=0,
                   help='Continue training until at least this many environment steps are collected')
    p.add_argument('--num-workers', type=int, default=5)
    p.add_argument('--max-episode-steps', type=int, default=500)
    p.add_argument('--updates', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--buffer-size', type=int, default=50000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--pi-smoothing', type=float, default=0.9,
                   help='EMA factor for paper-style worker performance index')
    p.add_argument('--federated-temperature', type=float, default=50.0)
    p.add_argument('--client-heterogeneity', type=float, default=0.25)
    p.add_argument('--client-heterogeneity-mode', default='env_params_only',
                   choices=['none', 'reward_action_noise', 'env_params',
                            'env_params_only', 'mixed',
                            'reward_scale_only', 'env_params_reward_scale'])
    p.add_argument('--eval-interval', type=int, default=5)
    p.add_argument('--eval-episodes', type=int, default=5)
    p.add_argument('--log-dir', default='logs/logs_fsac_paper')
    p.add_argument('--exp-name', default=None)
    p.add_argument('--baseline-mode', default=None,
                   choices=['paper_sac', 'paper_fsac',
                            'fedavg_sac', 'fedsoftmax_sac_noea', 'fedbest_sac',
                            'fedmedian_sac', 'fedtrimmedmean_sac',
                            'attention_sac_lite',
                            'fedavg_fsac', 'fedsoftmax_fsac_noea', 'fedbest_fsac',
                            'fedmedian_fsac', 'fedtrimmedmean_fsac',
                            'attention_fsac_lite'],
                   help='SAC/FSAC baseline variant. Overrides --federated flags.')
    p.add_argument('--federated', action='store_true',
                   help='Legacy alias for --baseline-mode paper_fsac')
    p.add_argument('--no-federation', dest='federated', action='store_false')
    p.set_defaults(federated=True)
    return p.parse_args()


def _actor_state(policy: FSACPolicy) -> dict:
    return {k: v.detach().cpu().clone() for k, v in policy.actor.state_dict().items()}


def _load_actor_state(policy: FSACPolicy, state: dict) -> None:
    policy.actor.load_state_dict({k: v.clone() for k, v in state.items()})


def _blend_with_best(policy: FSACPolicy, best_state: dict, worker_pi: float,
                     best_pi: float, temperature: float) -> float:
    local_state = _actor_state(policy)
    temp = max(1e-6, float(temperature))
    logits = np.asarray([worker_pi, best_pi], dtype=np.float64) / temp
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    mixed = {
        k: weights[0] * local_state[k] + weights[1] * best_state[k]
        for k in local_state
    }
    _load_actor_state(policy, mixed)
    return float(-(weights * np.log(weights + 1e-12)).sum())


def _softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(1e-6, float(temperature))
    logits = np.asarray(scores, dtype=np.float64) / temp
    logits -= np.max(logits)
    weights = np.exp(logits)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(logits) / len(logits)
    return weights / total


def _average_actor_states(states, weights) -> dict:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    return {
        key: sum(float(w) * state[key] for w, state in zip(weights, states))
        for key in states[0]
    }


def _median_actor_state(states) -> dict:
    return {
        key: torch.median(torch.stack([state[key] for state in states], dim=0), dim=0).values
        for key in states[0]
    }


def _trimmed_mean_actor_state(states, trim_ratio: float = 0.2) -> dict:
    n_states = len(states)
    trim = int(np.floor(n_states * trim_ratio))
    result = {}
    for key in states[0]:
        stacked = torch.stack([state[key] for state in states], dim=0)
        if trim > 0 and n_states > 2 * trim:
            sorted_vals = torch.sort(stacked, dim=0).values
            stacked = sorted_vals[trim:n_states - trim]
        result[key] = stacked.mean(dim=0)
    return result


def _actor_distance(state_a: dict, state_b: dict) -> float:
    total = 0.0
    for key in state_a:
        diff = state_a[key] - state_b[key]
        total += float(torch.sum(diff * diff).item())
    return float(np.sqrt(max(total, 0.0)))


def _attention_weights(states, scores: np.ndarray, best_state: dict,
                       temperature: float) -> np.ndarray:
    distances = np.asarray([_actor_distance(state, best_state) for state in states], dtype=np.float64)
    scale = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
    context_bonus = -distances / max(float(scale), 1e-6)
    logits = np.asarray(scores, dtype=np.float64) / max(float(temperature), 1e-6) + context_bonus
    logits -= np.max(logits)
    weights = np.exp(logits)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(logits) / len(logits)
    return weights / total


def _sync_actor_baseline(policies, perf_index: np.ndarray, mode: str,
                         temperature: float) -> float:
    legacy_aliases = {
        'fedavg_fsac': 'fedavg_sac',
        'fedsoftmax_fsac_noea': 'fedsoftmax_sac_noea',
        'fedbest_fsac': 'fedbest_sac',
        'fedmedian_fsac': 'fedmedian_sac',
        'fedtrimmedmean_fsac': 'fedtrimmedmean_sac',
        'attention_fsac_lite': 'attention_sac_lite',
    }
    mode = legacy_aliases.get(mode, mode)
    if mode == 'paper_sac':
        return 0.0

    states = [_actor_state(policy) for policy in policies]
    best_worker = int(np.argmax(perf_index))
    best_state = states[best_worker]

    if mode == 'paper_fsac':
        entropies = []
        best_pi = float(perf_index[best_worker])
        for wid, policy in enumerate(policies):
            if wid == best_worker:
                continue
            entropies.append(_blend_with_best(
                policy, best_state, float(perf_index[wid]), best_pi, temperature))
        return float(np.mean(entropies)) if entropies else 0.0

    if mode == 'fedbest_sac':
        for policy in policies:
            _load_actor_state(policy, best_state)
        return 0.0

    if mode == 'fedavg_sac':
        weights = np.ones(len(policies), dtype=np.float64) / len(policies)
    elif mode == 'fedsoftmax_sac_noea':
        weights = _softmax_weights(perf_index, temperature)
    elif mode == 'attention_sac_lite':
        weights = _attention_weights(states, perf_index, best_state, temperature)
    elif mode == 'fedmedian_sac':
        global_state = _median_actor_state(states)
        for policy in policies:
            _load_actor_state(policy, global_state)
        return 0.0
    elif mode == 'fedtrimmedmean_sac':
        global_state = _trimmed_mean_actor_state(states)
        for policy in policies:
            _load_actor_state(policy, global_state)
        return 0.0
    else:
        raise ValueError(f'unknown baseline mode: {mode}')

    global_state = _average_actor_states(states, weights)
    for policy in policies:
        _load_actor_state(policy, global_state)
    return float(-(weights * np.log(weights + 1e-12)).sum())


def _rollout(policy, env_name, max_steps, seed, client_id=None,
             heterogeneity=0.0, heterogeneity_mode='none', train=True):
    env = make_env(
        env_name,
        max_episode_steps=max_steps,
        client_id=client_id,
        heterogeneity=heterogeneity,
        heterogeneity_mode=heterogeneity_mode,
    )
    obs, _ = env.reset(seed=seed)
    total = 0.0
    transitions = []
    done = False
    truncated = False
    steps = 0
    while not (done or truncated) and steps < max_steps:
        action = policy.get_action(obs, deterministic=not train)
        next_obs, reward, done, truncated, _ = env.step(action)
        terminal = bool(done or truncated)
        transitions.append((obs, action, reward, next_obs, terminal))
        total += float(reward)
        obs = next_obs
        steps += 1
    env.close()
    return total, transitions, steps


def _evaluate(policies, env_name, max_steps, seed, episodes):
    rewards = []
    for wid, policy in enumerate(policies):
        for ep in range(episodes):
            reward, _, _ = _rollout(
                policy, env_name, max_steps,
                seed + 100000 + wid * 1000 + ep,
                train=False,
            )
            rewards.append(reward)
    return float(np.mean(rewards)), float(np.std(rewards))


def _write_row(path: Path, row: dict) -> None:
    full = {k: '' for k in METRIC_FIELDS}
    full.update(row)
    with path.open('a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=METRIC_FIELDS).writerow(full)


def main():
    args = parse_args()
    if args.env not in ('CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'LunarLander-v3'):
        raise SystemExit('paper FSAC baseline is configured for the supported discrete FedRL envs only')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    info = get_env_info(args.env)
    policies = [
        FSACPolicy(info['state_dim'], info['action_dim'])
        for _ in range(args.num_workers)
    ]
    optimizers = [optim.Adam(p.parameters(), lr=args.lr) for p in policies]
    buffers = [ReplayBuffer(args.buffer_size) for _ in policies]
    perf_index = np.zeros(args.num_workers, dtype=np.float64)

    mode = args.baseline_mode or ('paper_fsac' if args.federated else 'paper_sac')
    exp = args.exp_name or f"{mode}_{args.env}_s{args.seed}"
    run_dir = Path(args.log_dir) / exp
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / 'metrics.csv'
    metadata = vars(args).copy()
    metadata.update({
        'mode': mode,
        'env': args.env,
        'algorithm': 'Discrete-SAC',
        'federated_rule': mode,
        'shared_parameters': 'actor',
        'local_parameters': 'critic,target_critic,entropy_temperature',
        'ea_enabled': False,
        'source': 'Federated Reinforcement Learning for Sharing Experiences Between Multiple Workers',
    })
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with metrics_path.open('w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=METRIC_FIELDS).writeheader()

    start = time.time()
    total_steps = 0
    total_updates = 0
    best_eval = -np.inf

    round_idx = 0
    while round_idx < args.rounds or (
        args.target_env_steps > 0 and total_steps < args.target_env_steps
    ):
        round_start = time.time()
        worker_rewards = []
        sync_entropy = []
        for wid, policy in enumerate(policies):
            reward, transitions, steps = _rollout(
                policy,
                args.env,
                args.max_episode_steps,
                args.seed + round_idx * 10000 + wid,
                client_id=wid,
                heterogeneity=args.client_heterogeneity,
                heterogeneity_mode=args.client_heterogeneity_mode,
                train=True,
            )
            worker_rewards.append(reward)
            total_steps += steps
            for item in transitions:
                buffers[wid].add(*item)
            perf_index[wid] = args.pi_smoothing * perf_index[wid] + reward

            if len(buffers[wid]) >= args.batch_size:
                for _ in range(args.updates):
                    batch = buffers[wid].sample(args.batch_size)
                    loss = policy.update(batch, args.gamma, args.tau)
                    optimizers[wid].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                    optimizers[wid].step()
                    policy.sync_target(args.tau)
                    total_updates += 1

        sync_entropy_value = _sync_actor_baseline(
            policies, perf_index, mode, args.federated_temperature)
        if mode != 'paper_sac':
            sync_entropy.append(sync_entropy_value)

        if round_idx % args.eval_interval == 0 or round_idx == args.rounds - 1:
            eval_mean, eval_std = _evaluate(
                policies, args.env, args.max_episode_steps, args.seed,
                args.eval_episodes,
            )
            best_eval = max(best_eval, eval_mean)
            row = {
                'generation': round_idx,
                'total_env_steps': total_steps,
                'eval_reward_mean': eval_mean,
                'eval_reward_std': eval_std,
                'best_fitness': best_eval,
                'mean_fitness': float(np.mean(worker_rewards)),
                'rl_steps': total_updates,
                'buffer_size': int(sum(len(b) for b in buffers)),
                'gen_time': time.time() - round_start,
                'total_time': time.time() - start,
                'sync_applied': int(mode != 'paper_sac'),
                'fitness_std': float(np.std(worker_rewards)),
                'selected_clients': args.num_workers,
                'aggregation_entropy': float(np.mean(sync_entropy)) if sync_entropy else 0.0,
                'fed_round_applied': int(mode != 'paper_sac'),
                'archive_best': best_eval,
                'archive_size': 0,
                'aggregation_temperature': args.federated_temperature,
                'client_reward_mean': float(np.mean(worker_rewards)),
                'client_reward_std': float(np.std(worker_rewards)),
                'client_fitness_mean': float(np.mean(perf_index)),
                'client_fitness_std': float(np.std(perf_index)),
            }
            _write_row(metrics_path, row)
            print(
                f"{exp}: round={round_idx} steps={total_steps} "
                f"eval={eval_mean:.2f}+/-{eval_std:.2f} "
                f"worker={np.mean(worker_rewards):.2f}",
                flush=True,
            )
        round_idx += 1

    for idx, policy in enumerate(policies):
        torch.save(policy.state_dict(), run_dir / f'worker_{idx}.pt')
    print(metrics_path)


if __name__ == '__main__':
    main()
