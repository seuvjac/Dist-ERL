#!/usr/bin/env python3
"""Train a lightweight FedAvg-DQN baseline for discrete FedRL comparisons."""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_fsac_paper_baseline import METRIC_FIELDS, ReplayBuffer
from src.utils.environment import get_env_info, make_env


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


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
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--target-update-interval', type=int, default=10)
    p.add_argument('--epsilon-start', type=float, default=1.0)
    p.add_argument('--epsilon-end', type=float, default=0.05)
    p.add_argument('--epsilon-decay-rounds', type=int, default=80)
    p.add_argument('--client-heterogeneity', type=float, default=0.25)
    p.add_argument('--client-heterogeneity-mode', default='env_params_only',
                   choices=['none', 'reward_action_noise', 'env_params',
                            'env_params_only', 'mixed'])
    p.add_argument('--eval-interval', type=int, default=5)
    p.add_argument('--eval-episodes', type=int, default=5)
    p.add_argument('--log-dir', default='logs_dqn_fedrl')
    p.add_argument('--exp-name', default=None)
    return p.parse_args()


def _epsilon(args, round_idx: int) -> float:
    frac = min(1.0, round_idx / max(1, args.epsilon_decay_rounds))
    return float(args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start))


def _action(q_net, obs, action_dim, epsilon=0.0, train=True):
    if train and np.random.rand() < epsilon:
        return int(np.random.randint(action_dim))
    with torch.no_grad():
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).float().unsqueeze(0)
        return int(torch.argmax(q_net(obs_t), dim=-1).item())


def _rollout(q_net, env_name, action_dim, max_steps, seed, epsilon=0.0, client_id=None,
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
        action = _action(q_net, obs, action_dim, epsilon, train=train)
        next_obs, reward, done, truncated, _ = env.step(action)
        terminal = bool(done or truncated)
        transitions.append((obs, action, reward, next_obs, terminal))
        total += float(reward)
        obs = next_obs
        steps += 1
    env.close()
    return total, transitions, steps


def _update(q_net, target_net, optimizer, batch, gamma):
    obs = torch.from_numpy(batch['observations']).float()
    actions = torch.from_numpy(batch['actions']).long().view(-1, 1)
    rewards = torch.from_numpy(batch['rewards']).float()
    next_obs = torch.from_numpy(batch['next_observations']).float()
    dones = torch.from_numpy(batch['dones']).float()
    q = q_net(obs).gather(1, actions).squeeze(-1)
    with torch.no_grad():
        next_q = target_net(next_obs).max(dim=-1).values
        target = rewards + gamma * (1.0 - dones) * next_q
    loss = F.smooth_l1_loss(q, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()


def _average_state_dict(nets):
    states = [net.state_dict() for net in nets]
    return {
        key: sum(state[key] for state in states) / len(states)
        for key in states[0]
    }


def _evaluate(nets, env_name, action_dim, max_steps, seed, episodes):
    rewards = []
    for wid, net in enumerate(nets):
        for ep in range(episodes):
            reward, _, _ = _rollout(
                net, env_name, action_dim, max_steps,
                seed + 200000 + wid * 1000 + ep,
                train=False,
            )
            rewards.append(reward)
    return float(np.mean(rewards)), float(np.std(rewards))


def _write_row(path, row):
    full = {k: '' for k in METRIC_FIELDS}
    full.update(row)
    with path.open('a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=METRIC_FIELDS).writerow(full)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    info = get_env_info(args.env)
    if info['action_dim'] is None:
        raise SystemExit('FedAvg-DQN only supports discrete action environments')

    q_nets = [QNetwork(info['state_dim'], info['action_dim']) for _ in range(args.num_workers)]
    target_nets = [QNetwork(info['state_dim'], info['action_dim']) for _ in range(args.num_workers)]
    for target, q_net in zip(target_nets, q_nets):
        target.load_state_dict(q_net.state_dict())
    optimizers = [optim.Adam(q_net.parameters(), lr=args.lr) for q_net in q_nets]
    buffers = [ReplayBuffer(args.buffer_size) for _ in q_nets]

    exp = args.exp_name or f"fedavg_dqn_{args.env}_s{args.seed}"
    run_dir = Path(args.log_dir) / exp
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / 'metrics.csv'
    metadata = vars(args).copy()
    metadata.update({
        'mode': 'fedavg_dqn',
        'env': args.env,
        'algorithm': 'DQN',
        'federated_rule': 'fedavg_q_network',
        'source': 'TroddenSpade/Federated-DRL',
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
        epsilon = _epsilon(args, round_idx)
        worker_rewards = []
        for wid, q_net in enumerate(q_nets):
            reward, transitions, steps = _rollout(
                q_net, args.env, info['action_dim'], args.max_episode_steps,
                args.seed + round_idx * 10000 + wid, epsilon=epsilon, client_id=wid,
                heterogeneity=args.client_heterogeneity,
                heterogeneity_mode=args.client_heterogeneity_mode,
                train=True,
            )
            worker_rewards.append(reward)
            total_steps += steps
            for item in transitions:
                buffers[wid].add(*item)
            if len(buffers[wid]) >= args.batch_size:
                for _ in range(args.updates):
                    _update(q_net, target_nets[wid], optimizers[wid],
                            buffers[wid].sample(args.batch_size), args.gamma)
                    total_updates += 1
            if round_idx % max(1, args.target_update_interval) == 0:
                target_nets[wid].load_state_dict(q_net.state_dict())

        global_q = _average_state_dict(q_nets)
        for q_net, target_net in zip(q_nets, target_nets):
            q_net.load_state_dict(global_q)
            target_net.load_state_dict(global_q)

        if round_idx % args.eval_interval == 0 or round_idx == args.rounds - 1:
            eval_mean, eval_std = _evaluate(
                q_nets, args.env, info['action_dim'], args.max_episode_steps,
                args.seed, args.eval_episodes)
            best_eval = max(best_eval, eval_mean)
            _write_row(metrics_path, {
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
                'sync_applied': 1,
                'fitness_std': float(np.std(worker_rewards)),
                'selected_clients': args.num_workers,
                'fed_round_applied': 1,
                'archive_best': best_eval,
                'client_reward_mean': float(np.mean(worker_rewards)),
                'client_reward_std': float(np.std(worker_rewards)),
                'policy_exploration_noise': epsilon,
            })
            print(f"{exp}: round={round_idx} steps={total_steps} eval={eval_mean:.2f}+/-{eval_std:.2f}", flush=True)
        round_idx += 1

    for idx, q_net in enumerate(q_nets):
        torch.save(q_net.state_dict(), run_dir / f'worker_{idx}_q.pt')
    print(metrics_path)


if __name__ == '__main__':
    main()
