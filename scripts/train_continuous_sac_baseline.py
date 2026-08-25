#!/usr/bin/env python3
"""Train continuous SAC/FedSAC baselines for FedEvoSAC comparisons."""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import METRIC_FIELDS
from src.utils.environment import get_env_info, make_env
from src.utils.policies import SACPolicy
from src.utils.policy_utils import clip_action
from src.utils.replay_buffer import HybridReplayBuffer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='Reacher-v5')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--repeat-id', default='')
    p.add_argument('--seed-slot', type=int, default=-1)
    p.add_argument('--rounds', type=int, default=120)
    p.add_argument('--target-env-steps', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--max-episode-steps', type=int, default=1000)
    p.add_argument('--updates', type=int, default=8)
    p.add_argument('--update-to-data-ratio', type=float, default=0.02,
                   help='Minimum SAC gradient updates per newly collected transition.')
    p.add_argument('--max-updates-per-round', type=int, default=20,
                   help='Upper bound on SAC updates per worker and communication round.')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--buffer-size', type=int, default=200000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--actor-lr', type=float, default=0.0,
                   help='Actor learning rate; <=0 uses --lr.')
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--client-heterogeneity', type=float, default=0.6)
    p.add_argument('--client-heterogeneity-mode', default='mixed',
                   choices=['none', 'reward_action_noise', 'env_params',
                            'env_params_only', 'mixed',
                            'reward_scale_only', 'env_params_reward_scale'])
    p.add_argument('--walker-healthy-reward', type=float, default=1.0)
    p.add_argument('--walker-forward-reward-weight', type=float, default=1.0)
    p.add_argument('--hopper-healthy-reward', type=float, default=1.0)
    p.add_argument('--hopper-forward-reward-weight', type=float, default=1.0)
    p.add_argument('--eval-interval', type=int, default=5)
    p.add_argument('--eval-episodes', type=int, default=3)
    p.add_argument('--log-dir', default='logs/logs_sac_continuous')
    p.add_argument('--exp-name', default=None)
    p.add_argument('--baseline-mode', default='independent_sac',
                   choices=['independent_sac', 'fedavg_sac', 'fedbest_sac',
                            'fedsoftmax_sac_noea', 'fedmedian_sac'])
    p.add_argument('--federated-temperature', type=float, default=50.0)
    p.add_argument('--aggregation-interval', type=int, default=5,
                   help='Local rollout rounds between federated actor aggregations.')
    p.add_argument('--server-learning-rate', type=float, default=0.25,
                   help='Blend factor from the previous global actor to the aggregated actor.')
    p.add_argument('--aggregation-eval-episodes', type=int, default=3,
                   help='Fixed-seed episodes used to accept or reject a global actor update.')
    p.add_argument('--post-sync-critic-warmup-updates', type=int, default=2,
                   help='Critic-only updates after each accepted/rejected global actor synchronization.')
    p.add_argument('--disable-deployment-rollback', action='store_true',
                   help='Report the current candidate instead of the best deployment checkpoint.')
    p.add_argument('--deployment-rollback-tolerance', type=float, default=0.0,
                   help='Allowed candidate drop before retaining the best deployment checkpoint.')
    return p.parse_args()


def _actor_state(policy):
    return {k: v.detach().cpu().clone() for k, v in policy.actor.state_dict().items()}


def _load_actor_state(policy, state):
    policy.actor.load_state_dict({k: v.clone() for k, v in state.items()})


def _average_actor_states(states, weights):
    out = {}
    for key in states[0]:
        acc = torch.zeros_like(states[0][key], dtype=torch.float32)
        for state, weight in zip(states, weights):
            acc += state[key].float() * float(weight)
        out[key] = acc.to(states[0][key].dtype)
    return out


def _median_actor_state(states):
    out = {}
    for key in states[0]:
        out[key] = torch.stack([s[key].float() for s in states], dim=0).median(dim=0).values
    return out


def _aggregate_actor_state(policies, scores, mode, temperature):
    if mode == 'independent_sac':
        return _actor_state(policies[0]), 0.0
    states = [_actor_state(p) for p in policies]
    scores = np.asarray(scores, dtype=np.float64)
    if mode == 'fedbest_sac':
        return states[int(np.argmax(scores))], 0.0
    if mode == 'fedmedian_sac':
        return _median_actor_state(states), 0.0
    if mode == 'fedsoftmax_sac_noea':
        logits = (scores - scores.max()) / max(1e-6, float(temperature))
        weights = np.exp(np.clip(logits, -60.0, 0.0))
        weights = weights / max(1e-8, weights.sum())
    else:
        weights = np.ones(len(states), dtype=np.float64) / len(states)
    global_state = _average_actor_states(states, weights)
    weights = np.clip(weights, 1e-12, 1.0)
    return global_state, float(-(weights * np.log(weights)).sum())


def _blend_actor_states(base, target, rate):
    rate = float(np.clip(rate, 0.0, 1.0))
    return {
        key: ((1.0 - rate) * base[key].float() + rate * target[key].float()).to(base[key].dtype)
        for key in base
    }


def _rollout(
    policy,
    env_name,
    max_steps,
    seed,
    client_id,
    heterogeneity,
    heterogeneity_mode,
    train=True,
    env_kwargs=None,
):
    env = make_env(env_name, max_episode_steps=max_steps, client_id=client_id,
                   heterogeneity=heterogeneity, heterogeneity_mode=heterogeneity_mode,
                   **dict(env_kwargs or {}))
    obs, _ = env.reset(seed=seed)
    data = getattr(env.unwrapped, 'data', None)
    qpos = getattr(data, 'qpos', None)
    start_x = float(qpos[0]) if qpos is not None and len(qpos) else 0.0
    transitions = []
    total = 0.0
    forward_total = 0.0
    survive_total = 0.0
    ctrl_total = 0.0
    velocity_total = 0.0
    velocity_steps = 0
    for _ in range(max_steps):
        action = clip_action(policy.get_action(obs, deterministic=not train), env.action_space)
        next_obs, reward, terminated, truncated, info = env.step(action)
        transitions.append((obs, action, float(reward), next_obs, terminated or truncated))
        total += float(reward)
        forward_total += float(info.get('reward_forward', 0.0))
        survive_total += float(info.get('reward_survive', 0.0))
        ctrl_total += float(info.get('reward_ctrl', 0.0))
        if 'x_velocity' in info:
            velocity_total += float(info['x_velocity'])
            velocity_steps += 1
        obs = next_obs
        if terminated or truncated:
            break
    qpos = getattr(getattr(env.unwrapped, 'data', None), 'qpos', None)
    end_x = float(qpos[0]) if qpos is not None and len(qpos) else start_x
    env.close()
    steps = len(transitions)
    diagnostics = {
        'episode_length_mean': float(steps),
        'forward_return_mean': float(forward_total),
        'survive_return_mean': float(survive_total),
        'ctrl_return_mean': float(ctrl_total),
        'x_displacement_mean': float(end_x - start_x),
        'x_velocity_mean': float(velocity_total / max(1, velocity_steps)),
    }
    return total, transitions, steps, diagnostics


def _evaluate(
    policies,
    env_name,
    max_steps,
    seed,
    episodes,
    heterogeneity,
    heterogeneity_mode,
    env_kwargs,
):
    rewards = []
    diagnostics = []
    total_steps = 0
    for wid, policy in enumerate(policies):
        for ep in range(episodes):
            reward, _, steps, episode_diagnostics = _rollout(
                policy, env_name, max_steps, seed + wid * 1000 + ep,
                wid, heterogeneity, heterogeneity_mode, train=False,
                env_kwargs=env_kwargs)
            rewards.append(reward)
            diagnostics.append(episode_diagnostics)
            total_steps += steps
    diagnostic_mean = {
        key: float(np.mean([row[key] for row in diagnostics]))
        for key in diagnostics[0]
    } if diagnostics else {}
    return float(np.mean(rewards)), float(np.std(rewards)), int(total_steps), diagnostic_mean


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
    policies = [SACPolicy(info['state_dim'], info['action_dim']) for _ in range(args.num_workers)]
    global_actor_state = _actor_state(policies[0])
    for policy in policies:
        _load_actor_state(policy, global_actor_state)
    critic_optimizers = [
        optim.Adam(list(p.critic1.parameters()) + list(p.critic2.parameters()), lr=args.lr)
        for p in policies
    ]
    actor_lr = args.actor_lr if args.actor_lr > 0 else args.lr
    actor_optimizers = [optim.Adam(p.actor.parameters(), lr=actor_lr) for p in policies]
    alpha_optimizers = [optim.Adam([p.log_alpha], lr=args.lr) for p in policies]
    buffers = [HybridReplayBuffer(args.buffer_size) for _ in range(args.num_workers)]
    critic_warmup_remaining = [
        max(0, args.post_sync_critic_warmup_updates) for _ in range(args.num_workers)
    ]

    exp = args.exp_name or f"{args.baseline_mode}_{args.env}_s{args.seed}"
    run_dir = Path(args.log_dir) / exp
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / 'metrics.csv'
    metadata = vars(args).copy()
    metadata.update({'mode': args.baseline_mode, 'algorithm': 'SAC', 'env': args.env, 'seed': args.seed})
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with metrics_path.open('w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=METRIC_FIELDS).writeheader()

    start = time.time()
    total_steps = 0
    total_updates = 0
    best_eval = -np.inf
    best_eval_std = 0.0
    best_actor_state = None
    rollback_count = 0
    scores = np.zeros(args.num_workers, dtype=np.float64)
    env_kwargs = {}
    if args.env == 'Walker2d-v5':
        env_kwargs = {
            'healthy_reward': float(args.walker_healthy_reward),
            'forward_reward_weight': float(args.walker_forward_reward_weight),
        }
    elif args.env == 'Hopper-v5':
        env_kwargs = {
            'healthy_reward': float(args.hopper_healthy_reward),
            'forward_reward_weight': float(args.hopper_forward_reward_weight),
        }
    global_eval, global_eval_std, initial_eval_steps, global_diagnostics = _evaluate(
        policies, args.env, args.max_episode_steps, args.seed + 70_000_003,
        args.aggregation_eval_episodes, args.client_heterogeneity,
        args.client_heterogeneity_mode, env_kwargs)
    total_steps += initial_eval_steps
    round_idx = 0
    communication_round = 0
    while (
        (args.target_env_steps > 0 and total_steps < args.target_env_steps)
        or (args.target_env_steps <= 0 and round_idx < args.rounds)
    ):
        round_start = time.time()
        worker_rewards = []
        round_update_counts = []
        for wid, policy in enumerate(policies):
            reward, transitions, steps, _ = _rollout(
                policy, args.env, args.max_episode_steps, args.seed + round_idx * 10000 + wid,
                wid, args.client_heterogeneity, args.client_heterogeneity_mode, train=True,
                env_kwargs=env_kwargs)
            worker_rewards.append(reward)
            scores[wid] = 0.9 * scores[wid] + reward
            total_steps += steps
            for item in transitions:
                buffers[wid].add_rl_data(*item)
            local_updates = 0
            if len(buffers[wid]) >= args.batch_size:
                adaptive_updates = int(math.ceil(steps * max(0.0, args.update_to_data_ratio)))
                local_updates = min(
                    max(args.updates, adaptive_updates),
                    max(args.updates, args.max_updates_per_round),
                )
                for _ in range(local_updates):
                    batch = buffers[wid].sample(args.batch_size, ea_batch_ratio=0.0)
                    for key in ('observations', 'actions', 'rewards', 'next_observations', 'dones'):
                        batch[key] = np.asarray(batch[key], dtype=np.float32)
                    loss = policy.optimize_step(
                        batch,
                        critic_optimizers[wid],
                        actor_optimizers[wid],
                        alpha_optimizers[wid],
                        args.gamma,
                        args.tau,
                        grad_clip=10.0,
                        update_actor=critic_warmup_remaining[wid] <= 0,
                    )
                    if not torch.isfinite(loss):
                        continue
                    critic_warmup_remaining[wid] = max(
                        0, critic_warmup_remaining[wid] - 1)
                    total_updates += 1
            round_update_counts.append(local_updates)
        budget_reached = args.target_env_steps > 0 and total_steps >= args.target_env_steps
        aggregate_now = (
            args.baseline_mode != 'independent_sac'
            and (
                round_idx % max(1, args.aggregation_interval) == 0
                or budget_reached
                or (args.target_env_steps <= 0 and round_idx == args.rounds - 1)
            )
        )
        entropy = 0.0
        checkpoint_retained = 0
        proposal_mean = np.nan
        proposal_std = np.nan
        if aggregate_now:
            communication_round += 1
            aggregated_state, entropy = _aggregate_actor_state(
                policies, scores, args.baseline_mode, args.federated_temperature)
            proposed_state = _blend_actor_states(
                global_actor_state, aggregated_state, args.server_learning_rate)
            for optimizer in actor_optimizers:
                optimizer.state.clear()
            for policy in policies:
                _load_actor_state(policy, proposed_state)
            proposal_mean, proposal_std, proposal_eval_steps, proposal_diagnostics = _evaluate(
                policies, args.env, args.max_episode_steps, args.seed + 70_000_003,
                args.aggregation_eval_episodes, args.client_heterogeneity,
                args.client_heterogeneity_mode, env_kwargs)
            total_steps += proposal_eval_steps
            if (
                args.disable_deployment_rollback
                or proposal_mean >= global_eval - args.deployment_rollback_tolerance
            ):
                global_actor_state = proposed_state
                global_eval = proposal_mean
                global_eval_std = proposal_std
                global_diagnostics = proposal_diagnostics
            else:
                checkpoint_retained = 1
                rollback_count += 1
            for policy in policies:
                _load_actor_state(policy, global_actor_state)
            critic_warmup_remaining = [
                max(0, args.post_sync_critic_warmup_updates)
                for _ in range(args.num_workers)
            ]
        elif args.baseline_mode == 'independent_sac':
            proposal_mean, proposal_std, proposal_eval_steps, proposal_diagnostics = _evaluate(
                policies, args.env, args.max_episode_steps, args.seed + 70_000_003,
                args.aggregation_eval_episodes, args.client_heterogeneity,
                args.client_heterogeneity_mode, env_kwargs)
            total_steps += proposal_eval_steps
            global_eval, global_eval_std = proposal_mean, proposal_std
            global_diagnostics = proposal_diagnostics

        if (
            round_idx % args.eval_interval == 0
            or budget_reached
            or (args.target_env_steps <= 0 and round_idx == args.rounds - 1)
        ):
            if not np.isfinite(proposal_mean):
                proposal_mean, proposal_std, proposal_eval_steps, proposal_diagnostics = _evaluate(
                    policies, args.env, args.max_episode_steps, args.seed + 70_000_003,
                    args.aggregation_eval_episodes, args.client_heterogeneity,
                    args.client_heterogeneity_mode, env_kwargs)
                total_steps += proposal_eval_steps
            deploy_mean, deploy_std = global_eval, global_eval_std
            if deploy_mean >= best_eval:
                best_eval = deploy_mean
                best_eval_std = deploy_std
                best_actor_state = {key: value.clone() for key, value in global_actor_state.items()}
            _write_row(metrics_path, {
                'generation': round_idx,
                'communication_round': communication_round,
                'total_env_steps': total_steps,
                'eval_reward_mean': deploy_mean,
                'eval_reward_std': deploy_std,
                'eval_ea_mean': proposal_mean,
                'eval_ea_std': proposal_std,
                'best_fitness': best_eval,
                'mean_fitness': float(np.mean(worker_rewards)),
                'rl_steps': total_updates,
                'buffer_size': int(sum(len(b) for b in buffers)),
                'gen_time': time.time() - round_start,
                'total_time': time.time() - start,
                'sync_applied': int(aggregate_now),
                'fitness_std': float(np.std(worker_rewards)),
                'selected_clients': args.num_workers if aggregate_now else 0,
                'fed_round_applied': int(aggregate_now),
                'archive_best': best_eval,
                'deployable_eval_mean': deploy_mean,
                'deployable_eval_std': deploy_std,
                'aggregation_entropy': entropy,
                'client_reward_mean': float(np.mean(worker_rewards)),
                'client_reward_std': float(np.std(worker_rewards)),
                'deployment_rollback': checkpoint_retained,
                'deployment_rollback_count': rollback_count,
                'candidate_eval_mean': proposal_mean,
                'candidate_eval_std': proposal_std,
                'local_updates_per_worker': float(np.mean(round_update_counts)),
                **{f'eval_{key}': value for key, value in global_diagnostics.items()},
            })
            checkpoint_note = " deploy=checkpoint" if checkpoint_retained else ""
            print(
                f"{exp}: round={round_idx} steps={total_steps} "
                f"candidate={proposal_mean:.2f}+/-{proposal_std:.2f} "
                f"deploy={deploy_mean:.2f}{checkpoint_note}",
                flush=True,
            )
        round_idx += 1
    print(metrics_path)


if __name__ == '__main__':
    main()
