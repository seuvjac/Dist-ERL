"""Federated clients and aggregation utilities for EA-guided federated RL."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import ray
import torch
import torch.optim as optim

from src.utils.environment import get_env_info, make_env
from src.utils.policies import DDPGPolicy, FSACPolicy, PPOPolicy, SACPolicy, TD3Policy
from src.utils.policy_utils import ActorEvaluator, clip_action, encode_action_for_buffer
from src.utils.replay_buffer import HybridReplayBuffer

_GENOTYPE_PREFIXES = ('actor.',)


def _aggregation_weights(
    scores: Sequence[float],
    mode: str = 'fitness',
    temperature: float = 1.0,
) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=np.float64)
    if scores_arr.size == 0:
        return scores_arr
    if mode == 'uniform' or not np.isfinite(scores_arr).all():
        return np.ones_like(scores_arr) / len(scores_arr)
    if mode == 'softmax':
        temp = max(1e-6, float(temperature))
        logits = (scores_arr - scores_arr.max()) / temp
        coeffs = np.exp(np.clip(logits, -60.0, 0.0))
        total = float(coeffs.sum())
        if total <= 1e-8:
            return np.ones_like(scores_arr) / len(scores_arr)
        return coeffs / total
    shifted = scores_arr - scores_arr.min()
    if float(shifted.sum()) <= 1e-8:
        return np.ones_like(scores_arr) / len(scores_arr)
    return shifted / shifted.sum()


def aggregate_weight_dicts(
    client_weights: Sequence[Dict[str, np.ndarray]],
    scores: Sequence[float],
    mode: str = 'fitness',
    temperature: float = 1.0,
    min_score: Optional[float] = None,
    base_weights: Optional[Dict[str, np.ndarray]] = None,
    delta_clip_norm: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Aggregate client weights using score-normalized, optionally clipped deltas."""
    if not client_weights:
        return {}
    filtered_weights = []
    filtered_scores = []
    for weights, score in zip(client_weights, scores):
        if min_score is not None and float(score) < float(min_score):
            continue
        filtered_weights.append(weights)
        filtered_scores.append(score)
    if not filtered_weights:
        return {}
    score_arr = np.asarray(filtered_scores, dtype=np.float64)
    if score_arr.size > 1 and np.isfinite(score_arr).all():
        score_arr = (score_arr - score_arr.mean()) / (score_arr.std() + 1e-8)
        if mode == 'softmax':
            temperature = max(1e-6, temperature / max(1.0, float(np.std(filtered_scores))))
    coeffs = _aggregation_weights(score_arr, mode=mode, temperature=temperature)
    keys = sorted(set.intersection(*(set(w.keys()) for w in filtered_weights)))
    aggregated: Dict[str, np.ndarray] = {}
    for key in keys:
        base = filtered_weights[0][key]
        if base_weights is not None and key in base_weights and base_weights[key].shape == base.shape:
            center = base_weights[key].astype(np.float64)
        else:
            center = np.zeros_like(base, dtype=np.float64)
        acc_delta = np.zeros_like(base, dtype=np.float64)
        for coeff, weights in zip(coeffs, filtered_weights):
            arr = weights[key]
            if arr.shape != base.shape:
                continue
            delta = arr.astype(np.float64) - center
            if delta_clip_norm is not None and delta_clip_norm > 0:
                norm = float(np.linalg.norm(delta))
                if norm > delta_clip_norm:
                    delta = delta * (float(delta_clip_norm) / (norm + 1e-8))
            acc_delta += float(coeff) * delta
        aggregated[key] = (center + acc_delta).astype(base.dtype)
    return aggregated


def weight_entropy(scores: Sequence[float], mode: str = 'fitness', temperature: float = 1.0) -> float:
    coeffs = _aggregation_weights(scores, mode=mode, temperature=temperature)
    if coeffs.size == 0:
        return 0.0
    coeffs = np.clip(coeffs, 1e-12, 1.0)
    return float(-(coeffs * np.log(coeffs)).sum())


@ray.remote(num_gpus=0)
class FederatedClient:
    """
    Federated RL client.

    Each client owns a private environment stream and replay buffer.  It can
    evaluate server-provided EA candidates and locally refine a policy, but it
    only uploads model weights and scalar summaries to the server.
    """

    def __init__(
        self,
        client_id: int,
        env_name: str,
        algorithm: str = 'FSAC',
        buffer_size: int = 1000000,
        batch_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_episode_steps: int = 1000,
        heterogeneity: float = 0.2,
        heterogeneity_mode: str = 'reward_action_noise',
        policy_exploration_noise: float = 0.1,
    ):
        self.client_id = int(client_id)
        self.env_name = env_name
        self.algorithm = algorithm.upper()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.max_episode_steps = max_episode_steps
        self.heterogeneity = max(0.0, float(heterogeneity))
        self.heterogeneity_mode = heterogeneity_mode
        self.policy_exploration_noise = policy_exploration_noise

        env_info = get_env_info(env_name)
        self.state_dim = env_info['state_dim']
        self.action_dim = env_info['action_dim']
        self.is_discrete = hasattr(env_info.get('action_space'), 'n')
        self.reward_scale = 1.0 + self.heterogeneity * ((self.client_id % 5) - 2) / 4.0
        self.action_noise = self.heterogeneity * 0.05 * (1 + (self.client_id % 3))
        self.seed_offset = self.client_id * 10007

        self.replay_buffer = HybridReplayBuffer(buffer_size)
        self.policy = self._initialize_policy()
        if self.algorithm == 'SAC':
            self.critic_optimizer = optim.Adam(
                list(self.policy.critic1.parameters()) + list(self.policy.critic2.parameters()),
                lr=lr,
            )
            self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=lr)
            self.alpha_optimizer = optim.Adam([self.policy.log_alpha], lr=lr)
            self.optimizer = None
        else:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self._actor_eval = ActorEvaluator(
            self.state_dim, self.action_dim, algorithm=self.algorithm, discrete=self.is_discrete)
        self.training_steps = 0

    def _initialize_policy(self):
        if self.algorithm == 'DDPG':
            return DDPGPolicy(self.state_dim, self.action_dim)
        if self.algorithm == 'TD3':
            return TD3Policy(self.state_dim, self.action_dim)
        if self.algorithm == 'SAC':
            return SACPolicy(self.state_dim, self.action_dim)
        if self.algorithm == 'FSAC':
            return FSACPolicy(self.state_dim, self.action_dim)
        if self.algorithm == 'PPO':
            return PPOPolicy(self.state_dim, self.action_dim, discrete=self.is_discrete)
        raise ValueError(f'Unsupported algorithm: {self.algorithm}')

    def _is_genotype_key(self, name: str) -> bool:
        return name.startswith(_GENOTYPE_PREFIXES) and not name.startswith('target_')

    def _export_weights(self) -> Dict[str, np.ndarray]:
        exported = {}
        for name, param in self.policy.state_dict().items():
            if self._is_genotype_key(name):
                exported[name] = param.cpu().detach().numpy()
        return exported

    def _load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        state = self.policy.state_dict()
        for name, array in weights.items():
            if name in state and state[name].shape == array.shape:
                state[name] = torch.from_numpy(array.copy()).to(dtype=state[name].dtype)
        self.policy.load_state_dict(state)
        if hasattr(self.policy, 'target_actor'):
            self.policy.target_actor.load_state_dict(self.policy.actor.state_dict())
        if hasattr(self.policy, 'target_critic'):
            self.policy.target_critic.load_state_dict(self.policy.critic.state_dict())
        if hasattr(self.policy, 'target_critic1'):
            self.policy.target_critic1.load_state_dict(self.policy.critic1.state_dict())
            self.policy.target_critic2.load_state_dict(self.policy.critic2.state_dict())

    def _policy_action(self, observation: np.ndarray) -> np.ndarray:
        if self.algorithm in ('DDPG', 'TD3'):
            return self.policy.get_action(observation, exploration_noise=self.policy_exploration_noise)
        if self.algorithm in ('PPO', 'FSAC', 'SAC'):
            return self.policy.get_action(observation)
        return self.policy.get_action(observation)

    def evaluate_weights(
        self,
        weights: Dict[str, np.ndarray],
        seed: Optional[int] = None,
        num_episodes: int = 1,
    ) -> Dict[str, Any]:
        env = make_env(
            self.env_name,
            max_episode_steps=self.max_episode_steps,
            client_id=self.client_id,
            heterogeneity=self.heterogeneity,
            heterogeneity_mode=self.heterogeneity_mode,
        )
        self._actor_eval.load_weights(weights)
        rewards = []
        total_steps = 0
        for ep in range(num_episodes):
            reset_seed = None if seed is None else int(seed + self.seed_offset + ep)
            obs, _ = env.reset(seed=reset_seed)
            total = 0.0
            done = False
            truncated = False
            steps = 0
            while not (done or truncated) and steps < self.max_episode_steps:
                action = self._actor_eval.get_action(obs, env.action_space)
                if self.action_noise > 0 and not hasattr(env.action_space, 'n'):
                    action = clip_action(
                        action + np.random.normal(0, self.action_noise, np.shape(action)),
                        env.action_space,
                    )
                obs, reward, done, truncated, _ = env.step(action)
                total += float(reward) * self.reward_scale
                steps += 1
            total_steps += steps
            rewards.append(total)
        env.close()
        return {
            'client_id': self.client_id,
            'fitness': float(np.mean(rewards)) if rewards else 0.0,
            'fitness_std': float(np.std(rewards)) if rewards else 0.0,
            'total_steps': int(total_steps),
        }

    def local_train(
        self,
        weights: Dict[str, np.ndarray],
        num_episodes: int,
        updates: int,
        seed: Optional[int] = None,
        critic_warmup_updates: int = 0,
    ) -> Dict[str, Any]:
        self._load_weights(weights)
        if self.algorithm == 'SAC':
            self.actor_optimizer.state.clear()
        env = make_env(
            self.env_name,
            max_episode_steps=self.max_episode_steps,
            client_id=self.client_id,
            heterogeneity=self.heterogeneity,
            heterogeneity_mode=self.heterogeneity_mode,
        )
        total_reward = 0.0
        total_steps = 0

        for ep in range(num_episodes):
            reset_seed = None if seed is None else int(seed + self.seed_offset + ep)
            obs, _ = env.reset(seed=reset_seed)
            observations = [np.array(obs, copy=True)]
            actions, rewards, dones = [], [], []
            done = False
            truncated = False
            steps = 0
            while not (done or truncated) and steps < self.max_episode_steps:
                action = clip_action(self._policy_action(obs), env.action_space)
                next_obs, reward, done, truncated, _ = env.step(action)
                scaled_reward = float(reward) * self.reward_scale
                actions.append(encode_action_for_buffer(
                    action, env.action_space, self.action_dim, self.algorithm))
                rewards.append(scaled_reward)
                dones.append(done or truncated)
                observations.append(np.array(next_obs, copy=True))
                obs = next_obs
                total_reward += scaled_reward
                total_steps += 1
                steps += 1

            for idx in range(len(actions)):
                self.replay_buffer.add_rl_data(
                    observations[idx], actions[idx], rewards[idx],
                    observations[idx + 1], dones[idx],
                )

        losses = []
        for update_idx in range(updates):
            if len(self.replay_buffer) < self.batch_size:
                continue
            batch = self.replay_buffer.sample(self.batch_size, ea_batch_ratio=0.0)
            for key in ('observations', 'actions', 'rewards', 'next_observations', 'dones'):
                batch[key] = np.asarray(batch[key], dtype=np.float32)
            if self.algorithm == 'SAC':
                loss = self.policy.optimize_step(
                    batch,
                    self.critic_optimizer,
                    self.actor_optimizer,
                    self.alpha_optimizer,
                    self.gamma,
                    self.tau,
                    grad_clip=10.0,
                    update_actor=update_idx >= max(0, int(critic_warmup_updates)),
                )
            else:
                loss = self.policy.update(batch, self.gamma, self.tau)
                if not torch.isfinite(loss):
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()
                if hasattr(self.policy, 'sync_target'):
                    self.policy.sync_target(self.tau)
                if hasattr(self.policy, 'decay_epsilon'):
                    self.policy.decay_epsilon()
            if not torch.isfinite(loss):
                continue
            losses.append(float(loss.item() if hasattr(loss, 'item') else loss))
            self.training_steps += 1

        env.close()
        avg_reward = total_reward / max(1, num_episodes)
        return {
            'client_id': self.client_id,
            'weights': self._export_weights(),
            'avg_reward': float(avg_reward),
            'total_steps': int(total_steps),
            'training_steps': int(self.training_steps),
            'updates_applied': int(len(losses)),
            'buffer_size': int(len(self.replay_buffer)),
            'loss_mean': float(np.mean(losses)) if losses else 0.0,
            'epsilon': float(getattr(self.policy, 'exploration_epsilon', 0.0)),
            'reward_scale': float(self.reward_scale),
            'action_noise': float(self.action_noise),
        }
