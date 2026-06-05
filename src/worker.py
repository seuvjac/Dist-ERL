"""Rollout Worker: Distributed evaluation worker for FedEvoRL baselines."""

import ray
import numpy as np
from typing import Dict, Any

from .utils.environment import make_env, get_env_info
from .utils.policy_utils import ActorEvaluator, clip_action


@ray.remote
class RolloutWorker:
    """Rollout Worker - Parallel sampling, fitness evaluation, returns only (Seed, Score)"""

    def __init__(self, env_name: str = "Ant-v2", max_episode_steps: int = 1000,
                 algorithm: str = 'DDPG', state_dim: int = None, action_dim: int = None):
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.algorithm = algorithm
        self.env = None
        if state_dim is None or action_dim is None:
            env_info = get_env_info(env_name)
            state_dim = env_info['state_dim']
            action_dim = env_info['action_dim']
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._evaluator = ActorEvaluator(
            self.state_dim, self.action_dim, algorithm=algorithm)
        self._initialize_env()

    def _initialize_env(self):
        self.env = make_env(self.env_name, max_episode_steps=self.max_episode_steps)

    def evaluate(self, individual) -> float:
        seed = individual.seed
        weights = individual.weights
        self._evaluator.load_weights(weights)

        obs, _ = self.env.reset(seed=seed)
        total_reward = 0.0
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated) and step_count < self.max_episode_steps:
            action = self._evaluator.get_action(obs, self.env.action_space)
            obs, reward, done, truncated, _ = self.env.step(action)
            total_reward += reward
            step_count += 1

        return float(total_reward)

    def evaluate_individual(self, individual, num_episodes: int = 10, max_episode_steps: int = 1000) -> Dict[str, float]:
        rewards = []
        self._evaluator.load_weights(individual.weights)
        for ep in range(num_episodes):
            obs, _ = self.env.reset(seed=int(individual.seed + ep))
            total_reward = 0.0
            done = False
            truncated = False
            step_count = 0
            while not (done or truncated) and step_count < max_episode_steps:
                action = self._evaluator.get_action(obs, self.env.action_space)
                obs, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward
                step_count += 1
            rewards.append(total_reward)

        return {
            'eval_reward_mean': float(np.mean(rewards)),
            'eval_reward_std': float(np.std(rewards)),
            'eval_steps': int(sum(len(rewards) for _ in [1])),
        }

    def reproduce_trajectory(self, seed: int, weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        self._evaluator.load_weights(weights)
        obs, _ = self.env.reset(seed=seed)
        observations, actions, rewards, dones = [], [], [], []
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated) and step_count < self.max_episode_steps:
            action = self._evaluator.get_action(obs, self.env.action_space)
            observations.append(np.array(obs, copy=True))
            actions.append(action)
            obs, reward, done, truncated, _ = self.env.step(action)
            rewards.append(reward)
            dones.append(done or truncated)
            step_count += 1

        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'seed': seed,
        }

    def close(self):
        if self.env:
            self.env.close()
