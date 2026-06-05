"""RL Learner: Reinforcement Learning optimizer for Dist-ERL."""

import ray
import torch
import torch.nn.utils
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Optional
from .utils.environment import get_env_info, make_env
from .utils.replay_buffer import HybridReplayBuffer
from .utils.policies import DDPGPolicy, TD3Policy, PPOPolicy
from .utils.policy_utils import ActorEvaluator

_GENOTYPE_PREFIXES = ('actor.', 'critic.', 'critic1.', 'critic2.')


@ray.remote(num_gpus=0)
class RLLearner:
    """RL Learner - Trajectory reproduction, gradient optimization (DDPG/TD3/PPO)"""

    def __init__(self,
                 env_name: str = "Ant-v2",
                 algorithm: str = "DDPG",
                 buffer_size: int = 1000000,
                 batch_size: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 ea_batch_ratio: float = 0.5,
                 policy_exploration_noise: float = 0.1):
        self.env_name = env_name
        self.algorithm = algorithm.upper()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.ea_batch_ratio = ea_batch_ratio
        self.policy_exploration_noise = policy_exploration_noise

        env_info = get_env_info(env_name)
        self.state_dim = env_info.get('state_dim')
        self.action_dim = env_info.get('action_dim')
        self._actor_eval = ActorEvaluator(
            self.state_dim, self.action_dim, algorithm=self.algorithm)

        self.replay_buffer = HybridReplayBuffer(buffer_size)
        self.policy = self._initialize_policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.training_steps = 0

    def _initialize_policy(self):
        if self.state_dim is None or self.action_dim is None:
            raise ValueError("Unable to infer environment dimensions for policy initialization")

        if self.algorithm == "DDPG":
            return DDPGPolicy(state_dim=self.state_dim, action_dim=self.action_dim)
        if self.algorithm == "TD3":
            return TD3Policy(state_dim=self.state_dim, action_dim=self.action_dim)
        if self.algorithm == "PPO":
            return PPOPolicy(state_dim=self.state_dim, action_dim=self.action_dim)
        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _is_genotype_key(self, name: str) -> bool:
        return name.startswith(_GENOTYPE_PREFIXES) and not name.startswith('target_')

    def add_rl_experience(self, trajectory: Dict[str, Any]):
        observations = trajectory['observations']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        dones = trajectory['dones']

        n = min(len(actions), len(rewards), len(dones))
        for i in range(n):
            self.replay_buffer.add_rl_data(
                observation=observations[i],
                action=actions[i],
                reward=rewards[i],
                next_observation=observations[i + 1] if i + 1 < len(observations) else observations[i],
                done=dones[i],
            )

    def add_ea_experience(self, trajectory: Dict[str, Any]):
        observations = trajectory['observations']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        dones = trajectory['dones']

        n = min(len(actions), len(rewards), len(dones))
        for i in range(n):
            self.replay_buffer.add_reproduced_ea_transition(
                observation=observations[i],
                action=actions[i],
                reward=rewards[i],
                next_observation=observations[i + 1] if i + 1 < len(observations) else observations[i],
                done=dones[i],
                seed=trajectory.get('seed', 0),
                generation=trajectory.get('generation', 0),
            )

    def update_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = self.replay_buffer.sample(self.batch_size, ea_batch_ratio=self.ea_batch_ratio)
        for key in ('observations', 'actions', 'rewards', 'next_observations', 'dones'):
            arr = np.asarray(batch[key], dtype=np.float32)
            if not np.isfinite(arr).all():
                return 0.0

        loss = self.policy.update(batch, self.gamma, self.tau)
        if not torch.isfinite(loss):
            return 0.0

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1
        return loss.item() if hasattr(loss, 'item') else float(loss)

    def get_policy_weights(self) -> Dict[str, np.ndarray]:
        """Export genotype weights for EA (no target networks)."""
        weights = {}
        for name, param in self.policy.state_dict().items():
            if self._is_genotype_key(name):
                weights[name] = param.cpu().detach().numpy()
        return weights

    def load_policy_weights(self, weights: Dict[str, np.ndarray]):
        state_dict = self.policy.state_dict()
        for name, array in weights.items():
            if name in state_dict:
                state_dict[name] = torch.from_numpy(array.copy())
        self.policy.load_state_dict(state_dict)
        if hasattr(self.policy, 'target_actor'):
            self.policy.target_actor.load_state_dict(self.policy.actor.state_dict())
        if hasattr(self.policy, 'target_critic'):
            self.policy.target_critic.load_state_dict(self.policy.critic.state_dict())
        if hasattr(self.policy, 'target_critic1'):
            self.policy.target_critic1.load_state_dict(self.policy.critic1.state_dict())
            self.policy.target_critic2.load_state_dict(self.policy.critic2.state_dict())

    def warm_start_actor_from_elite(self, elite: Dict[str, Any], blend: float = 1.0) -> int:
        """Blend the RL actor toward an EA elite without touching critic state."""
        weights = elite.get('weights', {}) if elite else {}
        if not weights:
            return 0
        blend = float(np.clip(blend, 0.0, 1.0))
        state_dict = self.policy.state_dict()
        loaded = 0
        for name, current in list(state_dict.items()):
            if not name.startswith('actor.') or name not in weights:
                continue
            elite_tensor = torch.from_numpy(np.array(weights[name], copy=True)).to(current.dtype)
            if elite_tensor.shape != current.shape:
                continue
            state_dict[name] = current * (1.0 - blend) + elite_tensor * blend
            loaded += 1
        if loaded:
            self.policy.load_state_dict(state_dict)
            if hasattr(self.policy, 'target_actor'):
                self.policy.target_actor.load_state_dict(self.policy.actor.state_dict())
        return loaded

    def perturb_policy(self, noise_scale: float = 0.02) -> None:
        with torch.no_grad():
            for name, param in self.policy.named_parameters():
                if name.startswith('actor.'):
                    param.add_(torch.randn_like(param) * noise_scale)

    def reset_actor_tail(self) -> None:
        if hasattr(self.policy, 'reset_actor_last_layers'):
            self.policy.reset_actor_last_layers()

    def boost_rl_exploration(self, factor: float = 0.85) -> None:
        if self.algorithm in ('TD3', 'DDPG'):
            self.policy_exploration_noise = min(0.5, self.policy_exploration_noise / factor)
            if self.algorithm == 'TD3' and hasattr(self.policy, 'policy_noise'):
                self.policy.policy_noise = min(0.5, self.policy.policy_noise / factor)
            self.perturb_policy(0.03)

    def _policy_get_action(self, obs: np.ndarray) -> np.ndarray:
        if self.algorithm in ('TD3', 'DDPG'):
            return self.policy.get_action(obs, exploration_noise=self.policy_exploration_noise)
        return self.policy.get_action(obs)

    def evaluate_policy_actor_aligned(self, num_episodes: int = 3,
                                      max_episode_steps: int = 1000,
                                      seed: Optional[int] = None) -> Dict[str, float]:
        env = make_env(self.env_name, max_episode_steps=max_episode_steps)
        self._actor_eval.load_weights(self.get_policy_weights())
        rewards = []

        for episode in range(num_episodes):
            if seed is not None:
                obs, _ = env.reset(seed=int(seed + episode + 1000))
            else:
                obs, _ = env.reset()
            total_reward = 0.0
            done = False
            truncated = False
            steps = 0
            while not (done or truncated) and steps < max_episode_steps:
                action = self._actor_eval.get_action(obs, env.action_space)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
            rewards.append(total_reward)

        env.close()
        return {
            'eval_reward_mean': float(np.mean(rewards)) if rewards else 0.0,
            'eval_reward_std': float(np.std(rewards)) if rewards else 0.0,
            'eval_steps': int(len(rewards) * max_episode_steps),
        }

    def reproduce_from_elites(self, elite_individuals: List[Dict[str, Any]],
                              max_episode_steps: int = 1000) -> int:
        if not elite_individuals:
            return 0

        env = make_env(self.env_name, max_episode_steps=max_episode_steps)
        reproduced = 0

        for elite in elite_individuals:
            seed = int(elite['seed'])
            weights = elite['weights']
            self._actor_eval.load_weights(weights)
            obs, _ = env.reset(seed=seed)
            observations, actions, rewards, dones = [np.array(obs, copy=True)], [], [], []
            done = False
            truncated = False
            step = 0

            while not (done or truncated) and step < max_episode_steps:
                action = self._actor_eval.get_action(obs, env.action_space)
                actions.append(action)
                obs, reward, done, truncated, _ = env.step(action)
                observations.append(np.array(obs, copy=True))
                rewards.append(reward)
                dones.append(done or truncated)
                step += 1

            if len(actions) < 1:
                continue
            self.add_ea_experience({
                'observations': np.array(observations),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'dones': np.array(dones),
                'seed': seed,
                'generation': 0,
            })
            reproduced += 1

        env.close()
        return reproduced

    def _clip_action(self, action: np.ndarray, action_space):
        if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
            return np.clip(action, action_space.low, action_space.high)
        if hasattr(action_space, 'n'):
            if isinstance(action, np.ndarray):
                return int(np.argmax(action))
            return int(action)
        return action

    def collect_rl_trajectories(self, num_episodes: int = 1, max_episode_steps: int = 1000,
                                  seed: Optional[int] = None):
        env = make_env(self.env_name, max_episode_steps=max_episode_steps)
        total_steps = 0
        total_reward = 0.0

        for episode in range(num_episodes):
            if seed is not None:
                obs, _ = env.reset(seed=int(seed + episode))
            else:
                obs, _ = env.reset()

            trajectory = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
            done = False
            truncated = False
            step_count = 0
            episode_reward = 0.0
            trajectory['observations'].append(obs)

            while not (done or truncated) and step_count < max_episode_steps:
                action = self._policy_get_action(obs)
                action = self._clip_action(action, env.action_space)
                next_obs, reward, done, truncated, _ = env.step(action)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['dones'].append(done or truncated)
                trajectory['observations'].append(next_obs)
                obs = next_obs
                episode_reward += reward
                step_count += 1

            self.add_rl_experience({
                'observations': np.array(trajectory['observations']),
                'actions': np.array(trajectory['actions']),
                'rewards': np.array(trajectory['rewards']),
                'dones': np.array(trajectory['dones']),
            })
            total_steps += step_count
            total_reward += episode_reward

        env.close()
        return {
            'total_steps': total_steps,
            'avg_reward': (total_reward / num_episodes) if num_episodes else 0.0,
        }

    def evaluate_policy(self, num_episodes: int = 3, max_episode_steps: int = 1000,
                        seed: Optional[int] = None):
        env = make_env(self.env_name, max_episode_steps=max_episode_steps)
        rewards = []
        total_steps = 0

        for episode in range(num_episodes):
            if seed is not None:
                obs, _ = env.reset(seed=int(seed + episode + 1000))
            else:
                obs, _ = env.reset()

            done = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0

            while not (done or truncated) and episode_steps < max_episode_steps:
                action = self.policy.get_action(obs)
                action = self._clip_action(action, env.action_space)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                episode_steps += 1

            rewards.append(episode_reward)
            total_steps += episode_steps

        env.close()
        return {
            'eval_reward_mean': float(np.mean(rewards)) if rewards else 0.0,
            'eval_reward_std': float(np.std(rewards)) if rewards else 0.0,
            'eval_steps': total_steps,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            'training_steps': self.training_steps,
            'buffer_size': len(self.replay_buffer),
            'rl_buffer_size': self.replay_buffer.rl_size,
            'ea_seeds_size': self.replay_buffer.ea_seeds_size,
            'ea_transitions_size': self.replay_buffer.ea_transitions_size,
            'algorithm': self.algorithm,
            'policy_exploration_noise': self.policy_exploration_noise,
        }
