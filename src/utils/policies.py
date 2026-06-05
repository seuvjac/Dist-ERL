"""RL Policy implementations for Dist-ERL."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple


class BasePolicy(nn.Module):
    """Base policy class"""

    def __init__(self, state_dim: int = 111, action_dim: int = 8):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get action for given observation"""
        raise NotImplementedError

    def update(self, batch: Dict[str, np.ndarray], gamma: float, tau: float) -> torch.Tensor:
        """Update policy parameters"""
        raise NotImplementedError


def _build_deterministic_actor(state_dim: int, action_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim),
        nn.Tanh(),
    )


def _build_q_critic(state_dim: int, action_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class DDPGPolicy(BasePolicy):
    """Deep Deterministic Policy Gradient (single critic, no delayed update)."""

    def __init__(self, state_dim: int = 111, action_dim: int = 8, hidden_dim: int = 256):
        super().__init__(state_dim, action_dim)
        self.actor = _build_deterministic_actor(state_dim, action_dim, hidden_dim)
        self.critic = _build_q_critic(state_dim, action_dim, hidden_dim)
        self.target_actor = _build_deterministic_actor(state_dim, action_dim, hidden_dim)
        self.target_critic = _build_q_critic(state_dim, action_dim, hidden_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def reset_actor_last_layers(self) -> None:
        linear_layers = [m for m in self.actor.modules() if isinstance(m, nn.Linear)]
        for layer in linear_layers[-2:]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        self.target_actor.load_state_dict(self.actor.state_dict())

    def _soft_update(self, tau: float) -> None:
        for target, source in ((self.target_actor, self.actor), (self.target_critic, self.critic)):
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    def get_action(self, observation: np.ndarray, exploration_noise: float = 0.0) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis, :]
        with torch.no_grad():
            action = self.actor(torch.from_numpy(obs).float()).cpu().numpy()
        if single:
            action = action[0]
        if exploration_noise > 0:
            action = action + np.random.normal(0, exploration_noise, action.shape).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
        return action

    def update(self, batch: Dict[str, np.ndarray], gamma: float, tau: float) -> torch.Tensor:
        observations = torch.from_numpy(batch['observations']).float()
        actions = torch.from_numpy(batch['actions']).float()
        rewards = torch.from_numpy(batch['rewards']).float()
        next_observations = torch.from_numpy(batch['next_observations']).float()
        dones = torch.from_numpy(batch['dones']).float()

        with torch.no_grad():
            next_actions = self.target_actor(next_observations)
            next_sa = torch.cat([next_observations, next_actions], dim=-1)
            target_q = self.target_critic(next_sa)
            target = rewards.unsqueeze(-1) + gamma * (1 - dones.unsqueeze(-1)) * target_q

        state_actions = torch.cat([observations, actions], dim=-1)
        critic_loss = F.mse_loss(self.critic(state_actions), target)

        actor_actions = self.actor(observations)
        actor_loss = -self.critic(torch.cat([observations, actor_actions], dim=-1)).mean()

        self._soft_update(tau)
        return critic_loss + actor_loss


class TD3Policy(BasePolicy):
    """Twin Delayed Deep Deterministic Policy Gradient"""

    def __init__(self, state_dim: int = 111, action_dim: int = 8, hidden_dim: int = 256):
        super().__init__(state_dim, action_dim)

        self.actor = _build_deterministic_actor(state_dim, action_dim, hidden_dim)
        self.critic1 = _build_q_critic(state_dim, action_dim, hidden_dim)
        self.critic2 = _build_q_critic(state_dim, action_dim, hidden_dim)
        self.target_actor = _build_deterministic_actor(state_dim, action_dim, hidden_dim)
        self.target_critic1 = _build_q_critic(state_dim, action_dim, hidden_dim)
        self.target_critic2 = _build_q_critic(state_dim, action_dim, hidden_dim)

        # Copy parameters
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.policy_freq = 2
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self._update_step = 0

    def reset_actor_last_layers(self) -> None:
        linear_layers = [m for m in self.actor.modules() if isinstance(m, nn.Linear)]
        for layer in linear_layers[-2:]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        self.target_actor.load_state_dict(self.actor.state_dict())

    def _soft_update(self, tau: float) -> None:
        for target, source in (
            (self.target_actor, self.actor),
            (self.target_critic1, self.critic1),
            (self.target_critic2, self.critic2),
        ):
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    def get_action(self, observation: np.ndarray, exploration_noise: float = 0.0) -> np.ndarray:
        """Deterministic policy with optional Gaussian exploration noise."""
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis, :]
        with torch.no_grad():
            action = self.actor(torch.from_numpy(obs).float()).cpu().numpy()
        if single:
            action = action[0]
        if exploration_noise > 0:
            action = action + np.random.normal(0, exploration_noise, action.shape).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
        return action

    def update(self, batch: Dict[str, np.ndarray], gamma: float, tau: float) -> torch.Tensor:
        """TD3 update with delayed policy and target smoothing."""
        observations = torch.from_numpy(batch['observations']).float()
        actions = torch.from_numpy(batch['actions']).float()
        rewards = torch.from_numpy(batch['rewards']).float()
        next_observations = torch.from_numpy(batch['next_observations']).float()
        dones = torch.from_numpy(batch['dones']).float()

        with torch.no_grad():
            next_actions = self.target_actor(next_observations)
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)

            next_state_actions = torch.cat([next_observations, next_actions], dim=-1)
            target_q1 = self.target_critic1(next_state_actions)
            target_q2 = self.target_critic2(next_state_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards.unsqueeze(-1) + gamma * (1 - dones.unsqueeze(-1)) * target_q

        state_actions = torch.cat([observations, actions], dim=-1)
        current_q1 = self.critic1(state_actions)
        current_q2 = self.critic2(state_actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self._update_step += 1
        actor_loss = torch.tensor(0.0, device=observations.device)
        if self._update_step % self.policy_freq == 0:
            actor_actions = self.actor(observations)
            state_actor_actions = torch.cat([observations, actor_actions], dim=-1)
            actor_loss = -self.critic1(state_actor_actions).mean()
            self._soft_update(tau)

        return critic_loss + actor_loss


class PPOPolicy(BasePolicy):
    """Proximal Policy Optimization"""

    def __init__(self, state_dim: int = 111, action_dim: int = 8, hidden_dim: int = 256):
        super().__init__(state_dim, action_dim)

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim * 2)  # Mean and log_std
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Sample action from policy"""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
            mean, log_std = self.actor(obs_tensor).chunk(2, dim=-1)
            std = log_std.exp()

            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            return action.squeeze(0).numpy()

    def update(self, batch: Dict[str, np.ndarray], gamma: float, tau: float) -> torch.Tensor:
        """PPO update (simplified)"""
        observations = torch.from_numpy(batch['observations']).float()
        actions = torch.from_numpy(batch['actions']).float()
        rewards = torch.from_numpy(batch['rewards']).float()

        # Simplified PPO implementation
        # In practice, you'd need advantages, old log probs, etc.

        # Value loss
        values = self.critic(observations)
        value_loss = F.mse_loss(values.squeeze(), rewards)

        # Policy loss (simplified)
        mean, log_std = self.actor(observations).chunk(2, dim=-1)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        log_probs = normal.log_prob(actions).sum(dim=-1)

        # Dummy advantage (in practice, compute from rewards)
        advantages = rewards - values.squeeze().detach()
        policy_loss = -(log_probs * advantages).mean()

        total_loss = value_loss + policy_loss

        return total_loss
