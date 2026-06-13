"""RL Policy implementations for FedEvoRL."""

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


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor for continuous SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5.0, 2.0)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)


class SACPolicy(BasePolicy):
    """Continuous Soft Actor-Critic with actor-only export for federation/EA."""

    def __init__(self, state_dim: int = 111, action_dim: int = 8, hidden_dim: int = 256,
                 init_alpha: float = 0.2, target_entropy: float = None):
        super().__init__(state_dim, action_dim)
        self.actor = GaussianActor(state_dim, action_dim, hidden_dim)
        self.critic1 = _build_q_critic(state_dim, action_dim, hidden_dim)
        self.critic2 = _build_q_critic(state_dim, action_dim, hidden_dim)
        self.target_critic1 = _build_q_critic(state_dim, action_dim, hidden_dim)
        self.target_critic2 = _build_q_critic(state_dim, action_dim, hidden_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(init_alpha)), dtype=torch.float32))
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(1e-4, 10.0)

    def reset_actor_last_layers(self) -> None:
        linear_layers = [m for m in self.actor.modules() if isinstance(m, nn.Linear)]
        for layer in linear_layers[-3:]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def sync_target(self, tau: float = 1.0) -> None:
        for target, source in ((self.target_critic1, self.critic1), (self.target_critic2, self.critic2)):
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis, :]
        obs_t = torch.from_numpy(obs).float()
        with torch.no_grad():
            action = self.actor.deterministic(obs_t) if deterministic else self.actor.sample(obs_t)[0]
        action_np = action.cpu().numpy()
        return action_np[0] if single else action_np

    def update(self, batch: Dict[str, np.ndarray], gamma: float, tau: float) -> torch.Tensor:
        """Legacy combined loss. Prefer optimize_step() for correct SAC training."""
        critic_loss, actor_loss, alpha_loss = self.compute_losses(batch, gamma)
        return critic_loss + actor_loss + alpha_loss

    def compute_losses(self, batch: Dict[str, np.ndarray], gamma: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = torch.from_numpy(batch['observations']).float()
        actions = torch.from_numpy(batch['actions']).float()
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        rewards = torch.from_numpy(batch['rewards']).float().unsqueeze(-1)
        next_observations = torch.from_numpy(batch['next_observations']).float()
        dones = torch.from_numpy(batch['dones']).float().unsqueeze(-1)

        with torch.no_grad():
            next_actions, next_logp = self.actor.sample(next_observations)
            next_sa = torch.cat([next_observations, next_actions], dim=-1)
            next_q = torch.min(self.target_critic1(next_sa), self.target_critic2(next_sa))
            target_q = rewards + gamma * (1.0 - dones) * (next_q - self.alpha.detach() * next_logp)

        sa = torch.cat([observations, actions], dim=-1)
        critic_loss = F.smooth_l1_loss(self.critic1(sa), target_q) + F.smooth_l1_loss(self.critic2(sa), target_q)

        new_actions, logp = self.actor.sample(observations)
        new_sa = torch.cat([observations, new_actions], dim=-1)
        min_q = torch.min(self.critic1(new_sa), self.critic2(new_sa))
        actor_loss = (self.alpha.detach() * logp - min_q).mean()
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        return critic_loss, actor_loss, alpha_loss

    def optimize_step(
        self,
        batch: Dict[str, np.ndarray],
        critic_optimizer: torch.optim.Optimizer,
        actor_optimizer: torch.optim.Optimizer,
        alpha_optimizer: torch.optim.Optimizer,
        gamma: float,
        tau: float,
        grad_clip: float = 10.0,
    ) -> torch.Tensor:
        """Standard SAC update: critic, actor, and temperature are optimized separately."""
        observations = torch.from_numpy(batch['observations']).float()
        actions = torch.from_numpy(batch['actions']).float()
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        rewards = torch.from_numpy(batch['rewards']).float().unsqueeze(-1)
        next_observations = torch.from_numpy(batch['next_observations']).float()
        dones = torch.from_numpy(batch['dones']).float().unsqueeze(-1)

        with torch.no_grad():
            next_actions, next_logp = self.actor.sample(next_observations)
            next_sa = torch.cat([next_observations, next_actions], dim=-1)
            next_q = torch.min(self.target_critic1(next_sa), self.target_critic2(next_sa))
            target_q = rewards + gamma * (1.0 - dones) * (next_q - self.alpha.detach() * next_logp)

        sa = torch.cat([observations, actions], dim=-1)
        critic_loss = (
            F.smooth_l1_loss(self.critic1(sa), target_q)
            + F.smooth_l1_loss(self.critic2(sa), target_q)
        )
        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            max_norm=grad_clip,
        )
        critic_optimizer.step()

        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        for p in critic_params:
            p.requires_grad_(False)
        new_actions, logp = self.actor.sample(observations)
        new_sa = torch.cat([observations, new_actions], dim=-1)
        min_q = torch.min(self.critic1(new_sa), self.critic2(new_sa))
        actor_loss = (self.alpha.detach() * logp - min_q).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=grad_clip)
        actor_optimizer.step()
        for p in critic_params:
            p.requires_grad_(True)

        _, logp_alpha = self.actor.sample(observations)
        alpha_loss = -(self.log_alpha * (logp_alpha + self.target_entropy).detach()).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        self.sync_target(tau)
        return (critic_loss.detach() + actor_loss.detach() + alpha_loss.detach())


def _build_discrete_q_critic(state_dim: int, action_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim),
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


class FSACPolicy(BasePolicy):
    """Discrete Soft Actor-Critic for federated worker sharing."""

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 256,
                 init_alpha: float = 0.2, target_entropy: float = None):
        super().__init__(state_dim, action_dim)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic1 = _build_discrete_q_critic(state_dim, action_dim, hidden_dim)
        self.critic2 = _build_discrete_q_critic(state_dim, action_dim, hidden_dim)
        self.target_critic1 = _build_discrete_q_critic(state_dim, action_dim, hidden_dim)
        self.target_critic2 = _build_discrete_q_critic(state_dim, action_dim, hidden_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(init_alpha)), dtype=torch.float32))
        self.target_entropy = float(target_entropy) if target_entropy is not None else 0.98 * np.log(action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(1e-4, 10.0)

    def reset_actor_last_layers(self) -> None:
        linear_layers = [m for m in self.actor.modules() if isinstance(m, nn.Linear)]
        for layer in linear_layers[-2:]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def sync_target(self, tau: float = 1.0) -> None:
        for target, source in ((self.target_critic1, self.critic1), (self.target_critic2, self.critic2)):
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> int:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        with torch.no_grad():
            logits = self.actor(torch.from_numpy(obs).float())
            if deterministic:
                return int(torch.argmax(logits, dim=-1).cpu().numpy()[0])
            dist = torch.distributions.Categorical(logits=logits)
            return int(dist.sample().cpu().numpy()[0])

    def update(self, batch: Dict[str, np.ndarray], gamma: float, tau: float) -> torch.Tensor:
        observations = torch.from_numpy(batch['observations']).float()
        actions = torch.from_numpy(batch['actions'])
        if actions.ndim > 1:
            actions = torch.argmax(actions.float(), dim=-1)
        actions = actions.long().view(-1, 1)
        rewards = torch.from_numpy(batch['rewards']).float()
        next_observations = torch.from_numpy(batch['next_observations']).float()
        dones = torch.from_numpy(batch['dones']).float()

        q1 = self.critic1(observations).gather(1, actions).squeeze(-1)
        q2 = self.critic2(observations).gather(1, actions).squeeze(-1)

        with torch.no_grad():
            next_logits = self.actor(next_observations)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_probs = next_log_probs.exp()
            next_q = torch.min(self.target_critic1(next_observations), self.target_critic2(next_observations))
            next_v = (next_probs * (next_q - self.alpha.detach() * next_log_probs)).sum(dim=-1)
            target_q = rewards + gamma * (1.0 - dones) * next_v

        critic_loss = F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)

        logits = self.actor(observations)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        min_q = torch.min(self.critic1(observations), self.critic2(observations)).detach()
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(dim=-1).mean()
        entropy = -(probs.detach() * log_probs).sum(dim=-1).mean()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy).detach()

        return critic_loss + actor_loss + alpha_loss


class PPOPolicy(BasePolicy):
    """Lightweight PPO-style actor-critic for continuous and discrete actions."""

    def __init__(self, state_dim: int = 111, action_dim: int = 8, hidden_dim: int = 256,
                 discrete: bool = False):
        super().__init__(state_dim, action_dim)
        self.discrete = bool(discrete)

        actor_out = action_dim if self.discrete else action_dim * 2
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, actor_out),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def reset_actor_last_layers(self) -> None:
        linear_layers = [m for m in self.actor.modules() if isinstance(m, nn.Linear)]
        for layer in linear_layers[-2:]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample or greedily choose an action from the policy."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
            if self.discrete:
                logits = self.actor(obs_tensor)
                if deterministic:
                    return int(torch.argmax(logits, dim=-1).item())
                dist = torch.distributions.Categorical(logits=logits)
                return int(dist.sample().item())
            mean, log_std = self.actor(obs_tensor).chunk(2, dim=-1)
            if deterministic:
                return mean.squeeze(0).numpy()
            std = torch.clamp(log_std, -5.0, 2.0).exp()
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            return action.squeeze(0).numpy()

    def update(self, batch: Dict[str, np.ndarray], gamma: float, tau: float) -> torch.Tensor:
        """PPO update (simplified)"""
        observations = torch.from_numpy(batch['observations']).float()
        raw_actions = torch.from_numpy(batch['actions'])
        rewards = torch.from_numpy(batch['rewards']).float()

        values = self.critic(observations)
        value_loss = F.mse_loss(values.squeeze(), rewards)
        advantages = rewards - values.squeeze().detach()

        if self.discrete:
            actions = raw_actions
            if actions.ndim > 1:
                actions = torch.argmax(actions.float(), dim=-1)
            actions = actions.long()
            logits = self.actor(observations)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
        else:
            actions = raw_actions.float()
            mean, log_std = self.actor(observations).chunk(2, dim=-1)
            std = torch.clamp(log_std, -5.0, 2.0).exp()
            normal = torch.distributions.Normal(mean, std)
            log_probs = normal.log_prob(actions).sum(dim=-1)
            entropy = normal.entropy().sum(dim=-1).mean()

        policy_loss = -(log_probs * advantages).mean()
        total_loss = value_loss + policy_loss - 0.01 * entropy

        return total_loss
