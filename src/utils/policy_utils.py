"""Shared policy networks for Worker and Learner."""

from typing import Dict, Literal

import numpy as np
import torch
import torch.nn as nn

AlgorithmName = Literal['TD3', 'DDPG', 'PPO', 'FSAC', 'SAC']


def build_deterministic_actor(state_dim: int, action_dim: int, hidden_dim: int = 256) -> nn.Sequential:
    """Must match TD3Policy / DDPGPolicy actor in policies.py."""
    return nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim),
        nn.Tanh(),
    )


def build_ppo_actor(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    discrete: bool = False,
) -> nn.Sequential:
    actor_out = action_dim if discrete else action_dim * 2
    return nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, actor_out),
    )


def build_discrete_sac_actor(state_dim: int, action_dim: int, hidden_dim: int = 256) -> nn.Sequential:
    """Must match FSACPolicy actor in policies.py."""
    return nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, action_dim),
    )


def build_continuous_sac_actor(state_dim: int, action_dim: int, hidden_dim: int = 256) -> nn.Module:
    from src.utils.policies import GaussianActor
    return GaussianActor(state_dim, action_dim, hidden_dim)


def _critic_template(state_dim: int, action_dim: int, hidden_dim: int, prefix: str) -> Dict[str, np.ndarray]:
    critic_in = state_dim + action_dim
    layers = nn.Sequential(
        nn.Linear(critic_in, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    template = {}
    for name, param in layers.state_dict().items():
        template[f'{prefix}.{name}'] = param.cpu().numpy().astype(np.float32)
    return template


def _value_template(state_dim: int, hidden_dim: int, prefix: str = 'critic') -> Dict[str, np.ndarray]:
    layers = nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )
    return {
        f'{prefix}.{name}': param.cpu().numpy().astype(np.float32)
        for name, param in layers.state_dict().items()
    }


def build_model_template(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    algorithm: str = 'DDPG',
    discrete: bool = False,
) -> Dict[str, np.ndarray]:
    """Weight template aligned with RL genotype keys used in the EA population."""
    algo = algorithm.upper()
    template: Dict[str, np.ndarray] = {}

    if algo == 'FSAC':
        actor = build_discrete_sac_actor(state_dim, action_dim, hidden_dim)
        for name, param in actor.state_dict().items():
            template[f'actor.{name}'] = param.cpu().numpy().astype(np.float32)
    elif algo == 'SAC':
        actor = build_continuous_sac_actor(state_dim, action_dim, hidden_dim)
        for name, param in actor.state_dict().items():
            template[f'actor.{name}'] = param.cpu().numpy().astype(np.float32)
    elif algo == 'PPO':
        actor = build_ppo_actor(state_dim, action_dim, hidden_dim, discrete=discrete)
        for name, param in actor.state_dict().items():
            template[f'actor.{name}'] = param.cpu().numpy().astype(np.float32)
        template.update(_value_template(state_dim, hidden_dim, 'critic'))
    else:
        actor = build_deterministic_actor(state_dim, action_dim, hidden_dim)
        for name, param in actor.state_dict().items():
            template[f'actor.{name}'] = param.cpu().numpy().astype(np.float32)

    if algo == 'TD3':
        template.update(_critic_template(state_dim, action_dim, hidden_dim, 'critic1'))
        template.update(_critic_template(state_dim, action_dim, hidden_dim, 'critic2'))
    elif algo == 'DDPG':
        template.update(_critic_template(state_dim, action_dim, hidden_dim, 'critic'))
    elif algo in ('PPO', 'FSAC', 'SAC'):
        pass
    else:
        raise ValueError(f'build_model_template does not support algorithm={algorithm}')

    return template


def load_network_from_weights(
    network: nn.Sequential,
    weights: Dict[str, np.ndarray],
    prefix: str = 'actor.',
) -> None:
    state = network.state_dict()
    for key, tensor in state.items():
        full_key = f'{prefix}{key}'
        if full_key not in weights:
            raise KeyError(f'Missing weight key: {full_key}')
        state[key] = torch.from_numpy(weights[full_key].copy()).float()
    network.load_state_dict(state)


def load_actor_from_weights(
    actor: nn.Sequential,
    weights: Dict[str, np.ndarray],
    prefix: str = 'actor.',
) -> None:
    load_network_from_weights(actor, weights, prefix)


def actor_deterministic_action(
    actor: nn.Sequential,
    observation: np.ndarray,
    algorithm: str = 'DDPG',
    discrete: bool = False,
) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[np.newaxis, :]
    with torch.no_grad():
        output = actor(torch.from_numpy(obs).float())
    if algorithm.upper() == 'SAC':
        with torch.no_grad():
            return actor.deterministic(torch.from_numpy(obs).float()).squeeze(0).cpu().numpy()
    if algorithm.upper() in ('PPO', 'FSAC'):
        if discrete:
            return int(torch.argmax(output, dim=-1).cpu().numpy()[0])
        if algorithm.upper() == 'FSAC':
            return int(torch.argmax(output, dim=-1).cpu().numpy()[0])
        mean, _ = output.chunk(2, dim=-1)
        return mean.squeeze(0).cpu().numpy()
    return output.squeeze(0).cpu().numpy()


def clip_action(action: np.ndarray, action_space) -> np.ndarray:
    if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
        return np.clip(action, action_space.low, action_space.high)
    if hasattr(action_space, 'n'):
        return int(np.argmax(action)) if isinstance(action, np.ndarray) else int(action)
    return action


def encode_action_for_buffer(action, action_space, action_dim: int, algorithm: str = 'DDPG') -> np.ndarray:
    """Use scalar discrete actions for FSAC/PPO and one-hot actions for deterministic critics."""
    if hasattr(action_space, 'n'):
        if algorithm.upper() in ('PPO', 'FSAC'):
            return np.asarray(int(action), dtype=np.int64)
        encoded = np.zeros(int(action_dim), dtype=np.float32)
        encoded[int(action)] = 1.0
        return encoded
    return np.asarray(action, dtype=np.float32)


class ActorEvaluator:
    """Lightweight policy evaluator for Ray Workers."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 algorithm: str = 'DDPG', discrete: bool = False):
        self.algorithm = algorithm.upper()
        self.discrete = bool(discrete)
        if self.algorithm in ('TD3', 'DDPG'):
            self.actor = build_deterministic_actor(state_dim, action_dim, hidden_dim)
        elif self.algorithm == 'PPO':
            self.actor = build_ppo_actor(state_dim, action_dim, hidden_dim, discrete=self.discrete)
        elif self.algorithm == 'FSAC':
            self.actor = build_discrete_sac_actor(state_dim, action_dim, hidden_dim)
        elif self.algorithm == 'SAC':
            self.actor = build_continuous_sac_actor(state_dim, action_dim, hidden_dim)
        else:
            raise ValueError(f'ActorEvaluator does not support algorithm={algorithm}')
        self.actor.eval()

    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        load_actor_from_weights(self.actor, weights)

    def get_action(self, observation: np.ndarray, action_space=None) -> np.ndarray:
        action = actor_deterministic_action(
            self.actor, observation, algorithm=self.algorithm, discrete=self.discrete)
        if action_space is not None:
            action = clip_action(action, action_space)
        return action
