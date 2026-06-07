"""Shared actor network for Worker and Learner (TD3 / DDPG compatible)."""

from typing import Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn

AlgorithmName = Literal['TD3', 'DDPG', 'PPO']


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


def build_model_template(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    algorithm: str = 'DDPG',
) -> Dict[str, np.ndarray]:
    """Weight template aligned with RL genotype keys used in the EA population."""
    algo = algorithm.upper()
    template: Dict[str, np.ndarray] = {}

    actor = build_deterministic_actor(state_dim, action_dim, hidden_dim)
    for name, param in actor.state_dict().items():
        template[f'actor.{name}'] = param.cpu().numpy().astype(np.float32)

    if algo == 'TD3':
        template.update(_critic_template(state_dim, action_dim, hidden_dim, 'critic1'))
        template.update(_critic_template(state_dim, action_dim, hidden_dim, 'critic2'))
    elif algo == 'DDPG':
        template.update(_critic_template(state_dim, action_dim, hidden_dim, 'critic'))
    elif algo == 'PPO':
        pass
    else:
        raise ValueError(f'build_model_template does not support algorithm={algorithm}')

    return template


def load_actor_from_weights(
    actor: nn.Sequential,
    weights: Dict[str, np.ndarray],
    prefix: str = 'actor.',
) -> None:
    state = actor.state_dict()
    for key, tensor in state.items():
        full_key = f'{prefix}{key}'
        if full_key not in weights:
            raise KeyError(f'Missing weight key: {full_key}')
        state[key] = torch.from_numpy(weights[full_key].copy()).float()
    actor.load_state_dict(state)


def actor_deterministic_action(actor: nn.Sequential, observation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[np.newaxis, :]
    with torch.no_grad():
        action = actor(torch.from_numpy(obs).float())
    return action.squeeze(0).cpu().numpy()


def clip_action(action: np.ndarray, action_space) -> np.ndarray:
    if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
        return np.clip(action, action_space.low, action_space.high)
    if hasattr(action_space, 'n'):
        return int(np.argmax(action)) if isinstance(action, np.ndarray) else int(action)
    return action


def encode_action_for_buffer(action, action_space, action_dim: int) -> np.ndarray:
    """Store discrete actions as one-hot vectors so deterministic critics keep a fixed input shape."""
    if hasattr(action_space, 'n'):
        encoded = np.zeros(int(action_dim), dtype=np.float32)
        encoded[int(action)] = 1.0
        return encoded
    return np.asarray(action, dtype=np.float32)


class ActorEvaluator:
    """Lightweight actor for Ray Workers (TD3 / DDPG deterministic policy)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 algorithm: str = 'DDPG'):
        self.algorithm = algorithm.upper()
        if self.algorithm in ('TD3', 'DDPG'):
            self.actor = build_deterministic_actor(state_dim, action_dim, hidden_dim)
        else:
            raise ValueError(f'ActorEvaluator does not support algorithm={algorithm}')
        self.actor.eval()

    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        load_actor_from_weights(self.actor, weights)

    def get_action(self, observation: np.ndarray, action_space=None) -> np.ndarray:
        action = actor_deterministic_action(self.actor, observation)
        if action_space is not None:
            action = clip_action(action, action_space)
        return action
