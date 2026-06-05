"""Environment utilities for Dist-ERL."""

import os
import warnings
from typing import Any, Dict

import gymnasium as gym

# Literature task IDs (*-v2) → runnable Gymnasium MuJoCo (v5 backend, paper labels unchanged).
MUJOCO_V2_RUNTIME_MAP: Dict[str, str] = {
    'HalfCheetah-v2': 'HalfCheetah-v5',
    'Swimmer-v2': 'Swimmer-v5',
    'Hopper-v2': 'Hopper-v5',
    'Ant-v2': 'Ant-v5',
    'Walker2d-v2': 'Walker2d-v5',
    'Humanoid-v2': 'Humanoid-v5',
}


def apply_headless_mujoco_runtime() -> None:
    """Headless MuJoCo for Ray workers / tmux (no OpenGL window)."""
    gl = os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', gl if gl in ('egl', 'osmesa') else 'egl')


apply_headless_mujoco_runtime()
warnings.filterwarnings('ignore', category=DeprecationWarning, module='gymnasium')


def resolve_gym_env_id(env_name: str) -> str:
    """Map paper-style MuJoCo-v2 IDs to runnable Gymnasium env IDs."""
    return MUJOCO_V2_RUNTIME_MAP.get(env_name, env_name)


def make_env(env_name: str, max_episode_steps: int = 1000, **kwargs) -> gym.Env:
    """Create environment with specified parameters (no rendering)."""
    gym_id = resolve_gym_env_id(env_name)
    kwargs.setdefault('render_mode', None)
    try:
        env = gym.make(gym_id, max_episode_steps=max_episode_steps, **kwargs)
    except TypeError:
        kwargs.pop('render_mode', None)
        env = gym.make(gym_id, **kwargs)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def get_env_info(env_name: str) -> dict:
    """Get environment information (uses resolved Gymnasium ID)."""
    gym_id = resolve_gym_env_id(env_name)
    env = make_env(env_name)
    info = {
        'env_name': env_name,
        'gym_id': gym_id,
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'state_dim': env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else None,
        'action_dim': (
            env.action_space.shape[0]
            if hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0
            else getattr(env.action_space, 'n', None)
        ),
        'max_episode_steps': getattr(env, '_max_episode_steps', 1000),
    }
    env.close()
    return info


class TrajectoryCollector:
    """Utility for collecting trajectories from environments"""

    def __init__(self, env_name: str, max_episode_steps: int = 1000):
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.env = make_env(env_name, max_episode_steps)

    def collect_trajectory(self, policy, seed: int = None) -> dict:
        obs, _ = self.env.reset(seed=seed)
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'seed': seed,
        }
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated) and step_count < self.max_episode_steps:
            action = policy.get_action(obs)
            trajectory['observations'].append(obs)
            obs, reward, done, truncated, _ = self.env.step(action)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done or truncated)
            step_count += 1

        return trajectory

    def close(self):
        self.env.close()
