"""Environment utilities for FedEvoRL."""

import os
import warnings
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

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


def _client_phase(client_id: int) -> float:
    """Symmetric client phases; first four clients cover both easier/harder sides."""
    phases = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0, -2.0 / 3.0, 0.0, 2.0 / 3.0)
    return phases[int(client_id) % len(phases)]


def resolve_gym_env_id(env_name: str) -> str:
    """Map paper-style MuJoCo-v2 IDs to runnable Gymnasium env IDs."""
    return MUJOCO_V2_RUNTIME_MAP.get(env_name, env_name)


class HeterogeneousClientEnv(gym.Wrapper):
    """Client-local MDP perturbations for federated RL experiments."""

    def __init__(
        self,
        env: gym.Env,
        client_id: int,
        heterogeneity: float,
        mode: str = 'reward_action_noise',
    ):
        super().__init__(env)
        self.client_id = int(client_id)
        self.heterogeneity = max(0.0, float(heterogeneity))
        self.mode = mode
        phase = _client_phase(self.client_id)
        noisy_mode = mode in ('reward_action_noise', 'mixed', 'env_params')
        self.reward_scale = 1.0 + (0.15 * self.heterogeneity * phase if noisy_mode else 0.0)
        self.reward_bias = 0.02 * self.heterogeneity * phase if noisy_mode else 0.0
        noise_boost = 2.5 if mode == 'mixed' else 1.0
        if self.env.unwrapped.__class__.__name__.lower().startswith('lunar'):
            noise_boost *= 1.25
        if self.env.unwrapped.__class__.__name__.lower().startswith('acrobot'):
            noise_boost *= 1.15
        self.action_noise = (
            0.035 * noise_boost * self.heterogeneity * (1 + (self.client_id % 3))
            if noisy_mode else 0.0
        )
        self.observation_noise = (
            0.012 * noise_boost * self.heterogeneity * (1 + (self.client_id % 2))
            if noisy_mode else 0.0
        )
        self.seed_offset = self.client_id * 9973
        self._rng = np.random.default_rng(self.seed_offset)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        local_seed = None if seed is None else int(seed + self.seed_offset)
        obs, info = self.env.reset(seed=local_seed, options=options)
        return self._perturb_observation(obs), info

    def step(self, action):
        if self.action_noise > 0 and hasattr(self.action_space, 'low'):
            noise = self._rng.normal(0.0, self.action_noise, np.shape(action))
            action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        elif self.action_noise > 0 and hasattr(self.action_space, 'n'):
            if self._rng.random() < self.action_noise:
                action = int(self._rng.integers(self.action_space.n))
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward) * self.reward_scale + self.reward_bias
        return self._perturb_observation(obs), reward, terminated, truncated, info

    def _perturb_observation(self, obs):
        if self.observation_noise <= 0:
            return obs
        arr = np.asarray(obs, dtype=np.float32)
        noise = self._rng.normal(0.0, self.observation_noise, arr.shape).astype(np.float32)
        return arr + noise


def _client_env_kwargs(env_name: str, client_id: int, heterogeneity: float, mode: str) -> Dict[str, Any]:
    """Best-effort Gymnasium kwargs for literature-style client heterogeneity."""
    if heterogeneity <= 0 or mode in ('none', 'reward_action_noise'):
        return {}
    phase = _client_phase(client_id)
    strength = float(heterogeneity) * phase

    if env_name == 'Pendulum-v1':
        return {'g': max(4.0, 10.0 * (1.0 + 0.25 * strength))}
    if env_name == 'LunarLander-v3':
        scale_boost = 1.8 if mode == 'mixed' else 1.0
        return {
            'gravity': float(np.clip(-10.0 * (1.0 + 0.18 * scale_boost * strength), -11.8, -6.0)),
            'enable_wind': True,
            'wind_power': float(np.clip(10.0 + 12.0 * scale_boost * strength, 0.0, 20.0)),
            'turbulence_power': float(np.clip(1.2 + 1.0 * scale_boost * strength, 0.0, 2.0)),
        }
    return {}


def _apply_classic_control_heterogeneity(
    env: gym.Env,
    env_name: str,
    client_id: int,
    heterogeneity: float,
    mode: str,
) -> None:
    if heterogeneity <= 0 or mode in ('none', 'reward_action_noise'):
        return
    phase = _client_phase(client_id)
    strength = float(heterogeneity) * phase
    base = env.unwrapped
    if env_name == 'Acrobot-v1':
        scale_boost = 2.6 if mode == 'mixed' else 1.0
        for attr, base_value, scale in (
            ('LINK_LENGTH_1', 1.0, 0.42),
            ('LINK_LENGTH_2', 1.0, -0.38),
            ('LINK_MASS_1', 1.0, 0.45),
            ('LINK_MASS_2', 1.0, -0.45),
            ('LINK_COM_POS_1', 0.5, 0.32),
            ('LINK_COM_POS_2', 0.5, -0.32),
        ):
            if hasattr(base, attr):
                setattr(base, attr, base_value * max(0.18, 1.0 + scale * scale_boost * strength))
        if hasattr(base, 'g'):
            base.g = 9.8 * max(0.35, 1.0 + 0.45 * scale_boost * strength)
        if hasattr(base, 'AVAIL_TORQUE'):
            torque_scale = max(0.22, 1.0 - 0.42 * scale_boost * strength)
            base.AVAIL_TORQUE = [float(t * torque_scale) for t in (-1.0, 0.0, 1.0)]
        if hasattr(base, 'MAX_VEL_1'):
            base.MAX_VEL_1 = 4 * np.pi * max(0.35, 1.0 + 0.30 * scale_boost * strength)
        if hasattr(base, 'MAX_VEL_2'):
            base.MAX_VEL_2 = 9 * np.pi * max(0.35, 1.0 - 0.30 * scale_boost * strength)
        if hasattr(base, 'dt'):
            base.dt = 0.2 * max(0.35, 1.0 + 0.25 * scale_boost * strength)
        return
    if env_name == 'MountainCar-v0':
        scale_boost = 3.0 if mode == 'mixed' else 1.0
        if hasattr(base, 'force'):
            base.force = 0.001 * max(0.25, 1.0 - 0.45 * scale_boost * strength)
        if hasattr(base, 'gravity'):
            base.gravity = 0.0025 * max(0.35, 1.0 + 0.40 * scale_boost * strength)
        if hasattr(base, 'goal_position'):
            base.goal_position = float(np.clip(0.5 + 0.08 * scale_boost * strength, 0.38, 0.62))
        if hasattr(base, 'min_position'):
            base.min_position = float(np.clip(-1.2 - 0.08 * scale_boost * strength, -1.35, -1.05))
        if hasattr(base, 'max_speed'):
            base.max_speed = 0.07 * max(0.45, 1.0 - 0.25 * scale_boost * strength)
        return
    if env_name != 'CartPole-v1':
        return
    scale_boost = 3.0 if mode == 'mixed' else 1.0
    if hasattr(base, 'gravity'):
        base.gravity = 9.8 * max(0.35, 1.0 + 0.30 * scale_boost * strength)
    if hasattr(base, 'masscart'):
        base.masscart = 1.0 * max(0.25, 1.0 + 0.45 * scale_boost * strength)
    if hasattr(base, 'masspole'):
        base.masspole = 0.1 * max(0.25, 1.0 - 0.45 * scale_boost * strength)
    if hasattr(base, 'length'):
        base.length = 0.5 * max(0.20, 1.0 + 0.60 * scale_boost * strength)
    if hasattr(base, 'force_mag'):
        base.force_mag = 10.0 * max(0.25, 1.0 - 0.45 * scale_boost * strength)
    if hasattr(base, 'tau'):
        base.tau = 0.02 * max(0.35, 1.0 + 0.25 * scale_boost * strength)
    if hasattr(base, 'masscart') and hasattr(base, 'masspole'):
        base.total_mass = base.masscart + base.masspole
    if hasattr(base, 'masspole') and hasattr(base, 'length'):
        base.polemass_length = base.masspole * base.length


def make_env(
    env_name: str,
    max_episode_steps: int = 1000,
    client_id: Optional[int] = None,
    heterogeneity: float = 0.0,
    heterogeneity_mode: str = 'reward_action_noise',
    **kwargs,
) -> gym.Env:
    """Create environment with specified parameters (no rendering)."""
    gym_id = resolve_gym_env_id(env_name)
    kwargs.setdefault('render_mode', None)
    if client_id is not None:
        kwargs.update(_client_env_kwargs(env_name, client_id, heterogeneity, heterogeneity_mode))
    try:
        env = gym.make(gym_id, max_episode_steps=max_episode_steps, **kwargs)
    except TypeError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop('render_mode', None)
        for key in ('g', 'gravity', 'enable_wind', 'wind_power', 'turbulence_power'):
            fallback_kwargs.pop(key, None)
        env = gym.make(gym_id, **fallback_kwargs)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if client_id is not None:
        _apply_classic_control_heterogeneity(
            env, env_name, client_id, heterogeneity, heterogeneity_mode)
    if client_id is not None and heterogeneity > 0 and heterogeneity_mode in (
        'reward_action_noise', 'mixed', 'env_params'
    ):
        env = HeterogeneousClientEnv(env, client_id, heterogeneity, heterogeneity_mode)
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
