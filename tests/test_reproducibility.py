"""Reproducibility: Worker and Learner produce aligned rollouts for same (weights, seed)."""

import numpy as np
import pytest

from src.utils.environment import get_env_info, make_env
from src.utils.policy_utils import ActorEvaluator, build_model_template


@pytest.fixture
def lunar_setup():
    env_name = 'LunarLanderContinuous-v3'
    info = get_env_info(env_name)
    template = build_model_template(info['state_dim'], info['action_dim'], algorithm='DDPG')
    weights = {k: v.copy() for k, v in template.items()}
    np.random.seed(0)
    for k in weights:
        if k.startswith('actor.'):
            weights[k] = np.random.randn(*weights[k].shape).astype(np.float32) * 0.1
    return env_name, info, weights


def _rollout(env, evaluator, weights, seed, max_steps=200):
    evaluator.load_weights(weights)
    obs, _ = env.reset(seed=seed)
    obs_list, act_list, rew_list = [], [], []
    done = truncated = False
    steps = 0
    while not (done or truncated) and steps < max_steps:
        action = evaluator.get_action(obs, env.action_space)
        obs_list.append(np.array(obs, dtype=np.float32))
        act_list.append(np.array(action, dtype=np.float32))
        obs, reward, done, truncated, _ = env.step(action)
        rew_list.append(float(reward))
        steps += 1
    return obs_list, act_list, rew_list


def test_worker_learner_trajectory_alignment(lunar_setup):
    env_name, info, weights = lunar_setup
    seed = 12345
    max_steps = 150

    env1 = make_env(env_name, max_episode_steps=max_steps)
    env2 = make_env(env_name, max_episode_steps=max_steps)
    ev1 = ActorEvaluator(info['state_dim'], info['action_dim'], algorithm='DDPG')
    ev2 = ActorEvaluator(info['state_dim'], info['action_dim'], algorithm='DDPG')

    o1, a1, r1 = _rollout(env1, ev1, weights, seed, max_steps)
    o2, a2, r2 = _rollout(env2, ev2, weights, seed, max_steps)

    env1.close()
    env2.close()

    assert len(o1) == len(o2) and len(o1) > 5
    obs_err = np.mean([np.linalg.norm(x - y) for x, y in zip(o1, o2)])
    act_err = np.mean([np.linalg.norm(x - y) for x, y in zip(a1, a2)])
    rew_err = np.mean([abs(x - y) for x, y in zip(r1, r2)])

    assert obs_err < 1e-5, f'obs mismatch {obs_err}'
    assert act_err < 1e-5, f'action mismatch {act_err}'
    assert rew_err < 1e-5, f'reward mismatch {rew_err}'


def test_actor_weights_change_fitness(lunar_setup):
    """Fitness must depend on weights, not random actions."""
    env_name, info, weights = lunar_setup
    seed = 7
    max_steps = 100
    env = make_env(env_name, max_episode_steps=max_steps)
    ev = ActorEvaluator(info['state_dim'], info['action_dim'])

    _, _, r1 = _rollout(env, ev, weights, seed, max_steps)
    w2 = {k: v.copy() for k, v in weights.items()}
    w2['actor.4.weight'] = -w2['actor.4.weight']
    _, _, r2 = _rollout(env, ev, w2, seed, max_steps)
    env.close()
    assert r1 != r2 or abs(sum(r1) - sum(r2)) > 1e-3
