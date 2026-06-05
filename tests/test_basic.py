"""Basic tests for Dist-ERL components."""

import pytest
import numpy as np
import ray
from src.manager import EAManager
from src.worker import RolloutWorker
from src.learner import RLLearner
from src.utils.replay_buffer import HybridReplayBuffer
from src.utils.environment import get_env_info
from src.utils.policy_utils import build_model_template
from src.utils.individual import Individual


def test_hybrid_replay_buffer():
    buffer = HybridReplayBuffer(capacity=100)
    obs = np.random.randn(10)
    action = np.random.randn(2)
    buffer.add_rl_data(obs, action, 1.0, obs, False)
    assert buffer.rl_size == 1
    buffer.add_ea_seed(seed=42, fitness=10.0, generation=1, individual_id=1)
    buffer.add_reproduced_ea_transition(obs, action, 1.0, obs, False, seed=42, generation=1)
    batch = buffer.sample(2, ea_batch_ratio=0.5)
    assert len(batch['observations']) == 2


def test_ea_manager():
    ray.init(ignore_reinit_error=True, num_cpus=1)
    info = get_env_info('Pendulum-v1')
    template = build_model_template(info['state_dim'], info['action_dim'], algorithm='DDPG')
    manager = EAManager.remote(population_size=10, elite_fraction=0.2)
    ray.get(manager.initialize_population.remote(template))
    population = ray.get(manager.get_population_for_evaluation.remote())
    assert len(population) == 10
    results = [{'id': i, 'fitness': float(i * 10)} for i in range(10)]
    ray.get(manager.update_fitness.remote(results))
    stats = ray.get(manager.get_stats.remote())
    assert stats['max_fitness'] == 90.0
    ray.shutdown()


def test_manager_evaluate_population():
    ray.init(ignore_reinit_error=True, num_cpus=2)
    info = get_env_info('Pendulum-v1')
    template = build_model_template(info['state_dim'], info['action_dim'], algorithm='DDPG')
    manager = EAManager.remote(population_size=6, elite_fraction=0.5)
    ray.get(manager.initialize_population.remote(template))
    workers = [RolloutWorker.remote('Pendulum-v1', 50, 'DDPG') for _ in range(2)]
    results = ray.get(manager.evaluate_population.remote(workers))
    assert len(results) == 6
    assert all(np.isfinite(res['fitness']) for res in results)
    ray.shutdown()


def test_federated_soft_migration_updates_multiple_non_elites():
    ray.init(ignore_reinit_error=True, num_cpus=1)
    info = get_env_info('Pendulum-v1')
    template = build_model_template(info['state_dim'], info['action_dim'], algorithm='DDPG')
    manager = EAManager.remote(population_size=6, elite_fraction=0.2, num_elitists=1)
    ray.get(manager.initialize_population.remote(template))
    ray.get(manager.update_fitness.remote([
        {'id': i, 'fitness': float(i)} for i in range(6)
    ]))

    rl_weights = {key: np.ones_like(value) for key, value in template.items()}
    inserted = ray.get(manager.inject_rl_individual.remote(
        rl_weights, 0.0, 2, 1.0))
    population = ray.get(manager.get_population_for_evaluation.remote())

    actor_key = next(key for key in rl_weights if key.startswith('actor.'))
    migrated = sum(
        np.allclose(ind['weights'][actor_key], rl_weights[actor_key])
        for ind in population[1:]
    )
    assert inserted == 2
    assert migrated == 2
    ray.shutdown()


def test_learner_warm_start_actor_from_elite():
    ray.init(ignore_reinit_error=True, num_cpus=1)
    info = get_env_info('Pendulum-v1')
    template = build_model_template(info['state_dim'], info['action_dim'], algorithm='DDPG')
    elite_weights = {key: np.ones_like(value) for key, value in template.items()}
    learner = RLLearner.remote(env_name='Pendulum-v1', algorithm='DDPG')

    loaded = ray.get(learner.warm_start_actor_from_elite.remote(
        {'weights': elite_weights}, 1.0))
    weights = ray.get(learner.get_policy_weights.remote())
    actor_key = next(key for key in elite_weights if key.startswith('actor.'))

    assert loaded > 0
    assert np.allclose(weights[actor_key], elite_weights[actor_key])
    ray.shutdown()


def test_ray_actors():
    ray.init(ignore_reinit_error=True, num_cpus=2)
    info = get_env_info('Pendulum-v1')
    template = build_model_template(info['state_dim'], info['action_dim'], algorithm='DDPG')
    worker = RolloutWorker.remote('Pendulum-v1', 100, 'DDPG')
    individual = Individual(id=1, weights=template, seed=42)
    fitness = ray.get(worker.evaluate.remote(individual))
    assert isinstance(fitness, float)

    learner = RLLearner.remote(env_name='Pendulum-v1', algorithm='DDPG')
    elites = [{'seed': 42, 'weights': template, 'id': 1, 'fitness': fitness}]
    n = ray.get(learner.reproduce_from_elites.remote(elites, max_episode_steps=100))
    assert n >= 1
    stats = ray.get(learner.get_stats.remote())
    assert stats['ea_transitions_size'] > 0
    ray.shutdown()
