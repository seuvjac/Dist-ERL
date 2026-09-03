"""Basic tests for FedEvoRL components."""

import pytest
import numpy as np
import ray
from src.manager import EAManager
from src.federated import FederatedClient
from src.worker import RolloutWorker
from src.learner import RLLearner
from src.utils.replay_buffer import HybridReplayBuffer
from src.utils.environment import get_env_info, make_env
from src.utils.policies import SACPolicy
from src.utils.policy_utils import build_model_template
from src.utils.individual import Individual
from src.main import FED_EVOSAC_ENVS
from scripts.train_continuous_sac_baseline import _rollout as baseline_rollout


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


def test_ant_v5_continuous_env_and_client_dynamics():
    info = get_env_info('Ant-v5')
    assert info['state_dim'] == 105
    assert info['action_dim'] == 8

    low = make_env(
        'Ant-v5', client_id=0, heterogeneity=0.15,
        heterogeneity_mode='env_params_only')
    high = make_env(
        'Ant-v5', client_id=2, heterogeneity=0.15,
        heterogeneity_mode='env_params_only')
    try:
        low_obs, _ = low.reset(seed=7)
        high_obs, _ = high.reset(seed=7)
        assert low_obs.shape == high_obs.shape == (105,)
        assert not np.allclose(
            low.unwrapped.model.body_mass,
            high.unwrapped.model.body_mass,
        )
        assert low.unwrapped.model.opt.gravity[2] != high.unwrapped.model.opt.gravity[2]
    finally:
        low.close()
        high.close()


def test_pusher_v5_continuous_env_and_client_dynamics():
    assert 'Pusher-v5' in FED_EVOSAC_ENVS
    assert 'HalfCheetah-v5' in FED_EVOSAC_ENVS
    info = get_env_info('Pusher-v5')
    assert info['state_dim'] == 23
    assert info['action_dim'] == 7

    low = make_env(
        'Pusher-v5', max_episode_steps=100, client_id=0,
        heterogeneity=0.15, heterogeneity_mode='env_params_only')
    high = make_env(
        'Pusher-v5', max_episode_steps=100, client_id=2,
        heterogeneity=0.15, heterogeneity_mode='env_params_only')
    try:
        low_obs, _ = low.reset(seed=7)
        high_obs, _ = high.reset(seed=7)
        assert low_obs.shape == high_obs.shape == (23,)
        assert low._max_episode_steps == high._max_episode_steps == 100
        assert not np.allclose(
            low.unwrapped.model.body_mass,
            high.unwrapped.model.body_mass,
        )
        assert not np.allclose(
            low.unwrapped.model.geom_friction,
            high.unwrapped.model.geom_friction,
        )
    finally:
        low.close()
        high.close()


def test_walker2d_uses_mirrored_gait_heterogeneity_without_reward_scaling():
    low = make_env(
        'Walker2d-v5', client_id=0, heterogeneity=0.30,
        heterogeneity_mode='env_params_only')
    middle = make_env(
        'Walker2d-v5', client_id=1, heterogeneity=0.30,
        heterogeneity_mode='env_params_only')
    high = make_env(
        'Walker2d-v5', client_id=2, heterogeneity=0.30,
        heterogeneity_mode='env_params_only')
    try:
        low_model = low.unwrapped.model
        middle_model = middle.unwrapped.model
        high_model = high.unwrapped.model

        # Right/left leg perturbations are mirrored across edge clients;
        # client 1 remains the canonical center MDP.
        assert low_model.body_mass[2] < middle_model.body_mass[2]
        assert low_model.body_mass[5] > middle_model.body_mass[5]
        assert high_model.body_mass[2] > middle_model.body_mass[2]
        assert high_model.body_mass[5] < middle_model.body_mass[5]
        assert low_model.actuator_gear[0, 0] > middle_model.actuator_gear[0, 0]
        assert low_model.actuator_gear[3, 0] < middle_model.actuator_gear[3, 0]
        assert high_model.actuator_gear[0, 0] < middle_model.actuator_gear[0, 0]
        assert high_model.actuator_gear[3, 0] > middle_model.actuator_gear[3, 0]
        assert low.unwrapped._healthy_reward == middle.unwrapped._healthy_reward == 1.0
        assert high.unwrapped._forward_reward_weight == 1.0
    finally:
        low.close()
        middle.close()
        high.close()


def test_federated_walker_evaluation_reports_locomotion_diagnostics():
    ray.init(ignore_reinit_error=True, num_cpus=1)
    info = get_env_info('Walker2d-v5')
    weights = build_model_template(
        info['state_dim'], info['action_dim'], algorithm='SAC', seed=7)
    client = FederatedClient.remote(
        client_id=0,
        env_name='Walker2d-v5',
        algorithm='SAC',
        max_episode_steps=25,
        heterogeneity=0.30,
        heterogeneity_mode='env_params_only',
        seed=7,
        env_kwargs={'healthy_reward': 0.05, 'forward_reward_weight': 1.0},
    )
    result = ray.get(client.evaluate_weights.remote(weights, seed=99, num_episodes=1))
    assert 0 < result['episode_length_mean'] <= 25
    assert np.isfinite(result['forward_return_mean'])
    assert np.isfinite(result['survive_return_mean'])
    assert np.isfinite(result['ctrl_return_mean'])
    assert np.isfinite(result['x_displacement_mean'])
    assert np.isfinite(result['x_velocity_mean'])
    assert result['survive_return_mean'] <= 0.05 * result['episode_length_mean'] + 1e-6
    ray.shutdown()


def test_baseline_walker_rollout_uses_locomotion_reward_profile():
    info = get_env_info('Walker2d-v5')
    policy = SACPolicy(info['state_dim'], info['action_dim'])
    _, _, steps, diagnostics = baseline_rollout(
        policy,
        'Walker2d-v5',
        max_steps=25,
        seed=101,
        client_id=2,
        heterogeneity=0.30,
        heterogeneity_mode='env_params_only',
        train=False,
        env_kwargs={'healthy_reward': 0.05, 'forward_reward_weight': 1.0},
    )
    assert 0 < steps <= 25
    assert diagnostics['episode_length_mean'] == steps
    assert diagnostics['survive_return_mean'] <= 0.05 * steps + 1e-6
    assert np.isfinite(diagnostics['x_velocity_mean'])


def test_hopper_locomotion_profile_removes_survival_reward_shortcut():
    info = get_env_info('Hopper-v5')
    policy = SACPolicy(info['state_dim'], info['action_dim'])
    _, _, steps, diagnostics = baseline_rollout(
        policy,
        'Hopper-v5',
        max_steps=25,
        seed=103,
        client_id=1,
        heterogeneity=0.25,
        heterogeneity_mode='env_params_only',
        train=False,
        env_kwargs={'healthy_reward': 0.05, 'forward_reward_weight': 1.0},
    )
    assert 0 < steps <= 25
    assert diagnostics['episode_length_mean'] == steps
    assert diagnostics['survive_return_mean'] <= 0.05 * steps + 1e-6
    assert np.isfinite(diagnostics['forward_return_mean'])
    assert np.isfinite(diagnostics['x_velocity_mean'])


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


def test_ea_manager_seed_reproduces_initial_population():
    ray.init(ignore_reinit_error=True, num_cpus=2)
    info = get_env_info('Pendulum-v1')
    template = build_model_template(info['state_dim'], info['action_dim'], algorithm='DDPG')
    manager_a = EAManager.remote(population_size=4, seed=17)
    manager_b = EAManager.remote(population_size=4, seed=17)
    ray.get([
        manager_a.initialize_population.remote(template),
        manager_b.initialize_population.remote(template),
    ])
    population_a, population_b = ray.get([
        manager_a.get_population_for_evaluation.remote(),
        manager_b.get_population_for_evaluation.remote(),
    ])
    actor_key = next(key for key in template if key.startswith('actor.'))
    for individual_a, individual_b in zip(population_a, population_b):
        assert individual_a['seed'] == individual_b['seed']
        assert np.array_equal(
            individual_a['weights'][actor_key], individual_b['weights'][actor_key])

    fitness = [{'id': idx, 'fitness': float(idx)} for idx in range(4)]
    ray.get([
        manager_a.update_fitness.remote(fitness),
        manager_b.update_fitness.remote(fitness),
    ])
    ray.get([manager_a.evolve_population.remote(), manager_b.evolve_population.remote()])
    evolved_a, evolved_b = ray.get([
        manager_a.get_population_for_evaluation.remote(),
        manager_b.get_population_for_evaluation.remote(),
    ])
    for individual_a, individual_b in zip(evolved_a, evolved_b):
        assert individual_a['seed'] == individual_b['seed']
        assert np.array_equal(
            individual_a['weights'][actor_key], individual_b['weights'][actor_key])
    ray.shutdown()


def test_ea_manager_anchor_perturb_uses_reproducible_standard_anchor():
    ray.init(ignore_reinit_error=True, num_cpus=2)
    info = get_env_info('Pendulum-v1')
    template = build_model_template(
        info['state_dim'], info['action_dim'], algorithm='SAC', seed=123)
    manager_a = EAManager.remote(population_size=4, seed=17)
    manager_b = EAManager.remote(population_size=4, seed=91)
    ray.get([
        manager_a.initialize_population.remote(template, 'anchor_perturb', 55, 0.03),
        manager_b.initialize_population.remote(template, 'anchor_perturb', 55, 0.03),
    ])
    population_a, population_b = ray.get([
        manager_a.get_population_for_evaluation.remote(),
        manager_b.get_population_for_evaluation.remote(),
    ])
    actor_key = next(key for key in template if key.startswith('actor.'))
    assert np.array_equal(population_a[0]['weights'][actor_key], template[actor_key])
    assert np.array_equal(
        population_a[1]['weights'][actor_key], population_b[1]['weights'][actor_key])
    assert not np.array_equal(
        population_a[0]['weights'][actor_key], population_a[1]['weights'][actor_key])
    ray.shutdown()


def test_ea_manager_antithetic_init_pairs_mean_actor_and_freezes_log_std():
    ray.init(ignore_reinit_error=True, num_cpus=1)
    info = get_env_info('Pendulum-v1')
    template = build_model_template(
        info['state_dim'], info['action_dim'], algorithm='SAC', seed=123)
    manager = EAManager.remote(
        population_size=5,
        seed=17,
        ga_config={
            'actor_exclude_substrings': ('actor.log_std.',),
            'mutation_scale_floor': 0.05,
        },
    )
    ray.get(manager.initialize_population.remote(
        template, 'anchor_antithetic', 55, 0.12))
    population = ray.get(manager.get_population_for_evaluation.remote())
    mean_key = next(key for key in template if key.startswith('actor.mean.'))
    log_std_key = next(key for key in template if key.startswith('actor.log_std.'))
    assert np.allclose(
        population[1]['weights'][mean_key] + population[2]['weights'][mean_key],
        2.0 * template[mean_key],
    )
    assert np.array_equal(population[0]['weights'][log_std_key], template[log_std_key])
    assert np.array_equal(population[1]['weights'][log_std_key], template[log_std_key])
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
