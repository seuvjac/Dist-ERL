"""Shared RL / Re2 training step helpers."""

from typing import Any, Dict, List, Optional, Tuple

import ray

from src.config import uses_migration, uses_reproduction


def run_standard_rl_step(
    learner,
    rl_rollouts: int,
    max_episode_steps: int,
    seed: int,
    rl_updates: int,
) -> Tuple[Dict[str, Any], int]:
    rl_info = ray.get(learner.collect_rl_trajectories.remote(
        num_episodes=rl_rollouts,
        max_episode_steps=max_episode_steps,
        seed=seed,
    ))
    extra_steps = rl_info['total_steps']
    for _ in range(rl_updates):
        ray.get(learner.update_step.remote())
    return rl_info, extra_steps


def run_re2_sync_step(
    manager,
    learner,
    mode: str,
    ablation: str,
    generation: int,
    sync_interval: int,
    elite_seeds_k: int,
    max_episode_steps: int,
    rl_updates: int,
    rl_rollouts_between_sync: int,
    seed: int,
    skip_migration: bool = False,
    allow_migration: bool = True,
    inject_noise: float = 0.05,
    migration_copies: int = 1,
    migration_blend: float = 1.0,
    warm_start_rl: bool = False,
    warm_start_blend: float = 1.0,
) -> Dict[str, Any]:
    """One Re2 sync cycle or lightweight RL between syncs."""
    stats = {
        'sync_applied': 0,
        'reproduced_trajectories': 0,
        'migrated': 0,
        'rl_updates': 0,
        'extra_env_steps': 0,
        'federated_warm_start': 0,
        'migration_copies': 0,
    }

    if generation % sync_interval != 0:
        if rl_rollouts_between_sync > 0:
            _, extra = run_standard_rl_step(
                learner, rl_rollouts_between_sync, max_episode_steps, seed, max(1, rl_updates // 3)
            )
            stats['extra_env_steps'] = extra
            stats['rl_updates'] = max(1, rl_updates // 3)
        return stats

    stats['sync_applied'] = 1
    do_repro = uses_reproduction(mode, ablation)
    do_migrate = uses_migration(mode, ablation)

    elite_inds = []
    if do_repro or warm_start_rl:
        elite_inds = ray.get(manager.get_elite_individuals.remote(elite_seeds_k))

    if warm_start_rl and elite_inds:
        loaded = ray.get(learner.warm_start_actor_from_elite.remote(
            elite_inds[0], warm_start_blend))
        stats['federated_warm_start'] = int(loaded > 0)

    if do_repro:
        repro_count = ray.get(learner.reproduce_from_elites.remote(
            elite_inds, max_episode_steps
        ))
        stats['reproduced_trajectories'] = repro_count

    for _ in range(rl_updates):
        ray.get(learner.update_step.remote())
    stats['rl_updates'] = rl_updates

    if do_migrate and allow_migration and not skip_migration:
        rl_weights = ray.get(learner.get_policy_weights.remote())
        inserted = ray.get(manager.inject_rl_individual.remote(
            rl_weights, inject_noise, migration_copies, migration_blend))
        stats['migrated'] = int(inserted > 0)
        stats['migration_copies'] = int(inserted)

    return stats
