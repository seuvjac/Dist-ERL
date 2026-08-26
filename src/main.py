"""Main entry point for FedEvoRL / Dist-ERL baselines."""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import ray

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    ABLATION_CHOICES,
    ABLATION_FULL,
    ALL_MODES,
    DIST_ERL,
    EA_MODES,
    ERL_RE2,
    FED_ABLATION_CHOICES,
    FED_ABLATION_FULL,
    FED_ABLATION_NO_EA_INJECTION,
    FED_ABLATION_NO_HETEROGENEITY,
    FED_ABLATION_NO_LOCAL_RL,
    FED_ABLATION_RAW_SOFTMAX,
    FED_ABLATION_UNIFORM_AGG,
    FED_EVO_RL,
    RE2_MODES,
    RL_MODES,
    STANDARD_ERL,
)
from src.federated import FederatedClient, aggregate_weight_dicts, weight_entropy
from src.learner import RLLearner
from src.manager import EAManager
from src.sync_policy import MigrationGate
from src.training import run_re2_sync_step, run_standard_rl_step
from src.utils.environment import apply_headless_mujoco_runtime, get_env_info
from src.utils.policy_utils import build_model_template
from src.worker import RolloutWorker

FED_EVOFSAC_ENVS = ('CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'LunarLander-v3')
FED_EVOSAC_ENVS = (
    'Ant-v5',
    'Pusher-v5',
    'Swimmer-v5',
    'Reacher-v5',
    'Walker2d-v5',
    'Hopper-v5',
)

METRIC_FIELDS = [
    'generation', 'communication_round', 'total_env_steps', 'eval_reward_mean', 'eval_reward_std',
    'eval_ea_mean', 'eval_ea_std',
    'best_fitness', 'mean_fitness', 'rl_steps', 'buffer_size',
    'gen_time', 'total_time', 'sync_applied', 'reproduced_trajectories', 'migrated',
    'weight_diversity', 'fitness_std', 'comm_upload_bytes', 'comm_full_traj_bytes',
    'stagnation_boost', 'rl_reset', 'migration_allowed', 'migration_gate',
    'eval_rl_aligned', 'ea_median_fitness', 'policy_exploration_noise',
    'federated_warm_start', 'migration_copies',
    'client_reward_mean', 'client_reward_std', 'client_fitness_mean',
    'client_fitness_std', 'selected_clients', 'aggregation_entropy',
    'aggregation_score_mean', 'aggregation_score_std', 'aggregation_score_mode',
    'fed_aggregation_round',
    'fed_round_applied', 'archive_best', 'archive_size',
    'deployable_eval_mean', 'deployable_eval_std',
    'aggregation_temperature', 'deployment_rollback', 'deployment_rollback_count',
    'candidate_eval_mean', 'candidate_eval_std', 'local_updates_per_worker',
    'local_candidate_accept_rate', 'local_candidate_gain_mean',
    'accepted_client_uploads', 'injection_reference_score',
    'stagnation_boost_count',
    'eval_episode_length_mean', 'eval_forward_return_mean',
    'eval_survive_return_mean', 'eval_ctrl_return_mean',
    'eval_x_displacement_mean', 'eval_x_velocity_mean',
]

EVALUATION_DIAGNOSTIC_KEYS = (
    'episode_length_mean',
    'forward_return_mean',
    'survive_return_mean',
    'ctrl_return_mean',
    'x_displacement_mean',
    'x_velocity_mean',
)


def parse_args():
    parser = argparse.ArgumentParser(description="FedEvoRL: EA-guided Federated Reinforcement Learning")

    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--max-episode-steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--population-size', type=int, default=50, help='Population size')
    parser.add_argument('--elite-fraction', type=float, default=0.2,
                        help='Used only if --num-elitists is not set (elite count = max(1, pop*frac))')
    parser.add_argument('--num-elitists', type=int, default=1,
                        help='EA elites E (paper uses 1; ERL-Re2 baseline often uses elite_fraction)')
    parser.add_argument('--ea-mutation-prob', type=float, default=0.9, help='P(mutate) per non-elite')
    parser.add_argument('--ea-mutation-beta-frac', type=float, default=0.7,
                        help='Fraction of columns mutated per action row (ERL-Re² beta)')
    parser.add_argument('--ea-mutation-scale-mode', type=str, default='element',
                        choices=['element', 'layer_rms'],
                        help='Scale mutation by each value or by the source layer RMS')
    parser.add_argument('--ea-mutation-scale-floor', type=float, default=1e-3,
                        help='Minimum layer scale used by layer-RMS mutation')
    parser.add_argument('--ea-mutate-bias', action='store_true',
                        help='Mutate actor biases together with selected weight rows')
    parser.add_argument('--ea-freeze-sac-log-std', action='store_true',
                        help='Exclude SAC log_std from EA/crossover and federated sharing')
    parser.add_argument('--ea-prob-reset-and-super', type=float, default=0.05,
                        help='Prob of drastic / reset mutation (ERL-Re² prob_reset_and_sup)')
    parser.add_argument('--ea-init-mode', type=str, default='gaussian',
                        choices=['gaussian', 'anchor_perturb', 'anchor_antithetic'],
                        help='EA population initialization: independent Gaussian or a SAC anchor plus perturbations')
    parser.add_argument('--ea-init-seed', type=int, default=-1,
                        help='Optional fixed seed for the EA initial population; -1 uses --seed')
    parser.add_argument('--ea-init-noise-scale', type=float, default=0.05,
                        help='Relative layer-scale perturbation used by --ea-init-mode anchor_perturb')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of rollout workers')
    parser.add_argument('--num-clients', type=int, default=4, help='Number of federated RL clients')
    parser.add_argument('--client-fraction', type=float, default=1.0,
                        help='Fraction of clients sampled in each federated round')
    parser.add_argument('--client-rollouts', type=int, default=2,
                        help='Local rollout episodes per selected federated client')
    parser.add_argument('--client-updates', type=int, default=10,
                        help='Local gradient updates per selected federated client')
    parser.add_argument('--client-critic-warmup-updates', type=int, default=4,
                        help='Critic-only updates after loading the server actor')
    parser.add_argument('--client-actor-lr', type=float, default=0.0,
                        help='Optional SAC client actor learning rate; <=0 uses --lr')
    parser.add_argument('--client-validation-episodes', type=int, default=0,
                        help='Held-out episodes used to reject a degrading local actor upload')
    parser.add_argument('--client-rollback-margin', type=float, default=0.0,
                        help='Relative validation degradation tolerated before local rollback')
    parser.add_argument('--client-heterogeneity', type=float, default=0.2,
                        help='Synthetic client MDP heterogeneity strength')
    parser.add_argument('--client-heterogeneity-mode', type=str, default='env_params',
                        choices=[
                            'none',
                            'reward_action_noise',
                            'env_params',
                            'env_params_only',
                            'mixed',
                            'reward_scale_only',
                            'env_params_reward_scale',
                        ],
                        help='How client-local MDP heterogeneity is applied')
    parser.add_argument('--walker-healthy-reward', type=float, default=1.0,
                        help='Explicit Walker2d healthy reward; 1.0 is Gymnasium default')
    parser.add_argument('--walker-forward-reward-weight', type=float, default=1.0,
                        help='Explicit Walker2d forward reward weight')
    parser.add_argument('--hopper-healthy-reward', type=float, default=1.0,
                        help='Explicit Hopper healthy reward; 1.0 is Gymnasium default')
    parser.add_argument('--hopper-forward-reward-weight', type=float, default=1.0,
                        help='Explicit Hopper forward reward weight')
    parser.add_argument('--fed-aggregation', type=str, default='softmax',
                        choices=['fitness', 'uniform', 'softmax'],
                        help='Federated aggregation rule for client model uploads')
    parser.add_argument('--fed-aggregation-interval', type=int, default=5,
                        help='Run federated local train/aggregation every K generations')
    parser.add_argument('--fed-aggregation-temperature', type=float, default=75.0,
                        help='Softmax temperature for fitness-weighted client aggregation')
    parser.add_argument('--fed-raw-softmax-temperature', type=float, default=75.0,
                        help='Temperature used only by the raw-softmax ablation')
    parser.add_argument('--fed-score-normalization', type=str, default='relative_gain',
                        choices=['raw', 'batch_zscore', 'relative_gain', 'uniform'],
                        help='Normalize client rewards before reward-aware aggregation')
    parser.add_argument('--fed-score-warmup-rounds', type=int, default=0,
                        help='Use --fed-score-warmup-normalization for the first N federated rounds')
    parser.add_argument('--fed-score-warmup-normalization', type=str, default='batch_zscore',
                        choices=['raw', 'batch_zscore', 'relative_gain', 'uniform'],
                        help='Temporary score normalization during early federated rounds')
    parser.add_argument('--fed-score-ema-beta', type=float, default=0.9,
                        help='EMA beta for per-client reward baseline in relative_gain aggregation')
    parser.add_argument('--fed-score-min-std', type=float, default=1.0,
                        help='Minimum per-client reward std used by relative_gain aggregation')
    parser.add_argument('--fed-score-scale', type=float, default=1.0,
                        help='Scale non-raw aggregation scores to preserve softmax selection pressure')
    parser.add_argument('--fed-min-client-score-quantile', type=float, default=0.25,
                        help='Drop selected client uploads below this reward quantile before aggregation')
    parser.add_argument('--fed-delta-clip-norm', type=float, default=5.0,
                        help='Clip each client update delta before federated aggregation')
    parser.add_argument('--fed-inject-margin', type=float, default=-0.05,
                        help='Relative margin vs current EA best required before injecting aggregated actor')
    parser.add_argument('--fed-injection-warmup-rounds', type=int, default=0,
                        help='Skip Fed-to-EA injection for the first N federated rounds')
    parser.add_argument('--elite-archive-size', type=int, default=5,
                        help='Global top-k EA archive for FedEvoRL')
    parser.add_argument('--elite-archive-restore-copies', type=int, default=1,
                        help='Number of archived elites pinned back after each FedEvoRL generation')
    parser.add_argument('--archive-eval-candidates', type=int, default=3,
                        help='Top current EA candidates independently re-evaluated for archive admission')
    parser.add_argument('--archive-eval-episodes', type=int, default=3,
                        help='Fixed-seed episodes per client for independent archive validation')
    parser.add_argument('--archive-std-penalty', type=float, default=0.0,
                        help='Risk-aware archive score: mean_return - penalty * eval_std')
    parser.add_argument('--fed-inject-std-penalty', type=float, default=0.0,
                        help='Risk-aware Fed-to-EA injection score: candidate_mean - penalty * candidate_std')
    parser.add_argument('--algorithm', type=str, default='FSAC', choices=['FSAC', 'SAC', 'DDPG', 'TD3', 'PPO'],
                        help='RL algorithm (discrete uses FSAC; continuous FedEvoSAC uses SAC)')
    parser.add_argument('--policy-exploration-noise', type=float, default=0.1,
                        help='Exploration noise for deterministic continuous baselines')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max-generations', type=int, default=100, help='Maximum generations')
    parser.add_argument('--target-env-steps', type=int, default=0,
                        help='Continue training until at least this many environment interactions are counted')
    parser.add_argument('--sync-interval', type=int, default=20, help='RL-EA Re2 sync interval for erl_re2')
    parser.add_argument('--eval-interval', type=int, default=1, help='Evaluation interval in generations')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--rl-rollouts', type=int, default=2, help='RL rollout episodes per generation (standard ERL)')
    parser.add_argument('--rl-updates', type=int, default=10, help='Gradient steps per generation/sync')
    parser.add_argument('--elite-seeds', type=int, default=5, help='Top-k elite seeds for reproduction')
    parser.add_argument('--rl-rollouts-between-sync', type=int, default=1,
                        help='Lightweight RL rollouts between Re2 syncs')
    parser.add_argument('--ea-batch-ratio', type=float, default=0.3,
                        help='Fraction of replay batch from EA reproduced transitions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--repeat-id', type=str, default='',
                        help='Optional outer-repeat identifier for experiment bookkeeping')
    parser.add_argument('--seed-slot', type=int, default=-1,
                        help='Seed position inside an outer repeat')
    parser.add_argument('--stagnation-patience', type=int, default=12,
                        help='Generations without RL eval improvement before diversity boost')
    parser.add_argument('--stagnation-min-delta', type=float, default=5.0,
                        help='Minimum eval_reward improvement to reset stagnation counter')
    parser.add_argument('--stagnation-max-boosts', type=int, default=-1,
                        help='Maximum diversity boosts; negative keeps the legacy unlimited behavior')
    parser.add_argument('--stagnation-boost-cooldown', type=int, default=0,
                        help='Minimum generations between diversity boosts')
    parser.add_argument('--immigrant-fraction', type=float, default=0.15,
                        help='Fraction of population replaced on stagnation boost')
    parser.add_argument('--inject-noise', type=float, default=0.05,
                        help='Gaussian noise on actor weights when migrating RL into EA')
    parser.add_argument('--ea-weight-clip', type=float, default=5.0,
                        help='Clip evolved actor weights to keep EA mutation bounded')
    parser.add_argument('--migration-copies', type=int, default=3,
                        help='Number of weak non-elite EA individuals softly blended with RL on migration')
    parser.add_argument('--migration-blend', type=float, default=0.35,
                        help='Soft blend coefficient for RL weights during RL->EA migration')
    parser.add_argument('--warm-start-blend', type=float, default=0.65,
                        help='Blend RL actor toward the best EA elite on early erl_re2 syncs')
    parser.add_argument('--warm-start-generations', type=int, default=2,
                        help='Number of erl_re2 sync points that warm-start RL from the best EA elite')
    parser.add_argument('--migration-margin', type=float, default=0.05,
                        help='Relative margin above EA median required before RL->EA migration')
    parser.add_argument('--rl-reset-patience', type=int, default=20,
                        help='RL eval stall generations before actor tail soft-reset (buffer kept)')
    parser.add_argument('--migration-warmup-frac', type=float, default=0.3,
                        help='Fraction of training with migration disabled (early decoupling)')
    parser.add_argument('--migration-rl-beats-ea-gens', type=int, default=3,
                        help='Consecutive gens RL(actor-aligned) > EA median before migration allowed')
    parser.add_argument('--no-dynamic-migration', action='store_true',
                        help='Disable performance/warmup migration gate (always migrate on sync)')

    parser.add_argument('--mode', type=str, default=FED_EVO_RL, choices=list(ALL_MODES),
                        help='Training mode (FedEvoRL is the main method)')
    parser.add_argument('--ablation', type=str, default=ABLATION_FULL, choices=list(ABLATION_CHOICES),
                        help='Re2 ablation variant (erl_re2 only)')
    parser.add_argument('--fed-ablation', type=str, default=FED_ABLATION_FULL,
                        choices=list(FED_ABLATION_CHOICES),
                        help='FedEvoRL ablation variant (fed_evo_rl only)')

    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--wandb-key', type=str, default=None, help='Wandb API key')
    parser.add_argument('--wandb-project', type=str, default='FedEvoRL', help='Wandb project name')
    parser.add_argument('--exp-name', type=str, default=None, help='Experiment name')

    return parser.parse_args()


def _setup_local_logger(args):
    exp_name = args.exp_name or f"{args.mode}_{args.env}_{args.algorithm}_{args.seed}"
    if args.mode in RE2_MODES and args.ablation != ABLATION_FULL:
        exp_name = f"{exp_name}_abl_{args.ablation}"
    run_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)

    metadata = {
        'mode': args.mode,
        'ablation': args.ablation if args.mode in RE2_MODES else 'n/a',
        'fed_ablation': args.fed_ablation if args.mode == FED_EVO_RL else 'n/a',
        'env': args.env,
        'algorithm': args.algorithm,
        'seed': args.seed,
        'repeat_id': args.repeat_id,
        'seed_slot': args.seed_slot,
        'population_size': args.population_size,
        'num_workers': args.num_workers,
        'max_generations': args.max_generations,
        'target_env_steps': args.target_env_steps,
        'max_episode_steps': args.max_episode_steps,
        'ea_init_mode': args.ea_init_mode,
        'ea_init_seed': args.ea_init_seed,
        'ea_init_noise_scale': args.ea_init_noise_scale,
        'ea_mutation_scale_mode': args.ea_mutation_scale_mode,
        'ea_mutation_scale_floor': args.ea_mutation_scale_floor,
        'ea_mutate_bias': args.ea_mutate_bias,
        'ea_freeze_sac_log_std': args.ea_freeze_sac_log_std,
        'eval_episodes': args.eval_episodes,
        'batch_size': args.batch_size,
        'sync_interval': args.sync_interval,
        'elite_seeds': args.elite_seeds,
        'rl_updates': args.rl_updates,
        'ea_batch_ratio': args.ea_batch_ratio,
        'inject_noise': args.inject_noise,
        'migration_copies': args.migration_copies,
        'migration_blend': args.migration_blend,
        'warm_start_blend': args.warm_start_blend,
        'warm_start_generations': args.warm_start_generations,
        'migration_margin': args.migration_margin,
        'rl_reset_patience': args.rl_reset_patience,
        'migration_warmup_frac': args.migration_warmup_frac,
        'policy_exploration_noise': args.policy_exploration_noise,
        'num_clients': args.num_clients,
        'client_fraction': args.client_fraction,
        'client_rollouts': args.client_rollouts,
        'client_updates': args.client_updates,
        'client_critic_warmup_updates': args.client_critic_warmup_updates,
        'client_actor_lr': args.client_actor_lr,
        'client_validation_episodes': args.client_validation_episodes,
        'client_rollback_margin': args.client_rollback_margin,
        'client_heterogeneity': args.client_heterogeneity,
        'client_heterogeneity_mode': args.client_heterogeneity_mode,
        'walker_healthy_reward': args.walker_healthy_reward,
        'walker_forward_reward_weight': args.walker_forward_reward_weight,
        'hopper_healthy_reward': args.hopper_healthy_reward,
        'hopper_forward_reward_weight': args.hopper_forward_reward_weight,
        'fed_aggregation': args.fed_aggregation,
        'fed_aggregation_interval': args.fed_aggregation_interval,
        'fed_aggregation_temperature': args.fed_aggregation_temperature,
        'fed_raw_softmax_temperature': args.fed_raw_softmax_temperature,
        'fed_score_normalization': args.fed_score_normalization,
        'fed_score_warmup_rounds': args.fed_score_warmup_rounds,
        'fed_score_warmup_normalization': args.fed_score_warmup_normalization,
        'fed_score_ema_beta': args.fed_score_ema_beta,
        'fed_score_min_std': args.fed_score_min_std,
        'fed_min_client_score_quantile': args.fed_min_client_score_quantile,
        'fed_delta_clip_norm': args.fed_delta_clip_norm,
        'fed_inject_margin': args.fed_inject_margin,
        'fed_injection_warmup_rounds': args.fed_injection_warmup_rounds,
        'ea_weight_clip': args.ea_weight_clip,
        'elite_archive_size': args.elite_archive_size,
        'elite_archive_restore_copies': args.elite_archive_restore_copies,
        'archive_eval_candidates': args.archive_eval_candidates,
        'archive_eval_episodes': args.archive_eval_episodes,
        'archive_std_penalty': args.archive_std_penalty,
        'fed_inject_std_penalty': args.fed_inject_std_penalty,
        'fed_score_scale': args.fed_score_scale,
        'stagnation_patience': args.stagnation_patience,
        'stagnation_min_delta': args.stagnation_min_delta,
        'stagnation_max_boosts': args.stagnation_max_boosts,
        'stagnation_boost_cooldown': args.stagnation_boost_cooldown,
        'immigrant_fraction': args.immigrant_fraction,
    }
    with open(os.path.join(run_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    metrics_path = os.path.join(run_dir, 'metrics.csv')
    with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=METRIC_FIELDS).writeheader()
    return run_dir, metrics_path


def _append_local_metrics(metrics_path, log_data):
    row = {key: log_data.get(key, '') for key in METRIC_FIELDS}
    with open(metrics_path, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=METRIC_FIELDS).writerow(row)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _validate_algorithm_for_env(args, env_info) -> None:
    action_space = env_info.get('action_space')
    is_discrete = hasattr(action_space, 'n')
    algo = args.algorithm.upper()
    if args.mode == FED_EVO_RL:
        if is_discrete:
            if args.env not in FED_EVOFSAC_ENVS or algo != 'FSAC':
                raise ValueError(f"{args.env} must use --algorithm FSAC for discrete FedEvoFSAC")
        else:
            if args.env not in FED_EVOSAC_ENVS or algo != 'SAC':
                raise ValueError(f"{args.env} must use --algorithm SAC for continuous FedEvoSAC")
        return
    if is_discrete and algo != 'FSAC':
        raise ValueError(f"{args.env} has a discrete action space; use --algorithm FSAC")
    if not is_discrete and algo == 'FSAC':
        raise ValueError(f"{args.env} has a continuous action space; FSAC is discrete only")


def _cap_num_workers(requested: int, mode: str) -> int:
    """Avoid Ray scheduling stalls when workers + learner exceed CPU budget."""
    cpus = os.cpu_count() or 4
    reserve = 3 if mode in RL_MODES and mode in EA_MODES else 2
    budget = max(1, cpus - reserve)
    capped = min(requested, budget)
    if capped < requested:
        _log(f"  Capped num_workers {requested} -> {capped} (cpus={cpus}, reserve={reserve})")
    return capped


def _estimate_comm_bytes(population_size, state_dim, action_dim, max_episode_steps):
    """Seed upload vs hypothetical full trajectory upload."""
    upload = population_size * (4 + 4)  # seed + fitness
    full_traj = population_size * max_episode_steps * (state_dim + action_dim + 4) * 4
    return int(upload), int(full_traj)


def _mean_evaluation_diagnostics(results):
    """Average scalar rollout diagnostics across evaluated clients."""
    return {
        key: float(np.mean([float(row.get(key, 0.0)) for row in results]))
        for key in EVALUATION_DIAGNOSTIC_KEYS
    }


def _accepted_client_uploads(client_indices, train_results):
    """Return only locally validated actor updates for server aggregation."""
    accepted = [
        (int(idx), result)
        for idx, result in zip(client_indices, train_results)
        if int(result.get('candidate_accepted', 1)) == 1
    ]
    return (
        [idx for idx, _ in accepted],
        [result.get('upload_score', result['avg_reward']) for _, result in accepted],
        [result['weights'] for _, result in accepted],
    )


def _apply_fed_ablation_args(args) -> None:
    """Translate named FedEvoRL ablations into concrete CLI settings."""
    if args.mode != FED_EVO_RL:
        return
    if args.fed_ablation == FED_ABLATION_UNIFORM_AGG:
        args.fed_aggregation = 'uniform'
    elif args.fed_ablation == FED_ABLATION_NO_LOCAL_RL:
        args.client_updates = 0
        args.client_rollouts = 0
        args.client_validation_episodes = 0
        args.migration_copies = 0
    elif args.fed_ablation == FED_ABLATION_NO_EA_INJECTION:
        args.migration_copies = 0
    elif args.fed_ablation == FED_ABLATION_NO_HETEROGENEITY:
        args.client_heterogeneity = 0.0
        args.client_heterogeneity_mode = 'none'
    elif args.fed_ablation == FED_ABLATION_RAW_SOFTMAX:
        args.fed_aggregation = 'softmax'
        args.fed_score_normalization = 'raw'
        args.fed_aggregation_temperature = args.fed_raw_softmax_temperature
        args.fed_score_warmup_rounds = 0
        args.fed_injection_warmup_rounds = 0


def _aggregation_scores_from_rewards(
    client_indices,
    rewards,
    reward_mean,
    reward_var,
    reward_seen,
    mode: str,
    ema_beta: float,
    min_std: float,
    score_scale: float = 1.0,
):
    """Return aggregation scores and update per-client reward scale statistics."""
    if not rewards:
        return []
    raw = np.asarray(rewards, dtype=np.float64)
    indices = [int(i) for i in client_indices]
    mode = str(mode)
    if mode == 'raw':
        scores = raw.copy()
    elif mode == 'uniform':
        scores = np.zeros_like(raw)
    elif mode == 'batch_zscore':
        scores = (raw - raw.mean()) / (raw.std() + 1e-8)
    else:
        vals = []
        floor = max(1e-8, float(min_std))
        for idx, reward in zip(indices, raw):
            if reward_seen[idx] > 0:
                scale = max(floor, float(np.sqrt(max(reward_var[idx], 0.0))))
                vals.append((float(reward) - float(reward_mean[idx])) / scale)
            else:
                vals.append(0.0)
        scores = np.asarray(vals, dtype=np.float64)
    if mode not in ('raw', 'uniform'):
        scores = scores * float(max(0.0, score_scale))

    beta = float(np.clip(ema_beta, 0.0, 0.999))
    floor_var = max(1e-8, float(min_std) ** 2)
    for idx, reward in zip(indices, raw):
        reward = float(reward)
        if reward_seen[idx] <= 0:
            reward_mean[idx] = reward
            reward_var[idx] = floor_var
            reward_seen[idx] = 1
            continue
        prev_mean = float(reward_mean[idx])
        reward_mean[idx] = beta * prev_mean + (1.0 - beta) * reward
        reward_var[idx] = beta * float(reward_var[idx]) + (1.0 - beta) * (reward - prev_mean) ** 2
        reward_seen[idx] += 1
    return [float(x) for x in scores]


def _run_fed_evo_rl(args, env_info, metrics_path):
    """EA-guided federated RL training loop."""
    if args.algorithm.upper() not in ('FSAC', 'SAC'):
        raise ValueError("FedEvoRL supports FSAC (discrete) or SAC (continuous)")
    ga_config = {
        'mutation_prob': args.ea_mutation_prob,
        'mutation_beta_frac': args.ea_mutation_beta_frac,
        'prob_reset_and_super': args.ea_prob_reset_and_super,
        'actor_prefix': 'actor.',
        'actor_exclude_substrings': (
            ('actor.log_std.',) if args.ea_freeze_sac_log_std else ()
        ),
        'mutation_scale_mode': args.ea_mutation_scale_mode,
        'mutation_scale_floor': args.ea_mutation_scale_floor,
        'mutate_bias': args.ea_mutate_bias,
        'weight_clip': args.ea_weight_clip,
    }
    manager = EAManager.remote(
        args.population_size,
        args.elite_fraction,
        args.num_elitists,
        ga_config,
        args.seed,
    )
    template_seed = args.ea_init_seed if args.ea_init_seed >= 0 else args.seed
    template = build_model_template(
        env_info['state_dim'], env_info['action_dim'], algorithm=args.algorithm,
        discrete=hasattr(env_info.get('action_space'), 'n'), seed=template_seed)
    ray.get(manager.initialize_population.remote(
        template,
        args.ea_init_mode,
        args.ea_init_seed if args.ea_init_seed >= 0 else None,
        args.ea_init_noise_scale,
    ))

    env_kwargs = {}
    if args.env == 'Walker2d-v5':
        env_kwargs = {
            'healthy_reward': float(args.walker_healthy_reward),
            'forward_reward_weight': float(args.walker_forward_reward_weight),
        }
    elif args.env == 'Hopper-v5':
        env_kwargs = {
            'healthy_reward': float(args.hopper_healthy_reward),
            'forward_reward_weight': float(args.hopper_forward_reward_weight),
        }
    clients = [
        FederatedClient.remote(
            client_id=i,
            env_name=args.env,
            algorithm=args.algorithm,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            lr=args.lr,
            max_episode_steps=args.max_episode_steps,
            heterogeneity=args.client_heterogeneity,
            heterogeneity_mode=args.client_heterogeneity_mode,
            policy_exploration_noise=args.policy_exploration_noise,
            seed=args.seed,
            actor_lr=args.client_actor_lr if args.client_actor_lr > 0 else None,
            env_kwargs=env_kwargs,
        )
        for i in range(args.num_clients)
    ]

    start_time = time.time()
    total_env_steps = 0
    eval_reward_history = []
    total_env_steps_history = []
    rl_steps_history = []
    selected_count = max(1, int(round(args.num_clients * np.clip(args.client_fraction, 0.0, 1.0))))
    last_client_reward_mean = 0.0
    last_client_reward_std = 0.0
    last_aggregation_score_mean = 0.0
    last_aggregation_score_std = 0.0
    client_reward_ema = np.zeros(args.num_clients, dtype=np.float64)
    client_reward_var = np.ones(args.num_clients, dtype=np.float64) * max(1e-8, args.fed_score_min_std ** 2)
    client_reward_seen = np.zeros(args.num_clients, dtype=np.int64)
    cumulative_rl_updates = 0
    fed_aggregation_count = 0
    stagnation_best = float('-inf')
    stagnation_count = 0
    stagnation_boost_count = 0
    last_stagnation_boost_generation = -10**9

    generation = 0
    while (
        (args.target_env_steps > 0 and total_env_steps < args.target_env_steps)
        or (args.target_env_steps <= 0 and generation < args.max_generations)
    ):
        gen_start = time.time()
        population = ray.get(manager.get_population_for_evaluation.remote())

        eval_refs = []
        eval_keys = []
        generation_eval_seed = args.seed + 1_000_003 * generation + 10_007
        for individual in population:
            for client in clients:
                eval_refs.append(client.evaluate_weights.remote(
                    individual['weights'], generation_eval_seed, max(1, args.eval_episodes // 2)))
                eval_keys.append(individual['id'])
        eval_results = ray.get(eval_refs)
        population_eval_steps = int(sum(result.get('total_steps', 0) for result in eval_results))

        fitness_by_id = {individual['id']: [] for individual in population}
        for ind_id, result in zip(eval_keys, eval_results):
            fitness_by_id[ind_id].append(result['fitness'])
        fitness_rows = [
            {'id': ind_id, 'fitness': float(np.mean(values)) if values else 0.0}
            for ind_id, values in fitness_by_id.items()
        ]
        ray.get(manager.update_fitness.remote(fitness_rows))

        archive_candidates = ray.get(manager.get_elite_individuals.remote(
            min(args.archive_eval_candidates, args.population_size)))
        archive_seed = args.seed + 50_000_003
        archive_rows = []
        archive_eval_steps = 0
        for candidate in archive_candidates:
            refs = [
                client.evaluate_weights.remote(
                    candidate['weights'], archive_seed, args.archive_eval_episodes)
                for client in clients
            ]
            results = ray.get(refs)
            archive_eval_steps += int(sum(result.get('total_steps', 0) for result in results))
            scores = [float(result['fitness']) for result in results]
            archive_rows.append({
                **candidate,
                'fitness': float(np.mean(scores)),
                'fitness_std': float(np.std(scores)),
                **_mean_evaluation_diagnostics(results),
            })
        ray.get(manager.update_elite_archive_evaluated.remote(
            archive_rows, args.elite_archive_size, args.archive_std_penalty))
        archive_stats_before_fed = ray.get(manager.get_archive_stats.remote())
        ray.get(manager.evolve_population.remote())
        ray.get(manager.restore_elite_archive.remote(args.elite_archive_restore_copies))

        best = ray.get(manager.get_best_individual.remote())
        local_refinement_enabled = args.client_rollouts > 0 and args.client_updates > 0
        fed_round_applied = int(
            local_refinement_enabled
            and generation % max(1, args.fed_aggregation_interval) == 0
        )
        train_results = []
        aggregated = {}
        aggregated_shared = {}
        inserted = 0
        client_indices = []
        if fed_round_applied:
            client_indices = np.random.choice(
                np.arange(args.num_clients), size=selected_count, replace=False)
            train_refs = [
                clients[int(idx)].local_train.remote(
                    best.weights, args.client_rollouts, args.client_updates,
                    args.seed + generation * 1000,
                    args.client_critic_warmup_updates,
                    args.client_validation_episodes,
                    args.client_rollback_margin,
                    args.ea_freeze_sac_log_std,
                )
                for idx in client_indices
            ]
            train_results = ray.get(train_refs)
        client_rewards = [r.get('upload_score', r['avg_reward']) for r in train_results]
        aggregation_client_indices, aggregation_rewards, client_weights = (
            _accepted_client_uploads(client_indices, train_results)
        )
        aggregation_round_index = fed_aggregation_count if fed_round_applied else -1
        aggregation_score_mode = args.fed_score_normalization
        if fed_round_applied and fed_aggregation_count < max(0, args.fed_score_warmup_rounds):
            aggregation_score_mode = args.fed_score_warmup_normalization
        aggregation_scores = _aggregation_scores_from_rewards(
            aggregation_client_indices,
            aggregation_rewards,
            client_reward_ema,
            client_reward_var,
            client_reward_seen,
            aggregation_score_mode,
            args.fed_score_ema_beta,
            args.fed_score_min_std,
            args.fed_score_scale,
        ) if aggregation_rewards else []
        if aggregation_rewards:
            # A uniform/FedAvg ablation must include every selected client.
            # Score-based filtering here would silently turn it into a
            # trimmed average and invalidate the intended comparison.
            q = float(np.clip(args.fed_min_client_score_quantile, 0.0, 1.0))
            min_score = (
                float(np.quantile(aggregation_scores, q))
                if args.fed_aggregation != 'uniform' and len(aggregation_scores) > 1
                else None
            )
            aggregated_shared = aggregate_weight_dicts(
                client_weights,
                aggregation_scores,
                mode=args.fed_aggregation,
                temperature=args.fed_aggregation_temperature,
                min_score=min_score,
                base_weights=best.weights,
                delta_clip_norm=args.fed_delta_clip_norm,
            )
            if aggregated_shared:
                # Client-local SAC exploration parameters are absent from the
                # upload. Reattach the server copy for deterministic
                # evaluation and archive storage without federating log_std.
                aggregated = {
                    key: np.array(value, copy=True)
                    for key, value in best.weights.items()
                }
                aggregated.update(aggregated_shared)
        current_best = float(best.fitness)
        aggregated_eval_mean = float('-inf')
        aggregated_eval_std = 0.0
        aggregated_eval_steps = 0
        if aggregated:
            refs = [
                client.evaluate_weights.remote(
                    aggregated, archive_seed, args.archive_eval_episodes)
                for client in clients
            ]
            results = ray.get(refs)
            aggregated_eval_steps = int(sum(result.get('total_steps', 0) for result in results))
            scores = [float(result['fitness']) for result in results]
            aggregated_eval_mean = float(np.mean(scores))
            aggregated_eval_std = float(np.std(scores))
        injection_reference_score = float(
            archive_stats_before_fed.get('archive_best_score', current_best)
        )
        inject_threshold = (
            injection_reference_score
            + abs(injection_reference_score) * args.fed_inject_margin
        )
        aggregated_eval_score = (
            aggregated_eval_mean
            - max(0.0, args.fed_inject_std_penalty) * aggregated_eval_std
        )
        injection_warmup_active = (
            fed_round_applied
            and fed_aggregation_count < max(0, args.fed_injection_warmup_rounds)
        )
        inject_ok = aggregated_eval_score >= inject_threshold and not injection_warmup_active
        if aggregated and args.migration_copies > 0 and inject_ok:
            ray.get(manager.update_elite_archive_evaluated.remote([{
                'id': args.population_size + generation,
                'seed': int(archive_seed),
                'weights': aggregated,
                'fitness': aggregated_eval_mean,
                'fitness_std': aggregated_eval_std,
                **_mean_evaluation_diagnostics(results),
            }], args.elite_archive_size, args.archive_std_penalty))
            inserted = ray.get(manager.inject_rl_individual.remote(
                aggregated, args.inject_noise, args.migration_copies, args.migration_blend))
            ray.get(manager.restore_elite_archive.remote(args.elite_archive_restore_copies))

        total_env_steps += (
            population_eval_steps
            + int(sum(result.get('total_steps', 0) for result in train_results))
            + archive_eval_steps
            + aggregated_eval_steps
        )
        stats = ray.get(manager.get_stats.remote())
        upload_b, full_b = _estimate_comm_bytes(
            args.population_size, env_info['state_dim'], env_info['action_dim'], args.max_episode_steps)
        fed_upload = (
            selected_count * sum(arr.nbytes for arr in aggregated_shared.values())
            if aggregated_shared else 0
        )

        if client_rewards:
            client_reward_mean = float(np.mean(client_rewards))
            client_reward_std = float(np.std(client_rewards))
            last_client_reward_mean = client_reward_mean
            last_client_reward_std = client_reward_std
            aggregation_score_mean = float(np.mean(aggregation_scores)) if aggregation_scores else 0.0
            aggregation_score_std = float(np.std(aggregation_scores)) if aggregation_scores else 0.0
            last_aggregation_score_mean = aggregation_score_mean
            last_aggregation_score_std = aggregation_score_std
        else:
            client_reward_mean = last_client_reward_mean
            client_reward_std = last_client_reward_std
            aggregation_score_mean = last_aggregation_score_mean
            aggregation_score_std = last_aggregation_score_std
        client_fitness_values = [r['fitness'] for r in eval_results]
        deployable_eval_mean = float(stats.get('archive_best', stats.get('max_fitness', 0.0)))
        deployable_eval_std = float(stats.get('archive_best_std', 0.0))
        stagnation_boost = 0
        if deployable_eval_mean > stagnation_best + max(0.0, args.stagnation_min_delta):
            stagnation_best = deployable_eval_mean
            stagnation_count = 0
        else:
            stagnation_count += 1
        budget_remaining = args.target_env_steps <= 0 or total_env_steps < args.target_env_steps
        boost_limit_ok = (
            args.stagnation_max_boosts < 0
            or stagnation_boost_count < args.stagnation_max_boosts
        )
        boost_cooldown_ok = (
            generation - last_stagnation_boost_generation
            >= max(0, args.stagnation_boost_cooldown)
        )
        if (
            budget_remaining
            and args.stagnation_patience > 0
            and stagnation_count >= args.stagnation_patience
            and boost_limit_ok
            and boost_cooldown_ok
        ):
            stagnation_boost = ray.get(manager.boost_diversity.remote(
                args.immigrant_fraction, 0.35, 0.12))
            ray.get(manager.restore_elite_archive.remote(args.elite_archive_restore_copies))
            stagnation_count = 0
            stagnation_boost_count += 1
            last_stagnation_boost_generation = generation
        eval_reward_history.append(deployable_eval_mean)
        total_env_steps_history.append(total_env_steps)
        cumulative_rl_updates += int(sum(r.get('updates_applied', 0) for r in train_results))
        rl_steps_history.append(cumulative_rl_updates)
        active_client_count = selected_count if fed_round_applied else 0
        display_aggregation = (
            f"{args.fed_aggregation}/{aggregation_score_mode}"
            if fed_round_applied else "none"
        )

        log_data = {
            'generation': generation,
            # Every EA generation broadcasts actors for distributed client
            # evaluation and receives scalar fitness, so it is one protocol
            # communication round even without a SAC aggregation event.
            'communication_round': generation + 1,
            'total_env_steps': total_env_steps,
            # For FedEvoSAC, the deployable policy is the global elite/archive actor.
            # Local RL rollout scores remain in client_reward_mean/std.
            'eval_reward_mean': deployable_eval_mean,
            'eval_reward_std': deployable_eval_std,
            'eval_ea_mean': stats['max_fitness'],
            'best_fitness': stats['max_fitness'],
            'mean_fitness': stats['mean_fitness'],
            'fitness_std': stats.get('std_fitness', 0.0),
            'weight_diversity': stats.get('weight_diversity', 0.0),
            'rl_steps': rl_steps_history[-1],
            'buffer_size': int(sum(r.get('buffer_size', 0) for r in train_results)),
            'gen_time': time.time() - gen_start,
            'total_time': time.time() - start_time,
            'migrated': int(inserted > 0),
            'migration_copies': inserted,
            'comm_upload_bytes': upload_b + fed_upload,
            'comm_full_traj_bytes': full_b,
            'policy_exploration_noise': args.policy_exploration_noise,
            'client_reward_mean': client_reward_mean,
            'client_reward_std': client_reward_std,
            'client_fitness_mean': float(np.mean(client_fitness_values)) if client_fitness_values else 0.0,
            'client_fitness_std': float(np.std(client_fitness_values)) if client_fitness_values else 0.0,
            'selected_clients': active_client_count,
            'aggregation_entropy': weight_entropy(
                aggregation_scores,
                mode=args.fed_aggregation,
                temperature=args.fed_aggregation_temperature,
            ) if aggregation_scores else 0.0,
            'aggregation_score_mean': aggregation_score_mean,
            'aggregation_score_std': aggregation_score_std,
            'aggregation_score_mode': aggregation_score_mode if client_rewards else '',
            'fed_aggregation_round': aggregation_round_index,
            'fed_round_applied': fed_round_applied,
            'archive_best': stats.get('archive_best', 0.0),
            'archive_size': stats.get('archive_size', 0),
            'deployable_eval_mean': deployable_eval_mean,
            'deployable_eval_std': deployable_eval_std,
            'aggregation_temperature': args.fed_aggregation_temperature,
            'candidate_eval_mean': aggregated_eval_mean if np.isfinite(aggregated_eval_mean) else '',
            'candidate_eval_std': aggregated_eval_std if np.isfinite(aggregated_eval_mean) else '',
            'local_updates_per_worker': (
                float(np.mean([r.get('updates_applied', 0) for r in train_results]))
                if train_results else 0.0
            ),
            'local_candidate_accept_rate': (
                float(np.mean([r.get('candidate_accepted', 1) for r in train_results]))
                if train_results else 0.0
            ),
            'local_candidate_gain_mean': (
                float(np.mean([r.get('candidate_validation_gain', 0.0) for r in train_results]))
                if train_results else 0.0
            ),
            'accepted_client_uploads': len(aggregation_rewards),
            'injection_reference_score': injection_reference_score,
            'stagnation_boost': stagnation_boost,
            'stagnation_boost_count': stagnation_boost_count,
            'eval_episode_length_mean': stats.get(
                'archive_best_episode_length_mean', 0.0),
            'eval_forward_return_mean': stats.get(
                'archive_best_forward_return_mean', 0.0),
            'eval_survive_return_mean': stats.get(
                'archive_best_survive_return_mean', 0.0),
            'eval_ctrl_return_mean': stats.get(
                'archive_best_ctrl_return_mean', 0.0),
            'eval_x_displacement_mean': stats.get(
                'archive_best_x_displacement_mean', 0.0),
            'eval_x_velocity_mean': stats.get(
                'archive_best_x_velocity_mean', 0.0),
        }
        _append_local_metrics(metrics_path, log_data)
        _log(
            f"Gen {generation}: deploy={deployable_eval_mean:.2f}, "
            f"fed_rollout={client_reward_mean:.2f} +/- {client_reward_std:.2f}, "
            f"ea_best={stats['max_fitness']:.2f}, diversity={stats.get('weight_diversity', 0.0):.3f}, "
            f"clients={active_client_count}/{args.num_clients}, agg={display_aggregation}, "
            f"accept={log_data['local_candidate_accept_rate']:.2f}, "
            f"x_vel={log_data['eval_x_velocity_mean']:.3f}, "
            f"inject_warmup={int(injection_warmup_active)}"
        )
        if args.wandb:
            import wandb
            wandb.log(log_data)
        if fed_round_applied:
            fed_aggregation_count += 1
        generation += 1

    if eval_reward_history:
        _generate_training_plot(
            total_env_steps_history, eval_reward_history,
            rl_steps_history or [0] * len(total_env_steps_history),
            args.env, args.mode,
        )


def main():
    args = parse_args()
    _apply_fed_ablation_args(args)
    apply_headless_mujoco_runtime()
    np.random.seed(args.seed)
    if args.mode == FED_EVO_RL:
        args.num_clients = _cap_num_workers(args.num_clients, args.mode)
    else:
        args.num_workers = _cap_num_workers(args.num_workers, args.mode)
    ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level='warning')

    _log("Starting FedEvoRL Training")
    _log(f"  Environment: {args.env}")
    _log(f"  Mode: {args.mode}")
    _log(f"  MUJOCO_GL={os.environ.get('MUJOCO_GL', '?')}")
    if args.mode in RE2_MODES:
        _log(f"  Ablation: {args.ablation}")
    if args.mode == FED_EVO_RL:
        _log(f"  Fed ablation: {args.fed_ablation}")
    _log(f"  Population: {args.population_size}, Workers: {args.num_workers}, Clients: {args.num_clients}")
    _log(f"  RL algorithm: {args.algorithm}")
    _log(f"  Sync interval: {args.sync_interval}, Elite seeds: {args.elite_seeds}")

    if args.mode == ERL_RE2 and args.num_workers > 1:
        _log(f"  Note: erl_re2 typically uses num_workers=1 (current: {args.num_workers})")
    if args.mode == DIST_ERL and args.num_workers < 2:
        _log(f"  Note: dist_erl comparison uses distributed workers (current: {args.num_workers})")

    run_dir, metrics_path = _setup_local_logger(args)
    _log(f"  Local metrics: {metrics_path}")

    env_info = get_env_info(args.env)
    _validate_algorithm_for_env(args, env_info)
    _log(f"  State dim: {env_info['state_dim']}, Action dim: {env_info['action_dim']} "
         f"(gym_id={env_info.get('gym_id', args.env)})")

    if args.wandb:
        try:
            import wandb
            if args.wandb_key:
                os.environ['WANDB_API_KEY'] = args.wandb_key
            exp_name = args.exp_name or f"{args.mode}_{args.env}_{args.algorithm}_{args.seed}"
            wandb.login(key=args.wandb_key, relogin=False)
            wandb.init(project=args.wandb_project, name=exp_name, config=vars(args))
        except Exception as e:
            print(f"  Wandb failed: {e}")
            args.wandb = False

    if args.mode == FED_EVO_RL:
        _run_fed_evo_rl(args, env_info, metrics_path)
        _log("Training completed.")
        _log(f"Metrics saved to: {run_dir}")
        ray.shutdown()
        return

    manager = None
    learner = None
    workers = []

    if args.mode in EA_MODES:
        ga_config = {
            'mutation_prob': args.ea_mutation_prob,
            'mutation_beta_frac': args.ea_mutation_beta_frac,
            'prob_reset_and_super': args.ea_prob_reset_and_super,
            'actor_prefix': 'actor.',
            'actor_exclude_substrings': (
                ('actor.log_std.',) if args.ea_freeze_sac_log_std else ()
            ),
            'mutation_scale_mode': args.ea_mutation_scale_mode,
            'mutation_scale_floor': args.ea_mutation_scale_floor,
            'mutate_bias': args.ea_mutate_bias,
            'weight_clip': args.ea_weight_clip,
        }
        manager = EAManager.remote(
            args.population_size,
            args.elite_fraction,
            args.num_elitists,
            ga_config,
            args.seed,
        )
        template = build_model_template(
            env_info['state_dim'], env_info['action_dim'], algorithm=args.algorithm,
            discrete=hasattr(env_info.get('action_space'), 'n'))
        ray.get(manager.initialize_population.remote(template))

    if args.mode in RL_MODES:
        learner = RLLearner.remote(
            env_name=args.env,
            algorithm=args.algorithm,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            lr=args.lr,
            ea_batch_ratio=args.ea_batch_ratio,
            policy_exploration_noise=args.policy_exploration_noise,
        )

    if args.mode in EA_MODES:
        _log(f"  Spawning {args.num_workers} RolloutWorker(s)...")
        workers = [
            RolloutWorker.remote(
                args.env, args.max_episode_steps, args.algorithm,
                env_info['state_dim'], env_info['action_dim'],
            )
            for _ in range(args.num_workers)
        ]
        _log("  RolloutWorkers created (MuJoCo env loads on first rollout).")

    eval_reward_history = []
    rl_steps_history = []
    total_env_steps_history = []
    total_env_steps = 0
    start_time = time.time()
    best_eval_rl = float('-inf')
    stagnation_counter = 0
    rl_stall_counter = 0
    skip_migration_next_sync = False
    migration_gate = None
    if args.mode in RE2_MODES and not args.no_dynamic_migration:
        migration_gate = MigrationGate(
            args.max_generations,
            warmup_frac=args.migration_warmup_frac,
            beats_required=args.migration_rl_beats_ea_gens,
            margin=args.migration_margin,
        )
    last_allow_migration = True
    last_migration_reason = 'init'
    last_ea_median = 0.0

    for generation in range(args.max_generations):
        gen_start_time = time.time()
        re2_stats = {}

        if args.mode in EA_MODES:
            if generation == 0 or generation % max(1, args.eval_interval) == 0:
                _log(f"Gen {generation}: evaluating pop={args.population_size} "
                     f"on {len(workers)} worker(s)...")
            ray.get(manager.evaluate_population.remote(workers))
            total_env_steps += args.population_size * args.max_episode_steps
            ray.get(manager.evolve_population.remote())

        eval_stats = {}
        stagnation_boost = 0
        rl_reset = 0
        allow_migration = last_allow_migration
        migration_gate_reason = last_migration_reason
        ea_median = last_ea_median
        if generation % args.eval_interval == 0:
            if args.mode == 'pure_ea':
                best = ray.get(manager.get_best_individual.remote())
                eval_stats = ray.get(workers[0].evaluate_individual.remote(
                    best, num_episodes=args.eval_episodes, max_episode_steps=args.max_episode_steps))
            elif args.mode in RL_MODES:
                if args.mode in RE2_MODES:
                    eval_stats = ray.get(learner.evaluate_policy_actor_aligned.remote(
                        num_episodes=args.eval_episodes,
                        max_episode_steps=args.max_episode_steps,
                        seed=args.seed,
                    ))
                else:
                    eval_stats = ray.get(learner.evaluate_policy.remote(
                        num_episodes=args.eval_episodes,
                        max_episode_steps=args.max_episode_steps,
                        seed=args.seed,
                    ))
                if manager and workers:
                    best = ray.get(manager.get_best_individual.remote())
                    ea_eval = ray.get(workers[0].evaluate_individual.remote(
                        best, num_episodes=args.eval_episodes,
                        max_episode_steps=args.max_episode_steps))
                    eval_stats['eval_ea_mean'] = ea_eval['eval_reward_mean']
                    eval_stats['eval_ea_std'] = ea_eval['eval_reward_std']

                erl = eval_stats.get('eval_reward_mean')
                if manager:
                    ea_median = ray.get(manager.get_median_fitness.remote())
                eval_stats['eval_rl_aligned'] = erl
                eval_stats['ea_median_fitness'] = ea_median

                if migration_gate is not None and erl is not None:
                    allow_migration, migration_gate_reason = migration_gate.allow_migration(
                        generation, erl, ea_median)
                last_allow_migration = allow_migration
                last_migration_reason = migration_gate_reason
                last_ea_median = ea_median

                if erl is not None:
                    if erl > best_eval_rl + args.stagnation_min_delta:
                        best_eval_rl = erl
                        stagnation_counter = 0
                        rl_stall_counter = 0
                    else:
                        stagnation_counter += 1
                        rl_stall_counter += 1

                    if (rl_stall_counter >= args.rl_reset_patience
                            and args.mode in RE2_MODES and learner):
                        ray.get(learner.reset_actor_tail.remote())
                        ray.get(learner.boost_rl_exploration.remote(0.85))
                        rl_reset = 1
                        rl_stall_counter = 0
                        stagnation_counter = 0
                        skip_migration_next_sync = True
                        _log(f"Gen {generation}: RL soft-reset (actor tail + exploration boost), "
                             f"buffer kept")

                    if (stagnation_counter >= args.stagnation_patience
                            and manager and args.mode in EA_MODES):
                        n_imm = ray.get(manager.boost_diversity.remote(
                            args.immigrant_fraction))
                        if learner:
                            ray.get(learner.boost_rl_exploration.remote(0.9))
                        stagnation_boost = 1
                        stagnation_counter = 0
                        skip_migration_next_sync = True
                        _log(f"Gen {generation}: stagnation boost — immigrants={n_imm}, "
                             f"exploration boost, skip next RL migration")

        if args.mode == 'pure_rl':
            _, extra = run_standard_rl_step(
                learner, args.rl_rollouts, args.max_episode_steps, args.seed, args.rl_updates)
            total_env_steps += extra
        elif args.mode in (STANDARD_ERL, DIST_ERL):
            _, extra = run_standard_rl_step(
                learner, args.rl_rollouts, args.max_episode_steps, args.seed, args.rl_updates)
            total_env_steps += extra
        elif args.mode in RE2_MODES:
            migrate_ok = allow_migration if migration_gate else True
            re2_stats = run_re2_sync_step(
                manager, learner, args.mode, args.ablation, generation,
                args.sync_interval, args.elite_seeds, args.max_episode_steps,
                args.rl_updates, args.rl_rollouts_between_sync, args.seed,
                skip_migration=skip_migration_next_sync,
                allow_migration=migrate_ok,
                inject_noise=args.inject_noise,
                migration_copies=args.migration_copies,
                migration_blend=args.migration_blend,
                warm_start_rl=(generation // max(1, args.sync_interval)) < args.warm_start_generations,
                warm_start_blend=args.warm_start_blend,
            )
            if skip_migration_next_sync:
                skip_migration_next_sync = False
            total_env_steps += re2_stats.get('extra_env_steps', 0)

        gen_time = time.time() - gen_start_time
        total_time = time.time() - start_time
        total_env_steps_history.append(total_env_steps)

        log_data = {
            'generation': generation,
            'total_time': total_time,
            'total_env_steps': total_env_steps,
            'gen_time': gen_time,
            'sync_applied': re2_stats.get('sync_applied', 0),
            'reproduced_trajectories': re2_stats.get('reproduced_trajectories', 0),
            'migrated': re2_stats.get('migrated', 0),
            'federated_warm_start': re2_stats.get('federated_warm_start', 0),
            'migration_copies': re2_stats.get('migration_copies', 0),
        }

        if manager:
            ms = ray.get(manager.get_stats.remote())
            upload_b, full_b = _estimate_comm_bytes(
                args.population_size, env_info['state_dim'], env_info['action_dim'], args.max_episode_steps)
            log_data.update({
                'best_fitness': ms['max_fitness'],
                'mean_fitness': ms['mean_fitness'],
                'weight_diversity': ms.get('weight_diversity', 0.0),
                'fitness_std': ms.get('std_fitness', 0.0),
                'comm_upload_bytes': upload_b,
                'comm_full_traj_bytes': full_b,
            })
            line = (f"Gen {generation}: best_fitness={ms['max_fitness']:.2f}, "
                    f"mean={ms['mean_fitness']:.2f}, "
                    f"diversity={ms.get('weight_diversity', 0):.3f}, "
                    f"elite={ms.get('num_elitists', 1)}")
            if stagnation_boost:
                line += " [stagnation-boost]"
            line += f", time={gen_time:.2f}s"
            _log(line)

        if learner:
            ls = ray.get(learner.get_stats.remote())
            log_data.update({'rl_steps': ls['training_steps'], 'buffer_size': ls['buffer_size']})
            log_data['policy_exploration_noise'] = ls.get('policy_exploration_noise', '')
            rl_steps_history.append(ls['training_steps'])

        if args.mode in RE2_MODES:
            log_data.update({
                'migration_allowed': int(allow_migration),
                'migration_gate': migration_gate_reason,
                'rl_reset': rl_reset,
            })

        if eval_stats:
            log_data.update({
                'eval_reward_mean': eval_stats.get('eval_reward_mean', 0.0),
                'eval_reward_std': eval_stats.get('eval_reward_std', 0.0),
                'eval_ea_mean': eval_stats.get('eval_ea_mean', ''),
                'eval_ea_std': eval_stats.get('eval_ea_std', ''),
                'stagnation_boost': stagnation_boost,
                'eval_rl_aligned': eval_stats.get('eval_rl_aligned', ''),
                'ea_median_fitness': eval_stats.get('ea_median_fitness', ''),
            })
            eval_reward_history.append(eval_stats.get('eval_reward_mean', 0.0))
            msg = (f"Gen {generation}: eval_rl={eval_stats.get('eval_reward_mean', 0.0):.2f} "
                   f"+/- {eval_stats.get('eval_reward_std', 0.0):.2f}")
            if args.mode in RE2_MODES:
                msg += f", ea_med={ea_median:.1f}, migrate={'Y' if allow_migration else 'N'}"
            if 'eval_ea_mean' in eval_stats:
                msg += (f", eval_ea={eval_stats['eval_ea_mean']:.2f} "
                        f"+/- {eval_stats.get('eval_ea_std', 0.0):.2f}")
            if stagnation_counter > 0:
                msg += f", stall={stagnation_counter}/{args.stagnation_patience}"
            if rl_stall_counter > 0 and args.mode in RE2_MODES:
                msg += f", rl_stall={rl_stall_counter}/{args.rl_reset_patience}"
            _log(msg)

        _append_local_metrics(metrics_path, log_data)
        sys.stdout.flush()
        if args.wandb:
            import wandb
            wandb.log(log_data)

    _log("Training completed.")
    _log(f"Metrics saved to: {run_dir}")

    if eval_reward_history:
        _generate_training_plot(
            total_env_steps_history, eval_reward_history,
            rl_steps_history or [0] * len(total_env_steps_history),
            args.env, args.mode,
        )
    ray.shutdown()


def _generate_training_plot(total_steps, rewards, rl_steps, env_name, mode):
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(total_steps[:len(rewards)], rewards, linewidth=2, label='Eval Reward')
        ax1.set_xlabel('Total Environment Steps')
        ax1.set_ylabel('Eval Reward Mean')
        ax1.set_title(f'{mode} on {env_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(total_steps, rl_steps, 'g-', linewidth=2)
        ax2.set_xlabel('Total Environment Steps')
        ax2.set_ylabel('RL Training Steps')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        os.makedirs('plots/training', exist_ok=True)
        path = f'plots/training/training_{mode}_{env_name}_{int(time.time())}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Training plot: {path}")
        plt.close()
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
