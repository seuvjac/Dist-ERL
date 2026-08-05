"""Experiment modes and benchmark presets for FedEvoRL comparisons."""

# --- Training modes (baselines + full method) ---
PURE_RL = 'pure_rl'
PURE_EA = 'pure_ea'
STANDARD_ERL = 'standard_erl'      # EA + RL, single-worker evaluation
DIST_ERL = 'dist_erl'              # EA + RL, distributed evaluation baseline
ERL_RE2 = 'erl_re2'                # Re2 baseline: EA + RL + Re2, single-worker eval
FED_EVO_RL = 'fed_evo_rl'          # Main method: EA-guided federated RL

EA_MODES = (STANDARD_ERL, DIST_ERL, PURE_EA, ERL_RE2)
RL_MODES = (PURE_RL, STANDARD_ERL, DIST_ERL, ERL_RE2)
RE2_MODES = (ERL_RE2,)
FEDERATED_MODES = (FED_EVO_RL,)

ALL_MODES = (PURE_RL, PURE_EA, STANDARD_ERL, DIST_ERL, ERL_RE2, FED_EVO_RL)

# --- Ablation flags (only apply to RE2_MODES) ---
ABLATION_FULL = 'full'
ABLATION_NO_REPRODUCTION = 'no_reproduction'
ABLATION_NO_MIGRATION = 'no_migration'
ABLATION_NO_RE2 = 'no_re2'

ABLATION_CHOICES = (
    ABLATION_FULL,
    ABLATION_NO_REPRODUCTION,
    ABLATION_NO_MIGRATION,
    ABLATION_NO_RE2,
)

# FedEvoRL ablations used to isolate the main method components.
FED_ABLATION_FULL = 'full'
FED_ABLATION_UNIFORM_AGG = 'uniform_aggregation'
FED_ABLATION_NO_LOCAL_RL = 'no_local_rl'
FED_ABLATION_NO_EA_INJECTION = 'no_ea_injection'
FED_ABLATION_NO_HETEROGENEITY = 'no_heterogeneity'
FED_ABLATION_RAW_SOFTMAX = 'raw_softmax'

FED_ABLATION_CHOICES = (
    FED_ABLATION_FULL,
    FED_ABLATION_UNIFORM_AGG,
    FED_ABLATION_NO_LOCAL_RL,
    FED_ABLATION_NO_EA_INJECTION,
    FED_ABLATION_NO_HETEROGENEITY,
    FED_ABLATION_RAW_SOFTMAX,
)

MODE_LABELS = {
    PURE_RL: 'Pure FSAC',
    PURE_EA: 'Pure EA (ERL-Re² GA)',
    STANDARD_ERL: 'Standard ERL',
    DIST_ERL: 'Dist-ERL',
    ERL_RE2: 'ERL-Re2',
    FED_EVO_RL: 'FedEvoFSAC (ours)',
}

FED_ABLATION_LABELS = {
    FED_ABLATION_FULL: 'FedEvoFSAC (full)',
    FED_ABLATION_UNIFORM_AGG: 'w/o fitness aggregation',
    FED_ABLATION_NO_LOCAL_RL: 'w/o local RL updates',
    FED_ABLATION_NO_EA_INJECTION: 'w/o EA injection',
    FED_ABLATION_NO_HETEROGENEITY: 'IID clients',
    FED_ABLATION_RAW_SOFTMAX: 'raw reward softmax',
}

MODE_COLORS = {
    PURE_RL: '#0072B2',
    PURE_EA: '#009E73',
    STANDARD_ERL: '#E69F00',
    DIST_ERL: '#56B4E9',
    ERL_RE2: '#CC79A7',
    FED_EVO_RL: '#D55E00',
}

# High-contrast line styles for plots (colorblind-friendly tableau)
PLOT_STYLES = {
    PURE_RL: {'color': '#0072B2', 'ls': '-', 'lw': 2.8, 'marker': 'o', 'markevery': 12},
    PURE_EA: {'color': '#009E73', 'ls': '--', 'lw': 2.5, 'marker': 's', 'markevery': 12},
    STANDARD_ERL: {'color': '#E69F00', 'ls': '-.', 'lw': 2.5, 'marker': '^', 'markevery': 12},
    DIST_ERL: {'color': '#56B4E9', 'ls': ':', 'lw': 2.8, 'marker': 'D', 'markevery': 12},
    ERL_RE2: {'color': '#CC79A7', 'ls': '-', 'lw': 3.0, 'marker': 'v', 'markevery': 10},
    FED_EVO_RL: {'color': '#D55E00', 'ls': '-', 'lw': 3.2, 'marker': '*', 'markevery': 10},
    f'{ERL_RE2}__{ABLATION_NO_RE2}': {'color': '#56B4E9', 'ls': ':', 'lw': 2.8},
    f'{ERL_RE2}__{ABLATION_NO_REPRODUCTION}': {'color': '#F0E442', 'ls': '--', 'lw': 2.8},
    f'{ERL_RE2}__{ABLATION_NO_MIGRATION}': {'color': '#999999', 'ls': '-.', 'lw': 2.8},
    f'{ERL_RE2}__{ABLATION_FULL}': {'color': '#CC79A7', 'ls': '-', 'lw': 3.0},
    f'{FED_EVO_RL}__{FED_ABLATION_FULL}': {'color': '#D55E00', 'ls': '-', 'lw': 3.2, 'marker': '*', 'markevery': 10},
    f'{FED_EVO_RL}__{FED_ABLATION_UNIFORM_AGG}': {'color': '#0072B2', 'ls': '--', 'lw': 2.8, 'marker': 'o', 'markevery': 10},
    f'{FED_EVO_RL}__{FED_ABLATION_NO_LOCAL_RL}': {'color': '#009E73', 'ls': '-.', 'lw': 2.8, 'marker': 's', 'markevery': 10},
    f'{FED_EVO_RL}__{FED_ABLATION_NO_EA_INJECTION}': {'color': '#E69F00', 'ls': ':', 'lw': 3.0, 'marker': '^', 'markevery': 10},
    f'{FED_EVO_RL}__{FED_ABLATION_NO_HETEROGENEITY}': {'color': '#CC79A7', 'ls': '-', 'lw': 2.8, 'marker': 'D', 'markevery': 10},
    f'{FED_EVO_RL}__{FED_ABLATION_RAW_SOFTMAX}': {'color': '#882255', 'ls': '--', 'lw': 2.8, 'marker': 'v', 'markevery': 10},
}

# Six MuJoCo continuous control tasks (Todorov et al., 2012), Gymnasium *-v2 IDs
MUJOCO_V2_ENVS = [
    'Swimmer-v2',
    'Hopper-v2',
    'Ant-v2',
    'Pusher-v2',
    'Walker2d-v2',
    'Humanoid-v2',
]

FEDRL_HETEROGENEOUS_ENVS = [
    'CartPole-v1',
    'MountainCar-v0',
    'Acrobot-v1',
    'LunarLander-v3',
]

FEDRL_CONTINUOUS_ENVS = [
    'Ant-v5',
    'Pusher-v5',
    'Swimmer-v5',
    'Walker2d-v5',
    'Hopper-v5',
]

FEDRL_HETEROGENEITY_SCENARIOS = {
    'dynamics_mild': {
        'mode': 'env_params_only',
        'strength': 0.25,
        'label': 'Mild dynamics shift',
        'note': 'Client-specific gravity/mass/length/wind parameters only.',
    },
    'sensor_reward': {
        'mode': 'reward_action_noise',
        'strength': 0.35,
        'label': 'Sensor/reward shift',
        'note': 'Client-specific observation noise, reward scale/bias, and seed streams.',
    },
    'mixed_hard': {
        'mode': 'mixed',
        'strength': 0.50,
        'label': 'Hard mixed shift',
        'note': 'Dynamics shift plus observation/reward perturbations.',
    },
}


def env_run_preset(env_id: str) -> dict:
    """Per-task defaults for distributed runs (population, workers, horizon)."""
    presets = {
        'Swimmer-v2': {
            'population_size': 40, 'num_workers': 4,
            'max_generations': 200, 'max_episode_steps': 1000,
        },
        'Hopper-v2': {
            'population_size': 40, 'num_workers': 4,
            'max_generations': 200, 'max_episode_steps': 1000,
        },
        'Ant-v2': {
            'population_size': 50, 'num_workers': 8,
            'max_generations': 200, 'max_episode_steps': 1000,
        },
        'Walker2d-v2': {
            'population_size': 40, 'num_workers': 4,
            'max_generations': 200, 'max_episode_steps': 1000,
        },
        'Humanoid-v2': {
            'population_size': 30, 'num_workers': 4,
            'max_generations': 200, 'max_episode_steps': 1000,
        },
        'CartPole-v1': {
            'population_size': 24, 'num_workers': 2,
            'max_generations': 80, 'max_episode_steps': 500,
        },
        'MountainCar-v0': {
            'population_size': 24, 'num_workers': 2,
            'max_generations': 80, 'max_episode_steps': 200,
        },
        'Acrobot-v1': {
            'population_size': 24, 'num_workers': 2,
            'max_generations': 80, 'max_episode_steps': 500,
        },
        'LunarLander-v3': {
            'population_size': 30, 'num_workers': 4,
            'max_generations': 100, 'max_episode_steps': 1000,
        },
        'Swimmer-v5': {
            'population_size': 20, 'num_workers': 3,
            'max_generations': 80, 'max_episode_steps': 1000,
        },
        'Ant-v5': {
            'population_size': 12, 'num_workers': 3,
            'max_generations': 80, 'max_episode_steps': 1000,
        },
        'Pusher-v5': {
            'population_size': 12, 'num_workers': 3,
            'max_generations': 220, 'max_episode_steps': 100,
        },
        'Reacher-v5': {
            'population_size': 20, 'num_workers': 3,
            'max_generations': 80, 'max_episode_steps': 50,
        },
        'Walker2d-v5': {
            'population_size': 24, 'num_workers': 4,
            'max_generations': 80, 'max_episode_steps': 1000,
        },
        'BipedalWalker-v3': {
            'population_size': 24, 'num_workers': 4,
            'max_generations': 80, 'max_episode_steps': 1600,
        },
        'Hopper-v5': {
            'population_size': 24, 'num_workers': 4,
            'max_generations': 80, 'max_episode_steps': 1000,
        },
    }
    return presets.get(env_id, {
        'population_size': 30, 'num_workers': 4,
        'max_generations': 100, 'max_episode_steps': 1000,
    })


def _bench_entry(env_id: str, short: str, note: str) -> dict:
    p = env_run_preset(env_id)
    return {
        'id': env_id,
        'short': short,
        'note': note,
        **p,
    }


# Primary paper environments: MuJoCo-v2 + lightweight debug envs
BENCHMARK_ENVS = [
    _bench_entry('Swimmer-v2', 'Swimmer', 'MuJoCo v2 (main)'),
    _bench_entry('Hopper-v2', 'Hopper', 'MuJoCo v2 (main)'),
    _bench_entry('Ant-v2', 'Ant', 'MuJoCo v2 (main)'),
    _bench_entry('Walker2d-v2', 'Walker2d', 'MuJoCo v2 (main)'),
    _bench_entry('Humanoid-v2', 'Humanoid', 'MuJoCo v2 (main)'),
    _bench_entry('CartPole-v1', 'CartPole', 'FedRL heterogeneity sanity check'),
    _bench_entry('MountainCar-v0', 'MountainCar', 'FedRL exploration under heterogeneous hill dynamics'),
    _bench_entry('Acrobot-v1', 'Acrobot', 'FedRL heterogeneity sanity check'),
    _bench_entry('LunarLander-v3', 'LunarLanderDiscrete', 'FedRL heterogeneous Box2D'),
]

MUJOCO_ENVS = list(MUJOCO_V2_ENVS)
DEFAULT_SEEDS = list(range(10))

# Main comparison: related work / baselines, with FedEvoRL as the paper method
BASELINE_MODES = [PURE_RL, PURE_EA, STANDARD_ERL, ERL_RE2, DIST_ERL, FED_EVO_RL]

# Re2 mechanism ablation retained only for the ERL-Re2 baseline
RE2_ABLATION_VARIANTS = [
    (ABLATION_NO_RE2, 'ERL-Re2 w/o Re2'),
    (ABLATION_NO_REPRODUCTION, 'ERL-Re2 w/o Reproduction'),
    (ABLATION_NO_MIGRATION, 'ERL-Re2 w/o Migration'),
    (ABLATION_FULL, 'ERL-Re2 (full)'),
]

FED_ABLATION_VARIANTS = [
    (FED_ABLATION_FULL, FED_ABLATION_LABELS[FED_ABLATION_FULL]),
    (FED_ABLATION_UNIFORM_AGG, FED_ABLATION_LABELS[FED_ABLATION_UNIFORM_AGG]),
    (FED_ABLATION_NO_LOCAL_RL, FED_ABLATION_LABELS[FED_ABLATION_NO_LOCAL_RL]),
    (FED_ABLATION_NO_EA_INJECTION, FED_ABLATION_LABELS[FED_ABLATION_NO_EA_INJECTION]),
    (FED_ABLATION_NO_HETEROGENEITY, FED_ABLATION_LABELS[FED_ABLATION_NO_HETEROGENEITY]),
]

# ERL family progression
ERL_PROGRESSION_MODES = [STANDARD_ERL, ERL_RE2, DIST_ERL, FED_EVO_RL]


def uses_reproduction(mode: str, ablation: str) -> bool:
    if mode not in RE2_MODES:
        return False
    return ablation not in (ABLATION_NO_REPRODUCTION, ABLATION_NO_RE2)


def uses_migration(mode: str, ablation: str) -> bool:
    if mode not in RE2_MODES:
        return False
    return ablation not in (ABLATION_NO_MIGRATION, ABLATION_NO_RE2)


def effective_mode_label(mode: str, ablation: str) -> str:
    if mode in RE2_MODES and ablation != ABLATION_FULL:
        for key, label in RE2_ABLATION_VARIANTS:
            if key == ablation:
                return label
    if mode == FED_EVO_RL and ablation in FED_ABLATION_LABELS:
        return FED_ABLATION_LABELS[ablation]
    return MODE_LABELS.get(mode, mode)
