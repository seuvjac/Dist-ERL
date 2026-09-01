"""EA Manager: ERL-Re² genetic algorithm + distributed evaluation."""

import random

import ray
import numpy as np
from typing import List, Dict, Any, Optional

from .utils.individual import Individual
from .utils.erl_re2_ga import Er2GaConfig, erl_re2_epoch

_ARCHIVE_DIAGNOSTIC_KEYS = (
    'episode_length_mean',
    'forward_return_mean',
    'survive_return_mean',
    'ctrl_return_mean',
    'x_displacement_mean',
    'x_velocity_mean',
)


@ray.remote
class EAManager:
    """EA Manager — ERL-Re² GA (elite / tournament winners / discarders) + Ray eval."""

    def __init__(self, population_size: int = 50, elite_fraction: float = 0.2,
                 num_elitists: Optional[int] = None,
                 ga_config: Optional[Dict[str, Any]] = None,
                 seed: int = 0):
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)
        self.population_size = population_size
        if num_elitists is not None:
            self.num_elitists = max(1, num_elitists)
        else:
            self.num_elitists = max(1, int(population_size * elite_fraction))
        ga_cfg = ga_config or {}
        self.ga_config = Er2GaConfig(
            num_elitists=self.num_elitists,
            tournament_size=int(ga_cfg.get('tournament_size', 3)),
            mutation_prob=float(ga_cfg.get('mutation_prob', 0.9)),
            mutation_alpha=float(ga_cfg.get('mutation_alpha', 1.0)),
            mutation_beta_frac=float(ga_cfg.get('mutation_beta_frac', 0.7)),
            mut_strength=float(ga_cfg.get('mut_strength', 0.1)),
            super_mut_strength=float(ga_cfg.get('super_mut_strength', 10.0)),
            prob_reset_and_super=float(ga_cfg.get('prob_reset_and_super', 0.05)),
            actor_prefix=str(ga_cfg.get('actor_prefix', 'actor.')),
            actor_exclude_substrings=tuple(ga_cfg.get('actor_exclude_substrings', ())),
            mutation_scale_mode=str(ga_cfg.get('mutation_scale_mode', 'element')),
            mutation_scale_floor=float(ga_cfg.get('mutation_scale_floor', 1e-3)),
            mutate_bias=bool(ga_cfg.get('mutate_bias', False)),
        )
        self.weight_clip = float(ga_cfg.get('weight_clip', 5.0))
        self.population: List[Individual] = []
        self.generation = 0
        self.best_fitness_history = []
        self._model_template: Optional[Dict[str, np.ndarray]] = None
        self._last_selection_stats: Dict[str, float] = {}
        self._elite_index: int = 0
        self._elite_archive: List[Individual] = []
        self._archive_size: int = 0

    def initialize_population(
        self,
        model_template: Dict[str, np.ndarray],
        init_mode: str = 'gaussian',
        init_seed: Optional[int] = None,
        anchor_noise_scale: float = 0.05,
    ) -> None:
        self._model_template = {k: v for k, v in model_template.items()}
        init_mode = str(init_mode).lower()
        if init_mode not in ('gaussian', 'anchor_perturb', 'anchor_antithetic'):
            raise ValueError(f'Unsupported EA init_mode={init_mode}')
        init_rng = (
            np.random.default_rng(int(init_seed))
            if init_seed is not None and int(init_seed) >= 0
            else self._np_rng
        )
        anchor_noise_scale = max(0.0, float(anchor_noise_scale))

        antithetic_noise: Dict[int, Dict[str, np.ndarray]] = {}

        def is_evolvable(key: str) -> bool:
            return (
                key.startswith(self.ga_config.actor_prefix)
                and not any(
                    token in key
                    for token in self.ga_config.actor_exclude_substrings
                )
            )

        def perturbation(key: str, base: np.ndarray, i: int) -> np.ndarray:
            layer_rms = float(np.sqrt(np.mean(np.square(base, dtype=np.float64))))
            noise_std = anchor_noise_scale * max(
                layer_rms, self.ga_config.mutation_scale_floor)
            if noise_std <= 0:
                return np.zeros_like(base)
            if init_mode != 'anchor_antithetic':
                return init_rng.normal(0.0, noise_std, base.shape).astype(base.dtype)
            pair_id = (i - 1) // 2
            pair_noise = antithetic_noise.setdefault(pair_id, {})
            if key not in pair_noise:
                pair_noise[key] = init_rng.normal(
                    0.0, noise_std, base.shape).astype(base.dtype)
            sign = 1.0 if i % 2 == 1 else -1.0
            return sign * pair_noise[key]

        def make_individual(i: int) -> Individual:
            if init_mode == 'gaussian':
                weights = {
                    key: init_rng.normal(0, 0.1, array.shape).astype(array.dtype)
                    for key, array in model_template.items()
                }
            else:
                weights = {}
                for key, array in model_template.items():
                    base = np.array(array, copy=True)
                    # Keep one exact SAC actor and perturb only parameters that
                    # participate in deterministic EA fitness. In continuous
                    # SAC, log_std can therefore remain client-local.
                    if i > 0 and is_evolvable(key):
                        base = base + perturbation(key, base, i)
                    weights[key] = base.astype(array.dtype, copy=False)
            self._clip_weights(weights)
            return Individual(
                id=i,
                weights=weights,
                seed=int(init_rng.integers(0, 2**32)),
                hyperparams={'lr': 3e-4, 'tau': 0.005},
            )

        self.population = [make_individual(i) for i in range(self.population_size)]

    def _clip_weights(self, weights: Dict[str, np.ndarray]) -> None:
        if self.weight_clip <= 0:
            return
        for key, arr in weights.items():
            if key.startswith('actor.'):
                np.clip(arr, -self.weight_clip, self.weight_clip, out=arr)

    def _clip_population_weights(self) -> None:
        for ind in self.population:
            self._clip_weights(ind.weights)

    def get_population_for_evaluation(self) -> List[Dict[str, Any]]:
        return [
            {'id': ind.id, 'weights': ind.weights, 'seed': ind.seed}
            for ind in self.population
        ]

    def evaluate_population(self, workers: List[Any]) -> List[Dict[str, Any]]:
        if not workers:
            raise ValueError("At least one worker is required to evaluate the population")

        worker_count = len(workers)
        evaluation_refs = []
        for idx, individual in enumerate(self.population):
            worker = workers[idx % worker_count]
            evaluation_refs.append(worker.evaluate.remote(individual))

        fitness_values = ray.get(evaluation_refs)
        results = [
            {'id': individual.id, 'fitness': float(fitness), 'seed': individual.seed}
            for individual, fitness in zip(self.population, fitness_values)
        ]
        self.update_fitness(results)
        return results

    def update_fitness(self, results: List[Dict[str, Any]]) -> None:
        for result in results:
            ind_id = result['id']
            fitness = result['fitness']
            for ind in self.population:
                if ind.id == ind_id:
                    ind.fitness = fitness
                    break
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_fitness_history.append(self.population[0].fitness)

    def evolve_population(self) -> None:
        """ERL-Re² GA epoch (Algorithm 2 / mod_neuro_evo.SSNE.epoch)."""
        cfg = Er2GaConfig(
            num_elitists=self.num_elitists,
            tournament_size=self.ga_config.tournament_size,
            mutation_prob=self.ga_config.mutation_prob,
            mutation_alpha=self.ga_config.mutation_alpha,
            mutation_beta_frac=self.ga_config.mutation_beta_frac,
            mut_strength=self.ga_config.mut_strength,
            super_mut_strength=self.ga_config.super_mut_strength,
            prob_reset_and_super=self.ga_config.prob_reset_and_super,
            actor_prefix=self.ga_config.actor_prefix,
            actor_exclude_substrings=self.ga_config.actor_exclude_substrings,
            mutation_scale_mode=self.ga_config.mutation_scale_mode,
            mutation_scale_floor=self.ga_config.mutation_scale_floor,
            mutate_bias=self.ga_config.mutate_bias,
        )
        elite_idx, sel_stats = erl_re2_epoch(self.population, cfg, rng=self._rng)
        self._clip_population_weights()
        self._elite_index = elite_idx
        self._last_selection_stats = sel_stats
        self.generation += 1

    def update_elite_archive(self, archive_size: int = 5) -> Dict[str, float]:
        """Keep a global top-k archive independent of current-generation drift."""
        self._archive_size = max(0, int(archive_size))
        if self._archive_size <= 0 or not self.population:
            self._elite_archive = []
            return {'archive_size': 0, 'archive_best': 0.0}
        candidates = [ind.copy() for ind in self._elite_archive]
        candidates.extend(ind.copy() for ind in self.population)
        candidates.sort(key=lambda x: x.fitness, reverse=True)
        unique = []
        seen = set()
        for ind in candidates:
            key = (int(ind.seed), round(float(ind.fitness), 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append(ind)
            if len(unique) >= self._archive_size:
                break
        self._elite_archive = unique
        best = self._elite_archive[0].fitness if self._elite_archive else 0.0
        return {'archive_size': len(self._elite_archive), 'archive_best': float(best)}

    def update_elite_archive_evaluated(
        self,
        evaluated: List[Dict[str, Any]],
        archive_size: int = 5,
        std_penalty: float = 0.0,
    ) -> Dict[str, float]:
        """Update the archive only with independently validated candidates."""
        self._archive_size = max(0, int(archive_size))
        if self._archive_size <= 0:
            self._elite_archive = []
            return {'archive_size': 0, 'archive_best': 0.0, 'archive_best_std': 0.0}

        std_penalty = max(0.0, float(std_penalty))
        candidates = [ind.copy() for ind in self._elite_archive]
        for row in evaluated:
            fitness = float(row['fitness'])
            fitness_std = float(row.get('fitness_std', 0.0))
            archive_score = fitness - std_penalty * fitness_std
            archive_hyperparams = {
                'archive_eval_std': fitness_std,
                'archive_eval_score': archive_score,
            }
            archive_hyperparams.update({
                key: float(row[key])
                for key in _ARCHIVE_DIAGNOSTIC_KEYS
                if key in row and np.isfinite(float(row[key]))
            })
            candidates.append(Individual(
                id=int(row['id']),
                weights={key: np.array(value, copy=True) for key, value in row['weights'].items()},
                fitness=fitness,
                seed=int(row.get('seed', 0)),
                hyperparams=archive_hyperparams,
            ))
        for ind in candidates:
            params = ind.hyperparams or {}
            if 'archive_eval_score' not in params:
                params['archive_eval_std'] = float(params.get('archive_eval_std', 0.0))
                params['archive_eval_score'] = float(ind.fitness) - std_penalty * params['archive_eval_std']
                ind.hyperparams = params
        candidates.sort(
            key=lambda x: float((x.hyperparams or {}).get('archive_eval_score', x.fitness)),
            reverse=True,
        )

        unique = []
        seen = set()
        for ind in candidates:
            signature = tuple(
                (key, arr.shape, round(float(np.mean(arr)), 7), round(float(np.std(arr)), 7))
                for key, arr in sorted(ind.weights.items())
            )
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(ind)
            if len(unique) >= self._archive_size:
                break
        self._elite_archive = unique
        return self.get_archive_stats()

    def restore_elite_archive(self, copies: int = 1) -> int:
        """Pin archived elites back into the active population after GA/migration."""
        if not self._elite_archive or not self.population or int(copies) <= 0:
            return 0
        copies = min(int(copies), len(self._elite_archive), len(self.population))
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        restored = 0
        for idx in range(copies):
            archived = self._elite_archive[idx].copy()
            archived.id = self.population[idx].id
            self.population[idx] = archived
            restored += 1
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return restored

    def get_archive_stats(self) -> Dict[str, float]:
        if not self._elite_archive:
            return {'archive_size': 0, 'archive_best': 0.0, 'archive_best_std': 0.0}
        best = self._elite_archive[0]
        params = best.hyperparams or {}
        stats = {
            'archive_size': len(self._elite_archive),
            'archive_best': float(best.fitness),
            'archive_best_std': float(params.get('archive_eval_std', 0.0)),
            'archive_best_score': float(params.get('archive_eval_score', best.fitness)),
        }
        stats.update({
            f'archive_best_{key}': float(params.get(key, 0.0))
            for key in _ARCHIVE_DIAGNOSTIC_KEYS
        })
        return stats

    def boost_diversity(self, immigrant_fraction: float = 0.15,
                        mutation_rate: float = 0.25, mutation_strength: float = 0.15) -> int:
        if self._model_template is None:
            return 0
        from .utils.erl_re2_ga import b_mutate_inplace

        n_imm = max(1, int(self.population_size * immigrant_fraction))
        cfg = Er2GaConfig(
            num_elitists=self.num_elitists,
            mutation_alpha=1.0,
            mutation_beta_frac=mutation_rate,
            mut_strength=mutation_strength,
            prob_reset_and_super=min(0.02, self.ga_config.prob_reset_and_super),
            actor_prefix=self.ga_config.actor_prefix,
            actor_exclude_substrings=self.ga_config.actor_exclude_substrings,
            mutation_scale_mode=self.ga_config.mutation_scale_mode,
            mutation_scale_floor=self.ga_config.mutation_scale_floor,
            mutate_bias=self.ga_config.mutate_bias,
        )
        replaced = 0
        sources = self._elite_archive or self.population[:max(1, self.num_elitists)]
        for offset, ind in enumerate(self.population[-n_imm:]):
            source = sources[offset % len(sources)]
            ind.weights = {
                key: np.array(array, copy=True)
                for key, array in source.weights.items()
            }
            b_mutate_inplace(ind.weights, cfg, rng=self._rng)
            self._clip_weights(ind.weights)
            ind.seed = int(self._np_rng.integers(0, 2**32))
            ind.fitness = 0.0
            replaced += 1
        for ind in self.population[self.num_elitists:-n_imm]:
            b_mutate_inplace(ind.weights, cfg, rng=self._rng)
            self._clip_weights(ind.weights)
            ind.seed = int(self._np_rng.integers(0, 2**32))
        return replaced

    def inject_rl_individual(self, rl_weights: Dict[str, np.ndarray],
                            inject_noise: float = 0.02,
                            copies: int = 1,
                            blend: float = 1.0) -> int:
        """
        Federated RL->EA migration.

        The previous hard replacement of a single weakest individual tended to
        collapse search around the current RL actor.  Softly blending several
        weak non-elite individuals keeps useful RL structure while preserving
        client/population diversity.
        """
        if not self.population or int(copies) <= 0:
            return 0

        fitness = np.array([ind.fitness for ind in self.population], dtype=np.float64)
        weakest_indices = np.argsort(fitness)
        copies = max(1, min(int(copies), len(self.population) - self.num_elitists))
        blend = float(np.clip(blend, 0.0, 1.0))
        inserted = 0

        for idx in weakest_indices:
            idx = int(idx)
            if idx < self.num_elitists:
                continue
            ind = self.population[idx]
            scale = inject_noise * (1.0 + inserted)
            for key, arr in rl_weights.items():
                if key not in ind.weights:
                    continue
                is_evolvable = (
                    key.startswith(self.ga_config.actor_prefix)
                    and not any(
                        token in key
                        for token in self.ga_config.actor_exclude_substrings
                    )
                )
                if not is_evolvable:
                    continue
                target = np.array(arr, copy=True)
                current = ind.weights[key]
                if current.shape != target.shape:
                    continue
                mixed = (1.0 - blend) * current + blend * target
                if scale > 0:
                    mixed = mixed + self._np_rng.normal(0, scale, mixed.shape).astype(mixed.dtype)
                    if self.weight_clip > 0:
                        mixed = np.clip(mixed, -self.weight_clip, self.weight_clip)
                ind.weights[key] = mixed.astype(current.dtype, copy=False)
            ind.seed = int(self._np_rng.integers(0, 2**32))
            ind.fitness = float(np.median(fitness))
            inserted += 1
            if inserted >= copies:
                break
        return inserted

    def get_best_individual(self) -> Individual:
        return max(self.population, key=lambda x: x.fitness)

    def get_archive_best_individual(self) -> Individual:
        """Return the independently validated deployable actor."""
        if self._elite_archive:
            return self._elite_archive[0].copy()
        return max(self.population, key=lambda x: x.fitness).copy()

    def get_elite_seeds(self, k: int = 5) -> List[int]:
        elites = self.population[:k]
        return [ind.seed for ind in elites]

    def get_elite_individuals(self, k: int = 5) -> List[Dict[str, Any]]:
        elites = self.population[:k]
        return [
            {
                'id': ind.id,
                'seed': int(ind.seed),
                'fitness': float(ind.fitness),
                'weights': {key: arr.copy() for key, arr in ind.weights.items()},
            }
            for ind in elites
        ]

    def _weight_diversity(self) -> float:
        if len(self.population) < 2:
            return 0.0
        flat = []
        for ind in self.population:
            parts = [
                ind.weights[k].ravel()
                for k in sorted(ind.weights.keys())
                if k.startswith('actor.')
                and not any(
                    token in k
                    for token in self.ga_config.actor_exclude_substrings
                )
            ]
            if not parts:
                continue
            flat.append(np.concatenate(parts))
        if len(flat) < 2:
            return 0.0
        vectors = np.stack(flat)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        normed = vectors / norms
        sim = normed @ normed.T
        n = len(self.population)
        off_diag = sim[np.triu_indices(n, k=1)]
        return float(1.0 - np.mean(off_diag))

    def get_median_fitness(self) -> float:
        fitnesses = [ind.fitness for ind in self.population]
        return float(np.median(fitnesses)) if fitnesses else 0.0

    def get_stats(self) -> Dict[str, Any]:
        fitnesses = [ind.fitness for ind in self.population]
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'mean_fitness': float(np.mean(fitnesses)),
            'median_fitness': float(np.median(fitnesses)) if fitnesses else 0.0,
            'max_fitness': float(np.max(fitnesses)),
            'min_fitness': float(np.min(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'weight_diversity': self._weight_diversity(),
            'best_fitness_history': self.best_fitness_history,
            'num_elitists': self.num_elitists,
            'ea_elite': self._last_selection_stats.get('elite', 0),
            'ea_winners': self._last_selection_stats.get('winners', 0),
            'ea_discarded': self._last_selection_stats.get('discarded', 0),
            **self.get_archive_stats(),
        }
