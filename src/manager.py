"""EA Manager: ERL-Re² genetic algorithm + distributed evaluation."""

import ray
import numpy as np
from typing import List, Dict, Any, Optional

from .utils.individual import Individual
from .utils.erl_re2_ga import Er2GaConfig, erl_re2_epoch


@ray.remote
class EAManager:
    """EA Manager — ERL-Re² GA (elite / tournament winners / discarders) + Ray eval."""

    def __init__(self, population_size: int = 50, elite_fraction: float = 0.2,
                 num_elitists: Optional[int] = None,
                 ga_config: Optional[Dict[str, Any]] = None):
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
        )
        self.population: List[Individual] = []
        self.generation = 0
        self.best_fitness_history = []
        self._model_template: Optional[Dict[str, np.ndarray]] = None
        self._last_selection_stats: Dict[str, float] = {}
        self._elite_index: int = 0

    def initialize_population(self, model_template: Dict[str, np.ndarray]) -> None:
        self._model_template = {k: v for k, v in model_template.items()}

        def make_individual(i: int) -> Individual:
            weights = {
                key: np.random.normal(0, 0.1, array.shape).astype(array.dtype)
                for key, array in model_template.items()
            }
            return Individual(
                id=i,
                weights=weights,
                seed=np.random.randint(0, 2**32),
                hyperparams={'lr': 3e-4, 'tau': 0.005},
            )

        self.population = [make_individual(i) for i in range(self.population_size)]

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
        )
        elite_idx, sel_stats = erl_re2_epoch(self.population, cfg)
        self._elite_index = elite_idx
        self._last_selection_stats = sel_stats
        self.generation += 1

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
            prob_reset_and_super=0.1,
        )
        replaced = 0
        for ind in self.population[-n_imm:]:
            ind.weights = {
                key: np.random.normal(0, 0.15, array.shape).astype(array.dtype)
                for key, array in self._model_template.items()
            }
            ind.seed = int(np.random.randint(0, 2**32))
            ind.fitness = 0.0
            replaced += 1
        for ind in self.population[self.num_elitists:-n_imm]:
            b_mutate_inplace(ind.weights, cfg)
            ind.seed = int(np.random.randint(0, 2**32))
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
                target = np.array(arr, copy=True)
                current = ind.weights[key]
                if current.shape != target.shape:
                    continue
                mixed = (1.0 - blend) * current + blend * target
                if scale > 0 and key.startswith('actor.'):
                    mixed = mixed + np.random.normal(0, scale, mixed.shape).astype(mixed.dtype)
                ind.weights[key] = mixed.astype(current.dtype, copy=False)
            ind.seed = int(np.random.randint(0, 2**32))
            ind.fitness = float(np.median(fitness))
            inserted += 1
            if inserted >= copies:
                break
        return inserted

    def get_best_individual(self) -> Individual:
        return max(self.population, key=lambda x: x.fitness)

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
            parts = [ind.weights[k].ravel() for k in sorted(ind.weights.keys()) if k.startswith('actor.')]
            flat.append(np.concatenate(parts))
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
        }
