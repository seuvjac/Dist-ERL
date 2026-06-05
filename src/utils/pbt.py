"""Population Based Training utilities for FedEvoRL."""

import numpy as np
from typing import List, Dict, Any, Optional
from .individual import Individual


class PopulationBasedTraining:
    """Population Based Training logic for evolutionary optimization"""

    def __init__(self,
                 mutation_rate: float = 0.2,
                 mutation_strength: float = 0.1,
                 exploit_interval: int = 10):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.exploit_interval = exploit_interval
        self.generation = 0

    def exploit(self, population: List[Individual]) -> List[Individual]:
        """Perform exploit by cloning top performers into weaker individuals."""
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        half = max(1, len(sorted_pop) // 2)
        top_group = sorted_pop[:half]
        bottom_group = sorted_pop[half:]

        new_population = []
        for ind in top_group:
            new_population.append(ind)

        for ind in bottom_group:
            donor = np.random.choice(top_group)
            cloned_weights = {k: np.array(v, copy=True) for k, v in donor.weights.items()}
            cloned_hyperparams = dict(donor.hyperparams) if donor.hyperparams else None
            new_population.append(
                Individual(
                    id=ind.id,
                    weights=cloned_weights,
                    fitness=0.0,
                    seed=np.random.randint(0, 2**32),
                    hyperparams=cloned_hyperparams
                )
            )

        return new_population

    def explore(self, individual: Individual) -> Individual:
        """Mutate weights and optionally hyperparameters for an individual."""
        mutated_weights = self._mutate_weights(individual.weights)
        mutated_hyperparams = self._perturb_hyperparams(individual.hyperparams)

        return Individual(
            id=individual.id,
            weights=mutated_weights,
            fitness=individual.fitness,
            seed=np.random.randint(0, 2**32),
            hyperparams=mutated_hyperparams
        )

    def exploit_and_explore(self, population: List[Individual]) -> List[Individual]:
        """Run exploit/explore operations at scheduled intervals."""
        self.generation += 1
        if self.generation % self.exploit_interval != 0:
            return population

        exploited = self.exploit(population)
        half = max(1, len(exploited) // 2)
        out = []
        for idx, ind in enumerate(exploited):
            if idx >= half:
                out.append(self.explore(ind))
            else:
                out.append(ind)
        return out

    def _mutate_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply mutation to weights"""
        mutated_weights = {}
        for key, array in weights.items():
            # Gaussian mutation
            mutation = np.random.normal(0, self.mutation_strength, array.shape)
            mutation_mask = np.random.random(array.shape) < self.mutation_rate
            mutated_weights[key] = array + mutation_mask * mutation
        return mutated_weights

    def _perturb_hyperparams(self, hyperparams: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Perturb hyperparameters such as learning rate or tau."""
        if hyperparams is None:
            return None

        perturbed = dict(hyperparams)
        if 'lr' in perturbed:
            perturbed['lr'] = float(max(1e-6, perturbed['lr'] * (1.0 + np.random.normal(0, 0.1))))
        if 'tau' in perturbed:
            perturbed['tau'] = float(max(1e-6, perturbed['tau'] * (1.0 + np.random.normal(0, 0.05))))
        return perturbed

    def should_exploit(self, current_fitness: float, population_fitnesses: List[float]) -> bool:
        """Determine if an individual should exploit based on performance"""
        if not population_fitnesses:
            return False

        # Exploit if fitness is below median
        median_fitness = np.median(population_fitnesses)
        return current_fitness < median_fitness

    def select_parent(self, population: List[Individual], child_fitness: float) -> Optional[Individual]:
        """Select a parent for exploitation"""
        # Select from individuals better than current child
        better_individuals = [ind for ind in population if ind.fitness > child_fitness]

        if not better_individuals:
            return None

        # Select proportionally to fitness difference
        fitnesses = np.array([ind.fitness for ind in better_individuals])
        min_fitness = np.min(fitnesses)
        weights = fitnesses - min_fitness + 1e-6  # Ensure positive weights
        weights = weights / np.sum(weights)

        selected_idx = np.random.choice(len(better_individuals), p=weights)
        return better_individuals[selected_idx]
