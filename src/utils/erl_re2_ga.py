"""
ERL-Re² genetic algorithm (Algorithm 2), aligned with code/ERL-Re2/core/mod_neuro_evo.py.

Selection: elites + tournament winners + discarders.
Crossover: elite × winner replaces discarders (row-wise actor crossover).
Mutation: non-elites only; P(mut)=0.9; per action-row with prob alpha; beta column fraction;
  90% minor / 5% drastic / 5% reset Gaussian perturbations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .individual import Individual


@dataclass
class Er2GaConfig:
    num_elitists: int = 1
    tournament_size: int = 3
    mutation_prob: float = 0.9
    mutation_alpha: float = 1.0
    mutation_beta_frac: float = 0.7
    mut_strength: float = 0.1
    super_mut_strength: float = 10.0
    prob_reset_and_super: float = 0.05
    actor_prefix: str = 'actor.'


def _actor_keys(weights: Dict[str, np.ndarray], prefix: str) -> List[str]:
    return sorted(k for k in weights if k.startswith(prefix))


def clone_individual(master: Individual, replacee: Individual) -> None:
    for key, arr in master.weights.items():
        replacee.weights[key] = np.array(arr, copy=True)
    replacee.seed = int(master.seed)


def selection_tournament(index_rank: np.ndarray, num_offsprings: int,
                         tournament_size: int, rng: random.Random) -> List[int]:
    """Tournament on rank positions (0 = best), matching ERL-Re2 SSNE.selection_tournament."""
    total = len(index_rank)
    offsprings: List[int] = []
    while len(offsprings) < num_offsprings:
        draws = [rng.randrange(total) for _ in range(tournament_size)]
        winner_pos = min(draws)
        offsprings.append(int(index_rank[winner_pos]))
    offsprings = list(dict.fromkeys(offsprings))
    while len(offsprings) < num_offsprings:
        offsprings.append(offsprings[rng.randrange(len(offsprings))])
    if len(offsprings) % 2 != 0:
        offsprings.append(offsprings[rng.randrange(len(offsprings))])
    return offsprings[:num_offsprings]


def b_crossover_inplace(weights1: Dict[str, np.ndarray], weights2: Dict[str, np.ndarray],
                        prefix: str = 'actor.', rng: Optional[random.Random] = None) -> None:
    """
    Row-wise crossover on 2D actor weight matrices (ERL-Re2 crossover_inplace).
    """
    rng = rng or random.Random()
    keys = _actor_keys(weights1, prefix)
    weight_keys = [k for k in keys if weights1[k].ndim == 2 and 'weight' in k]

    for wkey in weight_keys:
        W1 = weights1[wkey]
        W2 = weights2[wkey]
        if W1.shape != W2.shape:
            continue
        bkey = wkey.replace('weight', 'bias')
        b1 = weights1.get(bkey)
        b2 = weights2.get(bkey)
        num_rows = W1.shape[0]
        num_cross = rng.randrange(num_rows * 2) if num_rows > 0 else 0
        for _ in range(num_cross):
            row = rng.randrange(num_rows)
            if rng.random() < 0.5:
                W1[row, :] = W2[row, :]
                if b1 is not None and b2 is not None and row < len(b1):
                    b1[row] = b2[row]
            else:
                W2[row, :] = W1[row, :]
                if b1 is not None and b2 is not None and row < len(b2):
                    b2[row] = b1[row]


def b_mutate_inplace(weights: Dict[str, np.ndarray], cfg: Er2GaConfig,
                     rng: Optional[random.Random] = None) -> None:
    """Behavior-level mutation on actor layers (ERL-Re2 mutate_inplace)."""
    rng = rng or random.Random()
    super_mut_prob = cfg.prob_reset_and_super
    reset_prob = min(1.0, super_mut_prob + cfg.prob_reset_and_super)
    keys = _actor_keys(weights, cfg.actor_prefix)

    for key in keys:
        W = weights[key]
        if W.ndim != 2 or 'weight' not in key:
            continue
        ssne_prob = rng.random() * 2.0
        if ssne_prob >= cfg.mutation_alpha:
            continue
        num_rows = W.shape[0]
        n_cols = max(1, int(W.shape[1] * cfg.mutation_beta_frac))
        for row in range(num_rows):
            if rng.random() >= cfg.mutation_alpha:
                continue
            col_indices = rng.sample(range(W.shape[1]), min(n_cols, W.shape[1]))
            r = rng.random()
            for col in col_indices:
                if r < super_mut_prob:
                    W[row, col] += rng.gauss(0, cfg.super_mut_strength * abs(W[row, col]) + 1e-8)
                elif r < reset_prob:
                    W[row, col] = rng.gauss(0, 1)
                else:
                    W[row, col] += rng.gauss(0, cfg.mut_strength * abs(W[row, col]) + 1e-8)
            np.clip(W[row, :], -1e6, 1e6, out=W[row, :])


def erl_re2_epoch(population: List[Individual], cfg: Er2GaConfig,
                  rng: Optional[random.Random] = None) -> Tuple[int, Dict[str, float]]:
    """
    One GA generation. Returns (best_elite_index, selection_stats).
    """
    rng = rng or random.Random()
    n = len(population)
    fitness = np.array([ind.fitness for ind in population], dtype=np.float64)
    index_rank = np.argsort(fitness)[::-1]
    num_elitists = min(cfg.num_elitists, n)
    elitist_index = list(index_rank[:num_elitists])

    offsprings = selection_tournament(
        index_rank, num_offsprings=max(0, n - num_elitists),
        tournament_size=cfg.tournament_size, rng=rng)

    unselects = [i for i in range(n) if i not in offsprings and i not in elitist_index]
    rng.shuffle(unselects)

    new_elitists: List[int] = []
    for ei in elitist_index:
        if unselects:
            replacee = unselects.pop(0)
        else:
            replacee = offsprings.pop(0)
        new_elitists.append(replacee)
        clone_individual(population[ei], population[replacee])

    if len(unselects) % 2 != 0 and unselects:
        unselects.append(unselects[rng.randrange(len(unselects))])

    for k in range(0, len(unselects), 2):
        if k + 1 >= len(unselects):
            break
        i, j = unselects[k], unselects[k + 1]
        elite_parent = population[rng.choice(elitist_index)]
        winner_parent = population[rng.choice(offsprings)]
        clone_individual(elite_parent, population[i])
        clone_individual(winner_parent, population[j])
        b_crossover_inplace(population[i].weights, population[j].weights,
                            prefix=cfg.actor_prefix, rng=rng)

    for idx in range(n):
        if idx not in new_elitists:
            if rng.random() < cfg.mutation_prob:
                b_mutate_inplace(population[idx].weights, cfg, rng=rng)
            population[idx].seed = int(rng.randint(0, 2**32 - 1))

    stats = {
        'elite': float(len(elitist_index)),
        'winners': float(len(offsprings)),
        'discarded': float(len(unselects)),
    }
    return int(elitist_index[0]), stats
