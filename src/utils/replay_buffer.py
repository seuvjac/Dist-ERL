"""Hybrid Replay Buffer for Dist-ERL."""

import numpy as np
import random
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class Transition:
    """Represents a single environment transition"""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool


@dataclass
class EASeed:
    """Represents an EA seed for later reproduction"""
    seed: int
    fitness: float
    generation: int
    individual_id: int


class HybridReplayBuffer:
    """Hybrid replay buffer storing both RL and EA experience"""

    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        self.rl_buffer = deque(maxlen=capacity // 2)  # RL data
        self.ea_seeds = deque(maxlen=capacity // 2)    # EA seeds for reproduction
        self.ea_transitions = deque(maxlen=capacity // 2)  # Reproduced EA transitions
        self.position = 0

    def add_rl_data(self, observation: np.ndarray, action: np.ndarray,
                   reward: float, next_observation: np.ndarray, done: bool):
        """
        Add RL-collected transition (complete transition data).
        This stores the full transition for immediate use in RL training.
        """
        transition = Transition(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done
        )

        self.rl_buffer.append({
            'transition': transition,
            'type': 'rl',
            'timestamp': self.position
        })
        self.position += 1

    def add_ea_seed(self, seed: int, fitness: float, generation: int, individual_id: int):
        """
        Add EA seed for later reproduction.
        This only stores the seed and metadata, not the full trajectory.
        The actual trajectory will be reproduced later by the Learner.
        """
        ea_seed = EASeed(
            seed=seed,
            fitness=fitness,
            generation=generation,
            individual_id=individual_id
        )

        self.ea_seeds.append({
            'seed_data': ea_seed,
            'type': 'ea_seed',
            'timestamp': self.position
        })
        self.position += 1

    def add_reproduced_ea_transition(self, observation: np.ndarray, action: np.ndarray,
                                   reward: float, next_observation: np.ndarray, done: bool,
                                   seed: int, generation: int):
        """
        Add a transition that was reproduced from an EA seed.
        This is called by the Learner after reproducing trajectories from elite seeds.
        """
        transition = Transition(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done
        )

        self.ea_transitions.append({
            'transition': transition,
            'seed': seed,
            'generation': generation,
            'type': 'ea_reproduced',
            'timestamp': self.position
        })
        self.position += 1

    def get_elite_seeds(self, k: int = 5) -> List[EASeed]:
        """
        Get the top-k elite seeds based on fitness for reproduction.
        """
        if not self.ea_seeds:
            return []

        # Sort seeds by fitness (descending)
        sorted_seeds = sorted(
            [item['seed_data'] for item in self.ea_seeds],
            key=lambda x: x.fitness,
            reverse=True
        )

        return sorted_seeds[:k]

    def sample(self, batch_size: int, ea_batch_ratio: float = 0.5) -> Dict[str, np.ndarray]:
        """Sample mixed batch; ea_batch_ratio = target fraction from EA transitions."""
        rl_size = len(self.rl_buffer)
        ea_size = len(self.ea_transitions)
        total_size = rl_size + ea_size

        if total_size == 0:
            raise ValueError("Buffer is empty")

        ea_batch_ratio = float(np.clip(ea_batch_ratio, 0.0, 1.0))
        if rl_size > 0 and ea_size > 0:
            ea_batch_size = int(round(batch_size * ea_batch_ratio))
            ea_batch_size = max(1, min(ea_batch_size, ea_size, batch_size - 1))
            rl_batch_size = batch_size - ea_batch_size
            rl_batch_size = max(1, min(rl_batch_size, rl_size))
        elif rl_size > 0:
            rl_batch_size = batch_size
            ea_batch_size = 0
        else:
            rl_batch_size = 0
            ea_batch_size = batch_size

        # Sample from RL buffer
        rl_transitions = []
        if rl_batch_size > 0 and rl_size > 0:
            indices = np.random.choice(rl_size, min(rl_batch_size, rl_size), replace=False)
            rl_transitions = [self.rl_buffer[i]['transition'] for i in indices]

        # Sample from EA transitions
        ea_transitions = []
        if ea_batch_size > 0 and ea_size > 0:
            indices = np.random.choice(ea_size, min(ea_batch_size, ea_size), replace=False)
            ea_transitions = [self.ea_transitions[i]['transition'] for i in indices]

        # Combine transitions
        all_transitions = rl_transitions + ea_transitions

        # Create type labels
        types = ['rl'] * len(rl_transitions) + ['ea'] * len(ea_transitions)

        # Convert to batch
        batch = {
            'observations': np.array([t.observation for t in all_transitions]),
            'actions': np.array([t.action for t in all_transitions]),
            'rewards': np.array([t.reward for t in all_transitions]),
            'next_observations': np.array([t.next_observation for t in all_transitions]),
            'dones': np.array([t.done for t in all_transitions]),
            'types': np.array(types)
        }

        return batch

    def __len__(self) -> int:
        """Total buffer size"""
        return len(self.rl_buffer) + len(self.ea_transitions)

    @property
    def rl_size(self) -> int:
        """Size of RL buffer"""
        return len(self.rl_buffer)

    @property
    def ea_seeds_size(self) -> int:
        """Size of EA seeds buffer"""
        return len(self.ea_seeds)

    @property
    def ea_transitions_size(self) -> int:
        """Size of reproduced EA transitions buffer"""
        return len(self.ea_transitions)

    def clear(self):
        """Clear all buffers"""
        self.rl_buffer.clear()
        self.ea_seeds.clear()
        self.ea_transitions.clear()
