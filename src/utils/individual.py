"""Data structures for FedEvoRL."""

from dataclasses import dataclass
import copy
import numpy as np
from typing import Dict, Any, Optional
import pickle


@dataclass
class Individual:
    """Represents an individual in the population"""
    id: int
    weights: Dict[str, np.ndarray]
    fitness: float = 0.0
    seed: int = 0
    hyperparams: Optional[Dict[str, Any]] = None

    def export_to_ray(self) -> Dict[str, Any]:
        """
        Efficient serialization for Ray Actor communication.
        Converts numpy arrays to bytes for faster serialization.
        """
        serialized_weights = {}
        for key, array in self.weights.items():
            # Convert numpy array to bytes for efficient Ray serialization
            serialized_weights[key] = array.tobytes()

        return {
            'id': self.id,
            'weights': serialized_weights,
            'weight_shapes': {k: v.shape for k, v in self.weights.items()},
            'weight_dtypes': {k: v.dtype.str for k, v in self.weights.items()},
            'fitness': self.fitness,
            'seed': self.seed
        }

    @classmethod
    def from_ray_export(cls, data: Dict[str, Any]) -> 'Individual':
        """
        Deserialize from Ray export format.
        """
        # Reconstruct weights from bytes
        weights = {}
        for key, bytes_data in data['weights'].items():
            shape = data['weight_shapes'][key]
            dtype = np.dtype(data['weight_dtypes'][key])
            weights[key] = np.frombuffer(bytes_data, dtype=dtype).reshape(shape)

        return cls(
            id=data['id'],
            weights=weights,
            fitness=data['fitness'],
            seed=data['seed']
        )

    def copy(self) -> 'Individual':
        """Create a deep copy of the individual"""
        new_weights = {}
        for key, array in self.weights.items():
            new_weights[key] = array.copy()

        return Individual(
            id=self.id,
            weights=new_weights,
            fitness=self.fitness,
            seed=self.seed,
            hyperparams=copy.deepcopy(self.hyperparams),
        )
