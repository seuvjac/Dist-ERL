"""Dist-ERL: Distributed Evolutionary Reinforcement Learning Framework."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .manager import EAManager
from .worker import RolloutWorker
from .learner import RLLearner
from .utils.individual import Individual

__all__ = ["EAManager", "RolloutWorker", "RLLearner", "Individual"]
