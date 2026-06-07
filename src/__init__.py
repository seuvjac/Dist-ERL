"""FedEvoRL: Evolutionary Federated Reinforcement Learning Framework."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .utils.individual import Individual

__all__ = ["EAManager", "RolloutWorker", "RLLearner", "Individual"]


def __getattr__(name):
    if name == "EAManager":
        from .manager import EAManager
        return EAManager
    if name == "RolloutWorker":
        from .worker import RolloutWorker
        return RolloutWorker
    if name == "RLLearner":
        from .learner import RLLearner
        return RLLearner
    raise AttributeError(name)
