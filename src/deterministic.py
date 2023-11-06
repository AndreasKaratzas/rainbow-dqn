
import torch
import random
import numpy as np


def set_seed(seed: int = 0):
    """Seed the pseudorandom number generator for the CPU, Cuda, numpy and Python.
    
    Parameters
    ----------
    seed : int, optional
        the given seed, by default 0
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_deterministic(seed: int = 0):
    """Try to configure the system for reproducible results.
    Experiment reproducibility is sometimes important. Pete Warden expounded about this
    in his blog: https://petewarden.com/2018/03/19/the-machine-learning-reproducibility-crisis/
    For Pytorch specifics see: https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    
    Parameters
    ----------
    seed : int, optional
        the given seed, by default 0
    """
    if seed is None:
        seed = 0

    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
