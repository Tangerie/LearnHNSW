import numpy as np
from typing import Callable

DistanceFunctionType = Callable[[np.ndarray, np.ndarray], np.floating]


# Euclidean distance 
def l2_distance(a: np.ndarray, b : np.ndarray):
    return np.linalg.norm(a - b)