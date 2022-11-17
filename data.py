from typing import Any
import numpy as np
import os

def get_random_vectors(n=1, dim=128, from_file=True) -> np.ndarray[Any, np.dtype[np.float64]]:
    filename = f"data/{n}-{dim}"
    if from_file and os.path.exists(filename):
        with np.load(filename) as data:
            return data.f
    data = np.random.rand(n, dim)
    if from_file:
        np.save(filename, data)
    return data
