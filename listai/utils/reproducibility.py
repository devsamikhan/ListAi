"""Deterministic random number generation and reproducibility utilities.

Uses only the Python standard library (`random`, `math`) so that both Engine A
(Pure Python Lists) and Engine B (NumPy) can share identical initial weights,
dataset splits, and mini-batch shuffle orders when needed.
"""

import math
import random
from typing import List, Sequence, TypeVar

T = TypeVar("T")


def create_rng(seed: int = 42) -> random.Random:
    """Create an isolated, deterministic standard-library Random instance."""
    return random.Random(seed)


def set_global_seed(seed: int = 42) -> None:
    """Seed Python's global standard-library random generator."""
    random.seed(seed)


def shuffled_indices(n: int, rng: random.Random) -> List[int]:
    """Return a deterministically shuffled list of indices [0, 1, ..., n - 1]."""
    indices = list(range(n))
    rng.shuffle(indices)
    return indices


def he_normal_matrix(
    rows: int,
    cols: int,
    fan_in: int,
    rng: random.Random,
) -> List[List[float]]:
    """Generate a matrix of shape (rows, cols) using He normal initialization.

    Each weight is drawn from N(0, sqrt(2 / fan_in)).
    """
    if rows <= 0 or cols <= 0 or fan_in <= 0:
        raise ValueError("rows, cols, and fan_in must be positive integers.")
    std = math.sqrt(2.0 / float(fan_in))
    return [[rng.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)]
