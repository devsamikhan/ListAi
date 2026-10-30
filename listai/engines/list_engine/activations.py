"""Pure Python List implementation of activation functions and their derivatives.

STRICT CONSTRAINT:
Only Python standard library (e.g. `math`) is permitted.
"""

import math
from typing import List, Sequence


def relu(Z: Sequence[Sequence[float]]) -> List[List[float]]:
    """Compute the Rectified Linear Unit activation element-wise: max(0, z)."""
    return [[val if val > 0.0 else 0.0 for val in row] for row in Z]


def relu_derivative(Z: Sequence[Sequence[float]]) -> List[List[float]]:
    """Compute the derivative of ReLU element-wise: 1.0 if z > 0 else 0.0."""
    return [[1.0 if val > 0.0 else 0.0 for val in row] for row in Z]


def softmax(Z: Sequence[Sequence[float]]) -> List[List[float]]:
    """Compute numerically stable Softmax row-wise across classes.

    Formula:
        shifted_z_ij = z_ij - max_k(z_ik)
        softmax_ij = exp(shifted_z_ij) / sum_k(exp(shifted_z_ik))
    Subtracting max(z) per row guarantees exp() never overflows.
    """
    probabilities: List[List[float]] = []

    for row in Z:
        if not row:
            probabilities.append([])
            continue

        max_val = max(row)
        exp_row = [math.exp(val - max_val) for val in row]
        sum_exp = sum(exp_row)

        if sum_exp == 0.0:
            # Degenerate case guard: uniform distribution
            uniform = 1.0 / len(row)
            probabilities.append([uniform] * len(row))
        else:
            probabilities.append([e / sum_exp for e in exp_row])

    return probabilities
