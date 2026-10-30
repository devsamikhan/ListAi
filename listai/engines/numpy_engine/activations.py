"""Vectorized ReLU and numerically stable Softmax activations using NumPy."""

import numpy as np


def relu(Z: np.ndarray) -> np.ndarray:
    """Compute Rectified Linear Unit activation: max(0, Z)."""
    return np.maximum(0.0, Z)


def relu_derivative(Z: np.ndarray) -> np.ndarray:
    """Compute derivative of ReLU activation: 1.0 where Z > 0, else 0.0."""
    return (Z > 0.0).astype(np.float64)


def softmax(Z: np.ndarray) -> np.ndarray:
    """Compute numerically stable Softmax row-wise across classes."""
    if Z.size == 0:
        return np.empty_like(Z)

    # Shift by row-maximum to avoid exp overflow
    shifted_Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(shifted_Z)
    sum_exp = np.sum(exp_Z, axis=1, keepdims=True)

    # Guard against division by zero
    sum_exp = np.where(sum_exp == 0.0, 1.0, sum_exp)
    return exp_Z / sum_exp
