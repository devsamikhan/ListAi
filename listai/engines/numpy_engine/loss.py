"""Vectorized categorical cross-entropy loss and combined gradients using NumPy."""

import numpy as np


def categorical_cross_entropy(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    epsilon: float = 1e-15,
) -> float:
    """Compute mean categorical cross-entropy loss across batch."""
    if Y_true.shape != Y_pred.shape:
        raise ValueError(f"Shape mismatch: {Y_true.shape} vs {Y_pred.shape}")

    n_samples = Y_true.shape[0]
    if n_samples == 0:
        return 0.0

    clipped_pred = np.clip(Y_pred, epsilon, 1.0 - epsilon)
    sample_losses = -np.sum(Y_true * np.log(clipped_pred), axis=1)
    return float(np.mean(sample_losses))


def softmax_cross_entropy_gradient(
    Y_pred: np.ndarray,
    Y_true: np.ndarray,
    normalize: bool = False,
) -> np.ndarray:
    """Compute combined analytical gradient dZ^[L] = Y_pred - Y_true."""
    if Y_pred.shape != Y_true.shape:
        raise ValueError(f"Shape mismatch: {Y_pred.shape} vs {Y_true.shape}")

    diff = Y_pred - Y_true
    if normalize and len(Y_pred) > 0:
        return diff / float(len(Y_pred))
    return diff
