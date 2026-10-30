"""Vectorized backpropagation and finite-difference gradient checking using NumPy."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from listai.engines.numpy_engine.activations import relu_derivative
from listai.engines.numpy_engine.loss import (
    categorical_cross_entropy,
    softmax_cross_entropy_gradient,
)
from listai.engines.numpy_engine.network import NumpyForwardCache, NumpyNeuralNetwork


@dataclass
class NumpyGradients:
    """Stores analytical gradients computed via NumPy."""

    dW: List[np.ndarray]
    db: List[np.ndarray]


def backward_pass(
    net: NumpyNeuralNetwork,
    cache: NumpyForwardCache,
    Y_true: np.ndarray,
) -> NumpyGradients:
    """Compute analytical gradients via vectorized backpropagation."""
    n_samples = Y_true.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot perform backward pass on empty batch.")

    scale = 1.0 / float(n_samples)

    # Layer 3
    dZ3 = softmax_cross_entropy_gradient(cache.A3, Y_true, normalize=False)
    dW3 = scale * (cache.A2.T @ dZ3)
    db3 = scale * np.sum(dZ3, axis=0)

    # Layer 2
    dA2 = dZ3 @ net.weights[2].T
    dZ2 = dA2 * relu_derivative(cache.Z2)
    dW2 = scale * (cache.A1.T @ dZ2)
    db2 = scale * np.sum(dZ2, axis=0)

    # Layer 1
    dA1 = dZ2 @ net.weights[1].T
    dZ1 = dA1 * relu_derivative(cache.Z1)
    dW1 = scale * (cache.A0.T @ dZ1)
    db1 = scale * np.sum(dZ1, axis=0)

    return NumpyGradients(
        dW=[dW1, dW2, dW3],
        db=[db1, db2, db3],
    )


def update_parameters(net: NumpyNeuralNetwork, grads: NumpyGradients, lr: float) -> None:
    """Update network parameters in-place via Gradient Descent."""
    for l in range(net.num_layers):
        net.weights[l] -= lr * grads.dW[l]
        net.biases[l] -= lr * grads.db[l]


def compute_loss_and_gradients(
    net: NumpyNeuralNetwork,
    X: np.ndarray,
    Y_true: np.ndarray,
) -> Tuple[float, NumpyGradients]:
    """Execute forward pass, cross-entropy loss, and backward pass."""
    A3, cache = net.forward(X)
    loss = categorical_cross_entropy(Y_true, A3)
    grads = backward_pass(net, cache, Y_true)
    return loss, grads


def numerical_gradient_check(
    net: NumpyNeuralNetwork,
    X: np.ndarray,
    Y_true: np.ndarray,
    epsilon: float = 1e-5,
) -> Dict[str, float]:
    """Perform finite-difference gradient check on NumPy engine."""
    _, analytical_grads = compute_loss_and_gradients(net, X, Y_true)

    ana_flat: List[float] = []
    num_flat: List[float] = []

    def eval_loss() -> float:
        preds, _ = net.forward(X)
        return categorical_cross_entropy(Y_true, preds)

    # Weights
    for l in range(net.num_layers):
        W = net.weights[l]
        dW = analytical_grads.dW[l]
        it = np.nditer(W, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = W[idx]

            W[idx] = orig + epsilon
            loss_plus = eval_loss()

            W[idx] = orig - epsilon
            loss_minus = eval_loss()

            W[idx] = orig

            g_num = (loss_plus - loss_minus) / (2.0 * epsilon)
            ana_flat.append(float(dW[idx]))
            num_flat.append(g_num)
            it.iternext()

    # Biases
    for l in range(net.num_layers):
        b = net.biases[l]
        db = analytical_grads.db[l]
        for idx in range(len(b)):
            orig = b[idx]

            b[idx] = orig + epsilon
            loss_plus = eval_loss()

            b[idx] = orig - epsilon
            loss_minus = eval_loss()

            b[idx] = orig

            g_num = (loss_plus - loss_minus) / (2.0 * epsilon)
            ana_flat.append(float(db[idx]))
            num_flat.append(g_num)

    ana_arr = np.array(ana_flat, dtype=np.float64)
    num_arr = np.array(num_flat, dtype=np.float64)

    diff_norm = float(np.linalg.norm(ana_arr - num_arr))
    denom = max(float(np.linalg.norm(ana_arr) + np.linalg.norm(num_arr)), 1e-12)
    rel_error = diff_norm / denom
    max_diff = float(np.max(np.abs(ana_arr - num_arr)))

    return {
        "relative_error": rel_error,
        "max_abs_difference": max_diff,
        "analytical_norm": float(np.linalg.norm(ana_arr)),
        "numerical_norm": float(np.linalg.norm(num_arr)),
        "total_parameters": float(len(ana_flat)),
    }
