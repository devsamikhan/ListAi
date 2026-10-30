"""Pure Python List implementation of explicit backpropagation and gradient checking.

STRICT CONSTRAINT:
Only Python standard library is permitted. Zero external numerical libraries.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

from listai.engines.list_engine.activations import relu_derivative
from listai.engines.list_engine.loss import (
    categorical_cross_entropy,
    softmax_cross_entropy_gradient,
)
from listai.engines.list_engine.network import ForwardCache, ListNeuralNetwork
from listai.engines.list_engine.operations import (
    matrix_elementwise_mul,
    matrix_mul,
    matrix_scalar_mul,
    matrix_sum_columns,
    transpose,
    vector_scalar_mul,
)


@dataclass
class Gradients:
    """Container for analytical parameter gradients."""

    dW: List[List[List[float]]]  # dW[0], dW[1], dW[2]
    db: List[List[float]]        # db[0], db[1], db[2]


def backward_pass(
    net: ListNeuralNetwork,
    cache: ForwardCache,
    Y_true: Sequence[Sequence[float]],
) -> Gradients:
    """Compute exact analytical gradients via explicit backpropagation.

    Layer 3 (Output):
        dZ3 = A3 - Y
        dW3 = (1 / N) * A2^T @ dZ3
        db3 = (1 / N) * sum_rows(dZ3)

    Layer 2:
        dA2 = dZ3 @ W3^T
        dZ2 = dA2 * relu_derivative(Z2)
        dW2 = (1 / N) * A1^T @ dZ2
        db2 = (1 / N) * sum_rows(dZ2)

    Layer 1:
        dA1 = dZ2 @ W2^T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = (1 / N) * A0^T @ dZ1
        db1 = (1 / N) * sum_rows(dZ1)
    """
    n_samples = len(Y_true)
    if n_samples == 0:
        raise ValueError("Cannot perform backward pass with empty batch.")

    scale = 1.0 / float(n_samples)

    # Layer 3
    dZ3 = softmax_cross_entropy_gradient(cache.A3, Y_true, normalize=False)
    dW3 = matrix_scalar_mul(matrix_mul(transpose(cache.A2), dZ3), scale)
    db3 = vector_scalar_mul(matrix_sum_columns(dZ3), scale)

    # Layer 2
    dA2 = matrix_mul(dZ3, transpose(net.weights[2]))
    dZ2 = matrix_elementwise_mul(dA2, relu_derivative(cache.Z2))
    dW2 = matrix_scalar_mul(matrix_mul(transpose(cache.A1), dZ2), scale)
    db2 = vector_scalar_mul(matrix_sum_columns(dZ2), scale)

    # Layer 1
    dA1 = matrix_mul(dZ2, transpose(net.weights[1]))
    dZ1 = matrix_elementwise_mul(dA1, relu_derivative(cache.Z1))
    dW1 = matrix_scalar_mul(matrix_mul(transpose(cache.A0), dZ1), scale)
    db1 = vector_scalar_mul(matrix_sum_columns(dZ1), scale)

    return Gradients(
        dW=[dW1, dW2, dW3],
        db=[db1, db2, db3],
    )


def update_parameters(net: ListNeuralNetwork, grads: Gradients, lr: float) -> None:
    """Update network weights and biases in-place via Gradient Descent: theta = theta - lr * d_theta."""
    for l in range(net.num_layers):
        # Update weights: W[l][r][c] -= lr * dW[l][r][c]
        W = net.weights[l]
        dW = grads.dW[l]
        for r in range(len(W)):
            for c in range(len(W[r])):
                W[r][c] -= lr * dW[r][c]

        # Update biases: b[l][c] -= lr * db[l][c]
        b = net.biases[l]
        db = grads.db[l]
        for c in range(len(b)):
            b[c] -= lr * db[c]


def compute_loss_and_gradients(
    net: ListNeuralNetwork,
    X: Sequence[Sequence[float]],
    Y_true: Sequence[Sequence[float]],
) -> Tuple[float, Gradients]:
    """Execute forward pass, loss calculation, and backward pass."""
    A3, cache = net.forward(X)
    loss = categorical_cross_entropy(Y_true, A3)
    grads = backward_pass(net, cache, Y_true)
    return loss, grads


def numerical_gradient_check(
    net: ListNeuralNetwork,
    X: Sequence[Sequence[float]],
    Y_true: Sequence[Sequence[float]],
    epsilon: float = 1e-5,
) -> Dict[str, float]:
    """Perform two-sided finite-difference gradient checking.

    Formula:
        g_num = (Loss(theta + eps) - Loss(theta - eps)) / (2 * eps)
        rel_error = ||g_ana - g_num||_2 / max(||g_ana||_2 + ||g_num||_2, 1e-12)
    """
    _, analytical_grads = compute_loss_and_gradients(net, X, Y_true)

    ana_flat: List[float] = []
    num_flat: List[float] = []

    # Helper to evaluate loss for a given parameter perturbation
    def eval_loss() -> float:
        preds, _ = net.forward(X)
        return categorical_cross_entropy(Y_true, preds)

    # 1. Weights
    for l in range(net.num_layers):
        W = net.weights[l]
        dW = analytical_grads.dW[l]
        for r in range(len(W)):
            for c in range(len(W[r])):
                original_val = W[r][c]

                # theta + eps
                W[r][c] = original_val + epsilon
                loss_plus = eval_loss()

                # theta - eps
                W[r][c] = original_val - epsilon
                loss_minus = eval_loss()

                # Reset
                W[r][c] = original_val

                g_num = (loss_plus - loss_minus) / (2.0 * epsilon)
                ana_flat.append(dW[r][c])
                num_flat.append(g_num)

    # 2. Biases
    for l in range(net.num_layers):
        b = net.biases[l]
        db = analytical_grads.db[l]
        for c in range(len(b)):
            original_val = b[c]

            # theta + eps
            b[c] = original_val + epsilon
            loss_plus = eval_loss()

            # theta - eps
            b[c] = original_val - epsilon
            loss_minus = eval_loss()

            # Reset
            b[c] = original_val

            g_num = (loss_plus - loss_minus) / (2.0 * epsilon)
            ana_flat.append(db[c])
            num_flat.append(g_num)

    # Compute Euclidean norms
    diff_norm_sq = sum((a - n) ** 2 for a, n in zip(ana_flat, num_flat))
    ana_norm_sq = sum(a ** 2 for a in ana_flat)
    num_norm_sq = sum(n ** 2 for n in num_flat)

    diff_norm = math.sqrt(diff_norm_sq)
    ana_norm = math.sqrt(ana_norm_sq)
    num_norm = math.sqrt(num_norm_sq)

    denom = max(ana_norm + num_norm, 1e-12)
    relative_error = diff_norm / denom

    max_abs_diff = max(abs(a - n) for a, n in zip(ana_flat, num_flat))

    return {
        "relative_error": relative_error,
        "max_abs_difference": max_abs_diff,
        "analytical_norm": ana_norm,
        "numerical_norm": num_norm,
        "total_parameters": float(len(ana_flat)),
    }
