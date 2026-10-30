"""Pure Python List implementation of categorical cross-entropy loss and gradients."""

import math
from typing import List, Sequence


def categorical_cross_entropy(
    Y_true: Sequence[Sequence[float]],
    Y_pred: Sequence[Sequence[float]],
    epsilon: float = 1e-15,
) -> float:
    """Compute mean categorical cross-entropy loss across N samples and C classes.

    Formula:
        Loss = - (1 / N) * sum_i(sum_c(Y_true[i][c] * ln(clip(Y_pred[i][c], eps, 1 - eps))))
    """
    if len(Y_true) != len(Y_pred):
        raise ValueError(f"Shape mismatch between targets ({len(Y_true)}) and predictions ({len(Y_pred)})")

    n_samples = len(Y_true)
    if n_samples == 0:
        return 0.0

    total_loss = 0.0
    for true_row, pred_row in zip(Y_true, Y_pred):
        if len(true_row) != len(pred_row):
            raise ValueError("Row dimension mismatch between targets and predictions.")
        for y_t, y_p in zip(true_row, pred_row):
            clipped_p = max(epsilon, min(1.0 - epsilon, y_p))
            total_loss -= y_t * math.log(clipped_p)

    return total_loss / n_samples


def softmax_cross_entropy_gradient(
    Y_pred: Sequence[Sequence[float]],
    Y_true: Sequence[Sequence[float]],
    normalize: bool = False,
) -> List[List[float]]:
    """Compute combined analytical gradient of categorical cross-entropy with Softmax.

    dZ^[L] = Y_pred - Y_true  (or (Y_pred - Y_true) / N if normalize=True)
    """
    if len(Y_pred) != len(Y_true):
        raise ValueError("Batch size mismatch between predictions and targets.")

    n_samples = len(Y_pred)
    scale = (1.0 / n_samples) if (normalize and n_samples > 0) else 1.0

    gradients: List[List[float]] = []
    for pred_row, true_row in zip(Y_pred, Y_true):
        if len(pred_row) != len(true_row):
            raise ValueError("Feature count mismatch between predictions and targets.")
        grad_row = [(p - t) * scale for p, t in zip(pred_row, true_row)]
        gradients.append(grad_row)

    return gradients
