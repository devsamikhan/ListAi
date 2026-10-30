"""Unit tests for categorical cross-entropy loss and Softmax combined gradients."""

import math
import unittest

from listai.engines.list_engine.loss import (
    categorical_cross_entropy,
    softmax_cross_entropy_gradient,
)


class TestLoss(unittest.TestCase):
    """Test categorical cross entropy and gradient calculations."""

    def test_categorical_cross_entropy_perfect_prediction(self) -> None:
        """When predictions match targets, cross-entropy is near 0."""
        Y_true = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        Y_pred = [
            [0.99999, 0.000005, 0.000005],
            [0.000005, 0.99999, 0.000005],
        ]
        loss = categorical_cross_entropy(Y_true, Y_pred)
        self.assertAlmostEqual(loss, 0.0, places=4)

    def test_categorical_cross_entropy_clipping_no_nan_or_inf(self) -> None:
        """Zero probability predictions must be clipped and not produce math domain errors."""
        Y_true = [[1.0, 0.0]]
        Y_pred = [[0.0, 1.0]]  # completely wrong
        loss = categorical_cross_entropy(Y_true, Y_pred, epsilon=1e-15)
        self.assertFalse(math.isnan(loss))
        self.assertFalse(math.isinf(loss))
        self.assertGreater(loss, 30.0)  # -ln(1e-15) approx 34.5

    def test_softmax_cross_entropy_gradient(self) -> None:
        """Combined gradient dZ = Y_pred - Y_true."""
        Y_pred = [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
        ]
        Y_true = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        # Unnormalized:
        grad_unnorm = softmax_cross_entropy_gradient(Y_pred, Y_true, normalize=False)
        expected_unnorm = [
            [-0.3, 0.2, 0.1],
            [0.1, -0.2, 0.1],
        ]
        for row_act, row_exp in zip(grad_unnorm, expected_unnorm):
            for val_act, val_exp in zip(row_act, row_exp):
                self.assertAlmostEqual(val_act, val_exp, places=7)

        # Normalized by batch size N=2:
        grad_norm = softmax_cross_entropy_gradient(Y_pred, Y_true, normalize=True)
        expected_norm = [
            [-0.15, 0.1, 0.05],
            [0.05, -0.1, 0.05],
        ]
        for row_act, row_exp in zip(grad_norm, expected_norm):
            for val_act, val_exp in zip(row_act, row_exp):
                self.assertAlmostEqual(val_act, val_exp, places=7)


if __name__ == "__main__":
    unittest.main()
