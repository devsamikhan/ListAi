"""Unit tests for Pure Python List activation functions (ReLU and numerically stable Softmax)."""

import math
import unittest

from listai.engines.list_engine.activations import relu, relu_derivative, softmax


class TestActivations(unittest.TestCase):
    """Test activation functions and their analytical derivatives."""

    def test_relu(self) -> None:
        """Verify ReLU sets negative values to 0 and passes positive values unchanged."""
        Z = [
            [-5.0, 0.0, 3.2],
            [10.0, -0.001, 1e-5],
        ]
        A = relu(Z)
        self.assertEqual(
            A,
            [
                [0.0, 0.0, 3.2],
                [10.0, 0.0, 1e-5],
            ],
        )

    def test_relu_derivative(self) -> None:
        """Verify ReLU derivative is 1.0 for z > 0 and 0.0 for z <= 0."""
        Z = [
            [-2.5, 0.0, 4.0],
            [0.001, -100.0, 50.0],
        ]
        dA = relu_derivative(Z)
        self.assertEqual(
            dA,
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
        )

    def test_softmax_validity_and_sum(self) -> None:
        """Softmax probabilities must sum to 1.0 and each be in [0, 1]."""
        Z = [
            [2.0, 1.0, 0.1],
            [-1.0, -2.0, -3.0],
        ]
        P = softmax(Z)
        for row in P:
            self.assertAlmostEqual(sum(row), 1.0, places=9)
            for p in row:
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_softmax_shift_invariance(self) -> None:
        """Softmax(Z + c) must equal Softmax(Z)."""
        Z1 = [[1.0, 2.0, 3.0]]
        Z2 = [[101.0, 102.0, 103.0]]
        P1 = softmax(Z1)
        P2 = softmax(Z2)
        for p1, p2 in zip(P1[0], P2[0]):
            self.assertAlmostEqual(p1, p2, places=10)

    def test_softmax_extreme_numerical_stability(self) -> None:
        """Verify large inputs (e.g. 1000.0) do not cause OverflowError in exp()."""
        Z_large = [[1000.0, 1001.0, 999.0]]
        P_large = softmax(Z_large)
        self.assertAlmostEqual(sum(P_large[0]), 1.0, places=9)
        self.assertGreater(P_large[0][1], P_large[0][0])
        self.assertGreater(P_large[0][0], P_large[0][2])


if __name__ == "__main__":
    unittest.main()
