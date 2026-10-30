"""Unit tests for finite-difference numerical gradient checking."""

import unittest

from listai.engines.list_engine.backprop import numerical_gradient_check
from listai.engines.list_engine.network import ListNeuralNetwork


class TestGradientCheck(unittest.TestCase):
    """Verify analytical backpropagation gradients against two-sided finite difference approximations."""

    def test_numerical_gradient_check_small_network(self) -> None:
        """Verify gradient checking on a compact 2-hidden layer network."""
        net = ListNeuralNetwork(layer_dims=(3, 5, 4, 2), seed=101)
        X = [
            [0.5, -0.3, 1.2],
            [-0.8, 0.4, 0.1],
            [1.0, -0.2, -0.5],
        ]
        Y = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        result = numerical_gradient_check(net, X, Y, epsilon=1e-5)

        self.assertLess(
            result["relative_error"],
            1e-5,
            f"Relative error too high: {result['relative_error']}",
        )
        self.assertLess(result["max_abs_difference"], 1e-4)

    def test_numerical_gradient_check_iris_architecture(self) -> None:
        """Verify gradient checking on the official 4 -> 16 -> 8 -> 3 Iris architecture."""
        net = ListNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=42)
        X = [
            [0.2, -0.5, 1.1, 0.7],
            [-1.1, 0.9, -0.3, 0.4],
            [0.8, 0.3, 0.5, -0.2],
            [0.1, -0.8, 1.3, -0.6],
        ]
        Y = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
        result = numerical_gradient_check(net, X, Y, epsilon=1e-5)

        self.assertLess(
            result["relative_error"],
            1e-5,
            f"Iris architecture relative error too high: {result['relative_error']}",
        )
        self.assertLess(result["max_abs_difference"], 1e-4)


if __name__ == "__main__":
    unittest.main()
