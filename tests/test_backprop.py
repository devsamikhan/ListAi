"""Unit tests for Pure Python List backpropagation and parameter updates."""

import unittest

from listai.engines.list_engine.backprop import (
    backward_pass,
    compute_loss_and_gradients,
    update_parameters,
)
from listai.engines.list_engine.network import ListNeuralNetwork
from listai.engines.list_engine.operations import matrix_shape


class TestBackprop(unittest.TestCase):
    """Test gradient shapes, backward propagation, and parameter update steps."""

    def test_gradient_shapes(self) -> None:
        """Verify gradient tensor dimensions match corresponding weight and bias shapes."""
        net = ListNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=42)
        X = [
            [0.5, -0.2, 1.1, 0.4],
            [1.2, 0.8, -0.5, -0.1],
            [-0.3, 1.5, 0.2, 0.9],
        ]
        Y = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        _, grads = compute_loss_and_gradients(net, X, Y)

        self.assertEqual(len(grads.dW), 3)
        self.assertEqual(len(grads.db), 3)

        self.assertEqual(matrix_shape(grads.dW[0]), (4, 16))
        self.assertEqual(matrix_shape(grads.dW[1]), (16, 8))
        self.assertEqual(matrix_shape(grads.dW[2]), (8, 3))

        self.assertEqual(len(grads.db[0]), 16)
        self.assertEqual(len(grads.db[1]), 8)
        self.assertEqual(len(grads.db[2]), 3)

    def test_parameter_update_reduces_loss(self) -> None:
        """A single gradient descent step with reasonable learning rate must reduce loss."""
        net = ListNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=42)
        X = [
            [0.5, -0.2, 1.1, 0.4],
            [1.2, 0.8, -0.5, -0.1],
            [-0.3, 1.5, 0.2, 0.9],
            [0.1, -0.1, 0.3, -0.4],
        ]
        Y = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]

        initial_loss, grads = compute_loss_and_gradients(net, X, Y)
        update_parameters(net, grads, lr=0.1)
        new_loss, _ = compute_loss_and_gradients(net, X, Y)

        self.assertLess(new_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
