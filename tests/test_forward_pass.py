"""Unit tests for Pure Python List neural network initialization and forward propagation."""

import unittest

from listai.engines.list_engine.network import ForwardCache, ListNeuralNetwork
from listai.engines.list_engine.operations import matrix_shape


class TestForwardPass(unittest.TestCase):
    """Test multi-layer network structure, dimensions, and forward inference."""

    def test_parameter_initialization(self) -> None:
        """Verify layer dimensions and zero biases."""
        net = ListNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=42)

        self.assertEqual(len(net.weights), 3)
        self.assertEqual(len(net.biases), 3)

        self.assertEqual(matrix_shape(net.weights[0]), (4, 16))
        self.assertEqual(matrix_shape(net.weights[1]), (16, 8))
        self.assertEqual(matrix_shape(net.weights[2]), (8, 3))

        self.assertEqual(len(net.biases[0]), 16)
        self.assertEqual(len(net.biases[1]), 8)
        self.assertEqual(len(net.biases[2]), 3)

        for b in net.biases:
            self.assertTrue(all(val == 0.0 for val in b))

    def test_forward_dimensions_and_probabilities(self) -> None:
        """Verify forward pass output dimensions and Softmax normalization."""
        net = ListNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=42)
        X = [
            [5.1, 3.5, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
            [6.3, 3.3, 6.0, 2.5],
            [5.8, 2.7, 5.1, 1.9],
            [4.9, 3.0, 1.4, 0.2],
        ]
        probs, cache = net.forward(X)

        self.assertEqual(matrix_shape(probs), (5, 3))
        self.assertEqual(matrix_shape(cache.A0), (5, 4))
        self.assertEqual(matrix_shape(cache.Z1), (5, 16))
        self.assertEqual(matrix_shape(cache.A1), (5, 16))
        self.assertEqual(matrix_shape(cache.Z2), (5, 8))
        self.assertEqual(matrix_shape(cache.A2), (5, 8))
        self.assertEqual(matrix_shape(cache.Z3), (5, 3))
        self.assertEqual(matrix_shape(cache.A3), (5, 3))

        # Check Softmax sums to 1.0 per row
        for row in probs:
            self.assertAlmostEqual(sum(row), 1.0, places=9)
            for p in row:
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

        # Check ReLU outputs are non-negative
        for row in cache.A1:
            self.assertTrue(all(val >= 0.0 for val in row))
        for row in cache.A2:
            self.assertTrue(all(val >= 0.0 for val in row))

    def test_predict_and_predict_proba(self) -> None:
        """Verify integer class predictions."""
        net = ListNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=42)
        X = [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, 1.0]]
        preds = net.predict(X)
        self.assertEqual(len(preds), 2)
        self.assertTrue(all(p in (0, 1, 2) for p in preds))

    def test_reproducibility_of_forward_pass(self) -> None:
        """Same random seed must yield identical outputs."""
        net1 = ListNeuralNetwork(seed=123)
        net2 = ListNeuralNetwork(seed=123)
        X = [[0.5, -0.2, 1.1, -0.9]]
        p1 = net1.predict_proba(X)
        p2 = net2.predict_proba(X)
        self.assertEqual(p1, p2)


if __name__ == "__main__":
    unittest.main()
