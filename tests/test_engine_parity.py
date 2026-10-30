"""Comprehensive parity tests verifying numerical equivalence between Engine A (Lists) and Engine B (NumPy)."""

import unittest
import numpy as np

import listai.engines.list_engine.activations as list_act
import listai.engines.list_engine.backprop as list_bp
import listai.engines.list_engine.loss as list_loss
from listai.engines.list_engine.network import ListNeuralNetwork
from listai.engines.list_engine.trainer import ListTrainer

import listai.engines.numpy_engine.activations as np_act
import listai.engines.numpy_engine.backprop as np_bp
import listai.engines.numpy_engine.loss as np_loss
from listai.engines.numpy_engine.network import NumpyNeuralNetwork
from listai.engines.numpy_engine.trainer import NumpyTrainer


class TestEngineParity(unittest.TestCase):
    """Verify that Pure Python List Engine and NumPy Engine produce identical numerical results."""

    def test_activation_parity(self) -> None:
        """Verify ReLU and Softmax match across engines within 1e-12."""
        Z_list = [
            [-3.5, 0.0, 1.2, 4.8],
            [10.5, -0.01, 0.001, -100.0],
            [100.0, 102.0, 98.0, 99.5],
        ]
        Z_np = np.array(Z_list, dtype=np.float64)

        # 1. ReLU
        a_list = list_act.relu(Z_list)
        a_np = np_act.relu(Z_np)
        np.testing.assert_allclose(a_list, a_np, rtol=1e-12, atol=1e-12)

        # 2. ReLU derivative
        da_list = list_act.relu_derivative(Z_list)
        da_np = np_act.relu_derivative(Z_np)
        np.testing.assert_allclose(da_list, da_np, rtol=1e-12, atol=1e-12)

        # 3. Softmax
        sm_list = list_act.softmax(Z_list)
        sm_np = np_act.softmax(Z_np)
        np.testing.assert_allclose(sm_list, sm_np, rtol=1e-12, atol=1e-12)

    def test_loss_and_gradient_parity(self) -> None:
        """Verify categorical cross entropy and combined gradients match within 1e-12."""
        Y_true_list = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        Y_pred_list = [
            [0.85, 0.10, 0.05],
            [0.05, 0.90, 0.05],
            [0.15, 0.15, 0.70],
        ]
        Y_true_np = np.array(Y_true_list, dtype=np.float64)
        Y_pred_np = np.array(Y_pred_list, dtype=np.float64)

        # Loss
        loss_list = list_loss.categorical_cross_entropy(Y_true_list, Y_pred_list)
        loss_np = np_loss.categorical_cross_entropy(Y_true_np, Y_pred_np)
        self.assertAlmostEqual(loss_list, loss_np, places=12)

        # Combined output gradient (unnormalized and normalized)
        grad_list_unnorm = list_loss.softmax_cross_entropy_gradient(Y_pred_list, Y_true_list, normalize=False)
        grad_np_unnorm = np_loss.softmax_cross_entropy_gradient(Y_pred_np, Y_true_np, normalize=False)
        np.testing.assert_allclose(grad_list_unnorm, grad_np_unnorm, rtol=1e-12, atol=1e-12)

        grad_list_norm = list_loss.softmax_cross_entropy_gradient(Y_pred_list, Y_true_list, normalize=True)
        grad_np_norm = np_loss.softmax_cross_entropy_gradient(Y_pred_np, Y_true_np, normalize=True)
        np.testing.assert_allclose(grad_list_norm, grad_np_norm, rtol=1e-12, atol=1e-12)

    def test_forward_pass_parity(self) -> None:
        """With identical weights, forward pass activations must match exactly within 1e-12."""
        net_list = ListNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=42)
        net_np = NumpyNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=999)

        # Synchronize parameters from list engine to numpy engine
        params = net_list.get_params()
        net_np.set_params(params)

        X = [
            [0.5, -1.2, 2.3, -0.4],
            [1.1, 0.9, -0.8, 0.2],
            [-0.7, 0.3, 1.5, -1.0],
            [0.0, -0.5, 0.5, 0.1],
        ]
        X_np = np.array(X, dtype=np.float64)

        out_list, cache_list = net_list.forward(X)
        out_np, cache_np = net_np.forward(X_np)

        np.testing.assert_allclose(cache_list.Z1, cache_np.Z1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cache_list.A1, cache_np.A1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cache_list.Z2, cache_np.Z2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cache_list.A2, cache_np.A2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cache_list.Z3, cache_np.Z3, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out_list, out_np, rtol=1e-12, atol=1e-12)

    def test_backward_pass_and_gradient_parity(self) -> None:
        """With identical weights and inputs, analytical gradients must match within 1e-12."""
        net_list = ListNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=42)
        net_np = NumpyNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=999)
        net_np.set_params(net_list.get_params())

        X = [
            [0.5, -1.2, 2.3, -0.4],
            [1.1, 0.9, -0.8, 0.2],
            [-0.7, 0.3, 1.5, -1.0],
            [0.0, -0.5, 0.5, 0.1],
        ]
        Y = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
        X_np = np.array(X, dtype=np.float64)
        Y_np = np.array(Y, dtype=np.float64)

        loss_list, grads_list = list_bp.compute_loss_and_gradients(net_list, X, Y)
        loss_np, grads_np = np_bp.compute_loss_and_gradients(net_np, X_np, Y_np)

        self.assertAlmostEqual(loss_list, loss_np, places=12)

        for l in range(3):
            np.testing.assert_allclose(grads_list.dW[l], grads_np.dW[l], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(grads_list.db[l], grads_np.db[l], rtol=1e-12, atol=1e-12)

    def test_multi_epoch_training_trajectory_parity(self) -> None:
        """Train both engines for 10 epochs without shuffling: trajectories must match within 1e-9."""
        net_list = ListNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=777)
        net_np = NumpyNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=999)
        net_np.set_params(net_list.get_params())

        X = [
            [0.5, -0.2, 1.1, 0.4],
            [1.2, 0.8, -0.5, -0.1],
            [-0.3, 1.5, 0.2, 0.9],
            [0.1, -0.1, 0.3, -0.4],
        ]
        y = [0, 1, 2, 0]

        trainer_list = ListTrainer(net_list, epochs=10, batch_size=2, learning_rate=0.05, shuffle=False)
        trainer_np = NumpyTrainer(net_np, epochs=10, batch_size=2, learning_rate=0.05, shuffle=False)

        hist_list = trainer_list.fit(X, y)
        hist_np = trainer_np.fit(X, y)

        np.testing.assert_allclose(
            hist_list.train_losses,
            hist_np.train_losses,
            rtol=1e-9,
            atol=1e-9,
        )
        self.assertEqual(net_list.predict(X), net_np.predict(np.array(X)).tolist())


if __name__ == "__main__":
    unittest.main()
