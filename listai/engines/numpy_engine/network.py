"""Vectorized Multilayer Perceptron implemented independently using NumPy."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np

from listai.engines.numpy_engine.activations import relu, softmax


@dataclass
class NumpyForwardCache:
    """Stores intermediate NumPy activations for backpropagation."""

    A0: np.ndarray
    Z1: np.ndarray
    A1: np.ndarray
    Z2: np.ndarray
    A2: np.ndarray
    Z3: np.ndarray
    A3: np.ndarray


class NumpyNeuralNetwork:
    """Multilayer Perceptron implemented entirely with NumPy arrays."""

    def __init__(
        self,
        layer_dims: Tuple[int, ...] = (4, 16, 8, 3),
        seed: int = 42,
    ) -> None:
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must have at least input and output dimensions.")

        self.layer_dims = tuple(layer_dims)
        self.seed = seed
        self.num_layers = len(layer_dims) - 1

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize weights with He normal initialization and biases with zeros."""
        rng = np.random.default_rng(self.seed)
        self.weights = []
        self.biases = []

        for l in range(self.num_layers):
            fan_in = self.layer_dims[l]
            fan_out = self.layer_dims[l + 1]

            std = np.sqrt(2.0 / float(fan_in))
            W_l = rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)
            b_l = np.zeros((fan_out,), dtype=np.float64)

            self.weights.append(W_l)
            self.biases.append(b_l)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, NumpyForwardCache]:
        """Perform vectorized forward propagation across all dense layers."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {X_arr.shape}")

        A0 = X_arr

        # Layer 1
        Z1 = A0 @ self.weights[0] + self.biases[0]
        A1 = relu(Z1)

        # Layer 2
        Z2 = A1 @ self.weights[1] + self.biases[1]
        A2 = relu(Z2)

        # Layer 3
        Z3 = A2 @ self.weights[2] + self.biases[2]
        A3 = softmax(Z3)

        cache = NumpyForwardCache(A0=A0, Z1=Z1, A1=A1, Z2=Z2, A2=A2, Z3=Z3, A3=A3)
        return A3, cache

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute predicted class probabilities."""
        probs, _ = self.forward(X)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices using argmax along the class axis."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_params(self) -> Dict[str, List]:
        """Return parameters as nested Python lists (for cross-engine parity tests)."""
        return {
            "weights": [W.tolist() for W in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    def set_params(self, params: Dict[str, Sequence]) -> None:
        """Load external weights and biases into NumPy arrays."""
        self.weights = [np.array(W, dtype=np.float64) for W in params["weights"]]
        self.biases = [np.array(b, dtype=np.float64) for b in params["biases"]]
