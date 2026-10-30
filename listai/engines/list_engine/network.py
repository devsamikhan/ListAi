"""Pure Python List implementation of Multilayer Perceptron (MLP) Neural Network.

Architecture:
    Input: 4 features
    Hidden Layer 1: 16 units, ReLU
    Hidden Layer 2: 8 units, ReLU
    Output Layer: 3 units, Softmax

Weights: He normal initialization N(0, sqrt(2 / fan_in))
Biases: Initialized to zeros
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from listai.config import ModelConfig
from listai.engines.list_engine.activations import relu, softmax
from listai.engines.list_engine.operations import (
    matrix_add_bias,
    matrix_mul,
    vector_zeros,
)
from listai.utils.reproducibility import create_rng, he_normal_matrix


@dataclass
class ForwardCache:
    """Stores intermediate pre-activations (Z) and post-activations (A) for backpropagation."""

    A0: List[List[float]]  # Input features X
    Z1: List[List[float]]  # Pre-activation Layer 1
    A1: List[List[float]]  # ReLU(Z1)
    Z2: List[List[float]]  # Pre-activation Layer 2
    A2: List[List[float]]  # ReLU(Z2)
    Z3: List[List[float]]  # Pre-activation Layer 3
    A3: List[List[float]]  # Softmax(Z3) - Output probabilities


class ListNeuralNetwork:
    """Multi-layer Perceptron built entirely with native Python Lists."""

    def __init__(
        self,
        layer_dims: Tuple[int, ...] = (4, 16, 8, 3),
        seed: int = 42,
    ) -> None:
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must have at least an input and output dimension.")

        self.layer_dims = tuple(layer_dims)
        self.seed = seed
        self.num_layers = len(layer_dims) - 1  # Number of weight layers (e.g., 3)

        self.weights: List[List[List[float]]] = []  # W[0], W[1], W[2]
        self.biases: List[List[float]] = []         # b[0], b[1], b[2]

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize weights using He normal initialization and biases to zero."""
        rng = create_rng(self.seed)
        self.weights = []
        self.biases = []

        for l in range(self.num_layers):
            fan_in = self.layer_dims[l]
            fan_out = self.layer_dims[l + 1]

            # Weight matrix of shape (fan_in, fan_out)
            W_l = he_normal_matrix(fan_in, fan_out, fan_in=fan_in, rng=rng)
            # Bias vector of shape (fan_out,)
            b_l = vector_zeros(fan_out)

            self.weights.append(W_l)
            self.biases.append(b_l)

    def forward(self, X: Sequence[Sequence[float]]) -> Tuple[List[List[float]], ForwardCache]:
        """Perform forward propagation across all layers and cache activations."""
        if not X:
            raise ValueError("Cannot perform forward propagation on empty input.")

        # A0 = X
        A0 = [list(row) for row in X]

        # Layer 1: Dense -> ReLU
        Z1 = matrix_add_bias(matrix_mul(A0, self.weights[0]), self.biases[0])
        A1 = relu(Z1)

        # Layer 2: Dense -> ReLU
        Z2 = matrix_add_bias(matrix_mul(A1, self.weights[1]), self.biases[1])
        A2 = relu(Z2)

        # Layer 3: Dense -> Softmax
        Z3 = matrix_add_bias(matrix_mul(A2, self.weights[2]), self.biases[2])
        A3 = softmax(Z3)

        cache = ForwardCache(A0=A0, Z1=Z1, A1=A1, Z2=Z2, A2=A2, Z3=Z3, A3=A3)
        return A3, cache

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        """Compute predicted class probabilities."""
        probabilities, _ = self.forward(X)
        return probabilities

    def predict(self, X: Sequence[Sequence[float]]) -> List[int]:
        """Predict integer class labels (argmax over class probabilities)."""
        probabilities = self.predict_proba(X)
        predictions: List[int] = []
        for row in probabilities:
            best_idx = 0
            best_prob = row[0]
            for idx in range(1, len(row)):
                if row[idx] > best_prob:
                    best_prob = row[idx]
                    best_idx = idx
            predictions.append(best_idx)
        return predictions

    def get_params(self) -> Dict[str, List]:
        """Return deep copies of all model parameters (for serialization and parity checks)."""
        return {
            "weights": [[[val for val in row] for row in W] for W in self.weights],
            "biases": [[val for val in b] for b in self.biases],
        }

    def set_params(self, params: Dict[str, List]) -> None:
        """Load external weights and biases into the model."""
        self.weights = [[[val for val in row] for row in W] for W in params["weights"]]
        self.biases = [[val for val in b] for b in params["biases"]]
