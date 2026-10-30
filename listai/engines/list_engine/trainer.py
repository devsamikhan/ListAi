"""Pure Python List mini-batch Stochastic Gradient Descent trainer."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from listai.config import TrainingConfig
from listai.data.preprocessing import one_hot_encode
from listai.engines.list_engine.backprop import (
    backward_pass,
    compute_loss_and_gradients,
    update_parameters,
)
from listai.engines.list_engine.loss import categorical_cross_entropy
from listai.engines.list_engine.network import ListNeuralNetwork
from listai.utils.reproducibility import create_rng, shuffled_indices


@dataclass
class TrainingHistory:
    """Records loss and metric progression across training epochs."""

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    epochs: int = 0


class ListTrainer:
    """Mini-batch SGD trainer using pure Python Lists."""

    def __init__(
        self,
        model: ListNeuralNetwork,
        epochs: int = 200,
        batch_size: int = 16,
        learning_rate: float = 0.05,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.shuffle = shuffle
        self.num_classes = model.layer_dims[-1]

    def _compute_accuracy(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> float:
        preds = self.model.predict(X)
        correct = sum(1 for p, t in zip(preds, y) if p == t)
        return float(correct) / float(len(y)) if y else 0.0

    def fit(
        self,
        X_train: Sequence[Sequence[float]],
        y_train: Sequence[int],
        X_val: Optional[Sequence[Sequence[float]]] = None,
        y_val: Optional[Sequence[int]] = None,
        verbose: bool = False,
        log_interval: int = 20,
    ) -> TrainingHistory:
        """Train the model with mini-batch Stochastic Gradient Descent."""
        n_samples = len(X_train)
        if n_samples == 0:
            raise ValueError("Training dataset cannot be empty.")

        Y_train_one_hot = one_hot_encode(y_train, self.num_classes)
        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        if has_val:
            Y_val_one_hot = one_hot_encode(y_val, self.num_classes)

        rng = create_rng(self.seed)
        history = TrainingHistory(epochs=self.epochs)

        for epoch in range(1, self.epochs + 1):
            # Mini-batch shuffling
            if self.shuffle:
                indices = shuffled_indices(n_samples, rng)
            else:
                indices = list(range(n_samples))

            # Iterate through mini-batches
            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx : start_idx + self.batch_size]
                X_batch = [X_train[i] for i in batch_indices]
                Y_batch = [Y_train_one_hot[i] for i in batch_indices]

                # Forward pass
                _, cache = self.model.forward(X_batch)

                # Explicit backward pass
                grads = backward_pass(self.model, cache, Y_batch)

                # Parameter update
                update_parameters(self.model, grads, self.learning_rate)

            # Evaluate epoch metrics
            train_preds, _ = self.model.forward(X_train)
            train_loss = categorical_cross_entropy(Y_train_one_hot, train_preds)
            train_acc = self._compute_accuracy(X_train, y_train)

            history.train_losses.append(train_loss)
            history.train_accuracies.append(train_acc)

            if has_val:
                val_preds, _ = self.model.forward(X_val)
                val_loss = categorical_cross_entropy(Y_val_one_hot, val_preds)
                val_acc = self._compute_accuracy(X_val, y_val)
                history.val_losses.append(val_loss)
                history.val_accuracies.append(val_acc)
            else:
                val_loss = 0.0
                val_acc = 0.0

            if verbose and (epoch == 1 or epoch % log_interval == 0 or epoch == self.epochs):
                val_str = f" | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}" if has_val else ""
                print(
                    f"Epoch [{epoch:3d}/{self.epochs:3d}] "
                    f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}{val_str}"
                )

        return history
