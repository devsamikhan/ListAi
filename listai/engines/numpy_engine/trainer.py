"""Mini-batch SGD trainer using NumPy."""

from typing import Optional, Sequence
import numpy as np

from listai.data.preprocessing import one_hot_encode
from listai.engines.list_engine.trainer import TrainingHistory
from listai.engines.numpy_engine.backprop import (
    backward_pass,
    compute_loss_and_gradients,
    update_parameters,
)
from listai.engines.numpy_engine.loss import categorical_cross_entropy
from listai.engines.numpy_engine.network import NumpyNeuralNetwork


class NumpyTrainer:
    """Mini-batch SGD trainer for NumPy Neural Network."""

    def __init__(
        self,
        model: NumpyNeuralNetwork,
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

    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.model.predict(X)
        return float(np.mean(preds == y)) if len(y) > 0 else 0.0

    def fit(
        self,
        X_train: Sequence[Sequence[float]],
        y_train: Sequence[int],
        X_val: Optional[Sequence[Sequence[float]]] = None,
        y_val: Optional[Sequence[int]] = None,
        verbose: bool = False,
        log_interval: int = 20,
    ) -> TrainingHistory:
        """Train the NumPy model using mini-batch SGD."""
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.int64)
        n_samples = X_tr.shape[0]

        # Convert labels to one-hot ndarray
        Y_tr_oh = np.array(one_hot_encode(y_train, self.num_classes), dtype=np.float64)

        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        if has_val:
            X_v = np.asarray(X_val, dtype=np.float64)
            y_v = np.asarray(y_val, dtype=np.int64)
            Y_v_oh = np.array(one_hot_encode(y_val, self.num_classes), dtype=np.float64)

        rng = np.random.default_rng(self.seed)
        history = TrainingHistory(epochs=self.epochs)

        for epoch in range(1, self.epochs + 1):
            if self.shuffle:
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx : start_idx + self.batch_size]
                X_batch = X_tr[batch_indices]
                Y_batch = Y_tr_oh[batch_indices]

                # Forward pass
                _, cache = self.model.forward(X_batch)

                # Backward pass
                grads = backward_pass(self.model, cache, Y_batch)

                # Parameter update
                update_parameters(self.model, grads, self.learning_rate)

            # Record epoch statistics
            train_preds, _ = self.model.forward(X_tr)
            train_loss = categorical_cross_entropy(Y_tr_oh, train_preds)
            train_acc = self._compute_accuracy(X_tr, y_tr)

            history.train_losses.append(train_loss)
            history.train_accuracies.append(train_acc)

            if has_val:
                val_preds, _ = self.model.forward(X_v)
                val_loss = categorical_cross_entropy(Y_v_oh, val_preds)
                val_acc = self._compute_accuracy(X_v, y_v)
                history.val_losses.append(val_loss)
                history.val_accuracies.append(val_acc)
            else:
                val_loss = 0.0
                val_acc = 0.0

            if verbose and (epoch == 1 or epoch % log_interval == 0 or epoch == self.epochs):
                val_str = f" | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}" if has_val else ""
                print(
                    f"[NumPy] Epoch [{epoch:3d}/{self.epochs:3d}] "
                    f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}{val_str}"
                )

        return history
