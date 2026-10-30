"""Unit tests for Pure Python List neural network training engine."""

import unittest

from listai.config import IRIS_CSV_PATH
from listai.data.loader import encode_labels, load_iris_csv
from listai.data.preprocessing import StandardScaler
from listai.data.splitting import stratified_split
from listai.engines.list_engine.network import ListNeuralNetwork
from listai.engines.list_engine.trainer import ListTrainer


class TestTrainer(unittest.TestCase):
    """Test full training workflow on standardized Iris data."""

    @classmethod
    def setUpClass(cls) -> None:
        if not IRIS_CSV_PATH.exists():
            from scripts.prepare_iris import fetch_or_validate_iris
            fetch_or_validate_iris(IRIS_CSV_PATH)

        X, y_names, _, class_names = load_iris_csv(IRIS_CSV_PATH)
        y, _ = encode_labels(y_names, class_names)
        split = stratified_split(X, y, seed=42)

        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(split.X_train)
        cls.y_train = split.y_train
        cls.X_val = scaler.transform(split.X_val)
        cls.y_val = split.y_val
        cls.X_test = scaler.transform(split.X_test)
        cls.y_test = split.y_test

    def test_training_convergence(self) -> None:
        """Verify training loss decreases and accuracy reaches high level (>90%)."""
        model = ListNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=42)
        trainer = ListTrainer(
            model=model,
            epochs=100,
            batch_size=16,
            learning_rate=0.05,
            seed=42,
            shuffle=True,
        )

        history = trainer.fit(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            verbose=False,
        )

        self.assertEqual(len(history.train_losses), 100)
        self.assertEqual(len(history.val_losses), 100)

        # Initial vs final loss
        initial_loss = history.train_losses[0]
        final_loss = history.train_losses[-1]
        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_loss, 0.4)

        # Final accuracy should be high
        final_train_acc = history.train_accuracies[-1]
        self.assertGreater(final_train_acc, 0.90)

        # Test evaluation
        test_preds = model.predict(self.X_test)
        test_acc = sum(1 for p, t in zip(test_preds, self.y_test)) / len(self.y_test)
        self.assertGreater(test_acc, 0.90)


if __name__ == "__main__":
    unittest.main()
