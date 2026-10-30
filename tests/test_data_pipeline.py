"""Unit tests for Iris dataset loading, stratified splitting, and data preprocessing."""

import math
from pathlib import Path
import unittest

from listai.config import IRIS_CSV_PATH
from listai.data.loader import encode_labels, load_iris_csv
from listai.data.preprocessing import StandardScaler, one_hot_decode, one_hot_encode
from listai.data.splitting import stratified_split


class TestDataPipeline(unittest.TestCase):
    """Test data ingestion, schema validation, stratification, and preprocessing."""

    @classmethod
    def setUpClass(cls) -> None:
        """Ensure iris.csv exists before running tests."""
        if not IRIS_CSV_PATH.exists():
            from scripts.prepare_iris import fetch_or_validate_iris
            fetch_or_validate_iris(IRIS_CSV_PATH)

    def test_load_iris_csv(self) -> None:
        """Verify 150 samples, 4 features, 3 distinct classes, 50 samples per class."""
        X, y_names, feature_names, class_names = load_iris_csv(IRIS_CSV_PATH)
        self.assertEqual(len(X), 150)
        self.assertEqual(len(y_names), 150)
        self.assertEqual(len(feature_names), 4)
        self.assertEqual(len(class_names), 3)

        # Check 50 samples per class
        counts = {c: 0 for c in class_names}
        for name in y_names:
            counts[name] += 1
        for c, count in counts.items():
            self.assertEqual(count, 50, f"Expected 50 samples for {c}, got {count}")

        # Check sample numeric validity
        for row in X:
            self.assertEqual(len(row), 4)
            for val in row:
                self.assertIsInstance(val, float)
                self.assertFalse(math.isnan(val))
                self.assertGreater(val, 0.0)

    def test_encode_labels(self) -> None:
        """Verify string labels map to 0, 1, 2 monotonically."""
        names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica", "Iris-setosa"]
        encoded, order = encode_labels(names)
        self.assertEqual(encoded, [0, 1, 2, 0])
        self.assertEqual(order, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

        with self.assertRaises(KeyError):
            encode_labels(["Invalid-species"], class_order=["Iris-setosa"])

    def test_stratified_split(self) -> None:
        """Verify train/val/test sizes and class distribution preservation."""
        X, y_names, feature_names, class_names = load_iris_csv(IRIS_CSV_PATH)
        y, _ = encode_labels(y_names, class_names)

        split = stratified_split(
            X, y,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            class_names=class_names,
            feature_names=feature_names,
        )

        self.assertEqual(split.total_size, 150)
        self.assertEqual(split.train_size, 105)  # 35 * 3
        self.assertEqual(split.val_size + split.test_size, 45)

        # Verify exact class stratification
        def class_distribution(labels):
            d = {}
            for lbl in labels:
                d[lbl] = d.get(lbl, 0) + 1
            return d

        train_dist = class_distribution(split.y_train)
        val_dist = class_distribution(split.y_val)
        test_dist = class_distribution(split.y_test)

        self.assertEqual(train_dist, {0: 35, 1: 35, 2: 35})
        # Check that val and test each hold ~7-8 of each class
        for c in (0, 1, 2):
            self.assertEqual(val_dist[c] + test_dist[c], 15)

    def test_standardization_no_leakage(self) -> None:
        """StandardScaler must fit on train only, leaving train mean near 0 and variance near 1."""
        X, y_names, _, class_names = load_iris_csv(IRIS_CSV_PATH)
        y, _ = encode_labels(y_names, class_names)
        split = stratified_split(X, y, seed=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(split.X_train)
        X_val_scaled = scaler.transform(split.X_val)
        X_test_scaled = scaler.transform(split.X_test)

        # Check train mean ~ 0, std ~ 1
        n_train = len(X_train_scaled)
        for col in range(4):
            col_vals = [row[col] for row in X_train_scaled]
            mean = sum(col_vals) / n_train
            var = sum((v - mean) ** 2 for v in col_vals) / n_train
            self.assertAlmostEqual(mean, 0.0, places=6)
            self.assertAlmostEqual(math.sqrt(var), 1.0, places=6)

        # Inverse transform on test set must restore original values accurately
        restored_test = scaler.inverse_transform(X_test_scaled)
        for original_row, restored_row in zip(split.X_test, restored_test):
            for orig, rest in zip(original_row, restored_row):
                self.assertAlmostEqual(orig, rest, places=7)

    def test_one_hot_encode_and_decode(self) -> None:
        """Verify one-hot matrix generation and argmax decoding."""
        labels = [0, 1, 2, 1, 0]
        one_hot = one_hot_encode(labels, num_classes=3)
        self.assertEqual(
            one_hot,
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        )
        decoded = one_hot_decode(one_hot)
        self.assertEqual(decoded, labels)


if __name__ == "__main__":
    unittest.main()
