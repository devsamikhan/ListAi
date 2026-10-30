"""Data preprocessing and transformation utilities using pure Python.

Contains:
- StandardScaler: Fits mean and std on training data only (no data leakage).
- one_hot_encode: Converts integer class labels to one-hot vectors.
- one_hot_decode: Converts one-hot or probability vectors back to class indices.
"""

import math
from typing import List, Optional, Sequence


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.

    Parameters are calculated using training data ONLY to prevent data leakage.
    Formula: z = (x - u) / s
    """

    def __init__(self) -> None:
        self.means: Optional[List[float]] = None
        self.stds: Optional[List[float]] = None
        self.n_features: Optional[int] = None

    def fit(self, X: Sequence[Sequence[float]]) -> "StandardScaler":
        """Compute the mean and standard deviation for each feature column on training data."""
        if not X or not X[0]:
            raise ValueError("Cannot fit StandardScaler on empty dataset.")

        n_samples = len(X)
        self.n_features = len(X[0])

        means: List[float] = [0.0] * self.n_features
        for row in X:
            if len(row) != self.n_features:
                raise ValueError("Inconsistent row dimension in dataset.")
            for col_idx in range(self.n_features):
                means[col_idx] += float(row[col_idx])

        self.means = [m / n_samples for m in means]

        variances: List[float] = [0.0] * self.n_features
        for row in X:
            for col_idx in range(self.n_features):
                diff = float(row[col_idx]) - self.means[col_idx]
                variances[col_idx] += diff * diff

        # Population standard deviation with numerical epsilon guard
        self.stds = []
        for v in variances:
            s = math.sqrt(v / n_samples)
            self.stds.append(s if s > 1e-12 else 1.0)

        return self

    def transform(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        """Standardize dataset using previously fitted training parameters."""
        if self.means is None or self.stds is None or self.n_features is None:
            raise RuntimeError("StandardScaler must be fitted before transforming.")

        transformed: List[List[float]] = []
        for row in X:
            if len(row) != self.n_features:
                raise ValueError(
                    f"Row dimension {len(row)} does not match fitted features {self.n_features}."
                )
            transformed_row = [
                (float(row[col_idx]) - self.means[col_idx]) / self.stds[col_idx]
                for col_idx in range(self.n_features)
            ]
            transformed.append(transformed_row)

        return transformed

    def fit_transform(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        """Fit to training data, then transform it."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        """Revert standardized features back to original scale."""
        if self.means is None or self.stds is None or self.n_features is None:
            raise RuntimeError("StandardScaler must be fitted before inverse transform.")

        restored: List[List[float]] = []
        for row in X:
            if len(row) != self.n_features:
                raise ValueError("Row dimension mismatch during inverse transform.")
            restored_row = [
                float(row[col_idx]) * self.stds[col_idx] + self.means[col_idx]
                for col_idx in range(self.n_features)
            ]
            restored.append(restored_row)

        return restored


def one_hot_encode(y: Sequence[int], num_classes: int) -> List[List[float]]:
    """Convert integer class indices to 2D one-hot encoded float matrices."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    encoded: List[List[float]] = []
    for label in y:
        if not (0 <= label < num_classes):
            raise ValueError(f"Label {label} outside valid range [0, {num_classes - 1}].")
        row = [0.0] * num_classes
        row[label] = 1.0
        encoded.append(row)

    return encoded


def one_hot_decode(Y: Sequence[Sequence[float]]) -> List[int]:
    """Convert one-hot or probability vectors back into predicted class integer indices."""
    if not Y:
        return []

    decoded: List[int] = []
    for row in Y:
        best_idx = 0
        best_val = row[0]
        for idx in range(1, len(row)):
            if row[idx] > best_val:
                best_val = row[idx]
                best_idx = idx
        decoded.append(best_idx)

    return decoded
