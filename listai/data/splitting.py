"""Stratified train, validation, and test dataset splitting using pure Python."""

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Sequence, Tuple

from listai.utils.reproducibility import create_rng


@dataclass(frozen=True)
class StratifiedSplit:
    """Holds data matrices and label vectors for train, val, and test splits."""

    X_train: List[List[float]]
    y_train: List[int]
    X_val: List[List[float]]
    y_val: List[int]
    X_test: List[List[float]]
    y_test: List[int]
    class_names: List[str]
    feature_names: List[str]

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def val_size(self) -> int:
        return len(self.X_val)

    @property
    def test_size(self) -> int:
        return len(self.X_test)

    @property
    def total_size(self) -> int:
        return self.train_size + self.val_size + self.test_size


def stratified_split(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    class_names: Optional[Sequence[str]] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> StratifiedSplit:
    """Perform a stratified split into training, validation, and test sets.

    Ensures that the proportion of each class is preserved across all three splits.
    """
    if len(X) != len(y):
        raise ValueError(f"Features length ({len(X)}) does not match labels length ({len(y)})")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0 (got {ratio_sum})")

    rng = create_rng(seed)

    # Group sample indices by class
    class_to_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(y):
        class_to_indices.setdefault(label, []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for label in sorted(class_to_indices.keys()):
        indices = list(class_to_indices[label])
        rng.shuffle(indices)

        n_total = len(indices)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        n_test = n_total - n_train - n_val

        # Safety adjustment if rounding overflowed
        if n_test < 0:
            n_val += n_test
            n_test = 0

        train_part = indices[:n_train]
        val_part = indices[n_train : n_train + n_val]
        test_part = indices[n_train + n_val :]

        train_indices.extend(train_part)
        val_indices.extend(val_part)
        test_indices.extend(test_part)

    # Deterministically shuffle combined sets so samples are not ordered by class
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    def select(indices: List[int]) -> Tuple[List[List[float]], List[int]]:
        sub_X = [list(X[i]) for i in indices]
        sub_y = [y[i] for i in indices]
        return sub_X, sub_y

    X_train, y_train = select(train_indices)
    X_val, y_val = select(val_indices)
    X_test, y_test = select(test_indices)

    c_names = list(class_names) if class_names else [f"class_{i}" for i in sorted(class_to_indices.keys())]
    f_names = list(feature_names) if feature_names else [f"feature_{i}" for i in range(len(X[0]))]

    return StratifiedSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_names=c_names,
        feature_names=f_names,
    )
