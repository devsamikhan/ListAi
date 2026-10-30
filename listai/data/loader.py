"""CSV dataset loading and validation for ListAI using Python Standard Library."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from listai.config import DatasetConfig, IRIS_CSV_PATH


def load_iris_csv(
    csv_path: Optional[Path] = None,
    expected_samples: int = 150,
    expected_features: int = 4,
) -> Tuple[List[List[float]], List[str], List[str], List[str]]:
    """Load and validate the Iris dataset CSV file using pure Python.

    Returns:
        (X, y_names, feature_names, class_names)
        - X: List[List[float]] of shape (num_samples, num_features)
        - y_names: List[str] of length num_samples with class label strings
        - feature_names: List[str] of feature column names
        - class_names: List[str] of unique sorted class names
    """
    path = Path(csv_path) if csv_path is not None else IRIS_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Iris CSV dataset not found at {path}. Run `python scripts/prepare_iris.py` first."
        )

    with open(path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    if not reader:
        raise ValueError(f"Iris CSV file at {path} is empty.")

    header = [h.strip() for h in reader[0]]
    if len(header) != expected_features + 1:
        raise ValueError(
            f"Expected {expected_features + 1} columns in header, got {len(header)}: {header}"
        )

    feature_names = header[:expected_features]
    target_column_name = header[expected_features]

    rows = reader[1:]
    if len(rows) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} rows in {path}, got {len(rows)}."
        )

    X: List[List[float]] = []
    y_names: List[str] = []

    for row_idx, row in enumerate(rows, start=2):
        if len(row) != expected_features + 1:
            raise ValueError(
                f"Row {row_idx} has {len(row)} values, expected {expected_features + 1}."
            )
        try:
            features = [float(val.strip()) for val in row[:expected_features]]
        except ValueError as err:
            raise ValueError(f"Non-numeric value in row {row_idx}: {err}") from err

        label = row[expected_features].strip()
        if not label:
            raise ValueError(f"Empty target label in row {row_idx}.")

        X.append(features)
        y_names.append(label)

    class_names = sorted(list(set(y_names)))
    return X, y_names, feature_names, class_names


def encode_labels(
    y_names: Sequence[str],
    class_order: Optional[Sequence[str]] = None,
) -> Tuple[List[int], List[str]]:
    """Convert string class labels to 0-indexed integers.

    Returns:
        (y_indices, class_order)
    """
    if class_order is None:
        unique_classes = sorted(list(set(y_names)))
    else:
        unique_classes = list(class_order)

    class_to_idx: Dict[str, int] = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_indices: List[int] = []
    for name in y_names:
        if name not in class_to_idx:
            raise KeyError(f"Label '{name}' not in known class order: {unique_classes}")
        y_indices.append(class_to_idx[name])

    return y_indices, unique_classes
