"""Script to download, format, and validate the canonical UCI Iris dataset into `data/iris.csv`."""

import csv
from pathlib import Path
import sys
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IRIS_CSV = DATA_DIR / "iris.csv"

UCI_IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

HEADER = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
EXPECTED_CLASSES = {"Iris-setosa": 50, "Iris-versicolor": 50, "Iris-virginica": 50}


def fetch_or_validate_iris(target_path: Path = IRIS_CSV) -> Path:
    """Fetch UCI Iris dataset or validate if already present."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            if len(reader) >= 151 and reader[0] == HEADER:
                print(f"[prepare_iris] Valid existing dataset found at {target_path}")
                return target_path

    print(f"[prepare_iris] Fetching official UCI Iris dataset from {UCI_IRIS_URL}...")
    try:
        with urllib.request.urlopen(UCI_IRIS_URL, timeout=15) as response:
            raw_text = response.read().decode("utf-8")
            raw_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    except Exception as exc:
        raise RuntimeError(f"Failed to download Iris dataset from {UCI_IRIS_URL}: {exc}") from exc

    if len(raw_lines) != 150:
        raise ValueError(f"Expected 150 samples from UCI, got {len(raw_lines)}")

    parsed_rows = []
    class_counts = {cls: 0 for cls in EXPECTED_CLASSES}

    for idx, line in enumerate(raw_lines, start=1):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Line {idx} invalid: expected 5 elements, got {len(parts)} ({line})")
        # Validate numeric features
        features = [float(x) for x in parts[:4]]
        species = parts[4]
        if species not in EXPECTED_CLASSES:
            raise ValueError(f"Unknown species '{species}' on line {idx}")
        class_counts[species] += 1
        parsed_rows.append([f"{f:.1f}" if f.is_integer() else f"{f}" for f in features] + [species])

    for cls, count in EXPECTED_CLASSES.items():
        if class_counts[cls] != count:
            raise ValueError(f"Class '{cls}' count mismatch: expected {count}, got {class_counts[cls]}")

    with open(target_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(parsed_rows)

    print(f"[prepare_iris] Successfully prepared {target_path} (150 samples: 50 per class).")
    return target_path


if __name__ == "__main__":
    fetch_or_validate_iris()
