"""Centralized configuration and defaults for ListAI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
IRIS_CSV_PATH: Path = DATA_DIR / "iris.csv"

FORBIDDEN_LIST_ENGINE_LIBRARIES: Tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "tensorflow",
    "torch",
    "keras",
    "jax",
    "sklearn",
    "scikit-learn",
)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset splitting and preprocessing."""

    csv_path: Path = IRIS_CSV_PATH
    num_samples: int = 150
    num_features: int = 4
    num_classes: int = 3
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    class_names: Tuple[str, ...] = (
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica",
    )
    feature_names: Tuple[str, ...] = (
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    )


@dataclass(frozen=True)
class ModelConfig:
    """Neural network architecture specification: 4 -> 16 (ReLU) -> 8 (ReLU) -> 3 (Softmax)."""

    input_dim: int = 4
    hidden_dims: Tuple[int, ...] = (16, 8)
    output_dim: int = 3
    weight_init: str = "he_normal"
    random_seed: int = 42

    @property
    def layer_dims(self) -> Tuple[int, ...]:
        """Return full sequence of layer dimensions: (4, 16, 8, 3)."""
        return (self.input_dim, *self.hidden_dims, self.output_dim)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for Mini-Batch Stochastic Gradient Descent."""

    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 0.05
    random_seed: int = 42
    shuffle: bool = True


@dataclass(frozen=True)
class ListAIConfig:
    """Aggregate project configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
