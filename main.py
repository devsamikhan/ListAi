"""Command-Line Interface entrypoint for ListAI.

Supports running:
    python main.py --engine list --epochs 200
    python main.py --engine numpy --epochs 200
    python main.py --engine both --epochs 200
"""

import argparse
import sys
import time
from typing import List, Optional

from listai.config import IRIS_CSV_PATH, ListAIConfig, TrainingConfig
from listai.data.loader import encode_labels, load_iris_csv
from listai.data.preprocessing import StandardScaler
from listai.data.splitting import stratified_split
from listai.engines.list_engine.loss import categorical_cross_entropy as list_cce
from listai.engines.list_engine.network import ListNeuralNetwork
from listai.engines.list_engine.trainer import ListTrainer
from listai.engines.numpy_engine.loss import categorical_cross_entropy as np_cce
from listai.engines.numpy_engine.network import NumpyNeuralNetwork
from listai.engines.numpy_engine.trainer import NumpyTrainer
from listai.evaluation.metrics import evaluate_predictions
from listai.evaluation.reports import format_classification_report
from scripts.prepare_iris import fetch_or_validate_iris


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for ListAI CLI."""
    parser = argparse.ArgumentParser(
        prog="listai",
        description=(
            "ListAI: Understanding Neural Networks Through Python Lists.\n"
            "Train and evaluate a pure Python List neural network vs an "
            "independent NumPy neural network on the UCI Iris dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["list", "numpy", "both"],
        default="both",
        help="Which neural network engine to run: 'list', 'numpy', or 'both' (default: 'both').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TrainingConfig.epochs,
        help=f"Number of training epochs (default: {TrainingConfig.epochs}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TrainingConfig.batch_size,
        help=f"Mini-batch size for SGD (default: {TrainingConfig.batch_size}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=TrainingConfig.learning_rate,
        help=f"Learning rate for SGD (default: {TrainingConfig.learning_rate}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TrainingConfig.random_seed,
        help=f"Random seed for reproducibility (default: {TrainingConfig.random_seed}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print training loss and accuracy every 20 epochs.",
    )
    return parser


def run_pipeline(
    engine: str = "both",
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 0.05,
    seed: int = 42,
    verbose: bool = False,
) -> int:
    """Execute full data loading, stratified splitting, training, and evaluation pipeline."""
    # Ensure dataset is ready
    fetch_or_validate_iris(IRIS_CSV_PATH)

    # 1. Load data
    X_raw, y_names, feature_names, class_names = load_iris_csv(IRIS_CSV_PATH)
    y_raw, _ = encode_labels(y_names, class_names)

    # 2. Stratified split (70% train, 15% val, 15% test)
    split = stratified_split(
        X_raw,
        y_raw,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
        class_names=class_names,
        feature_names=feature_names,
    )

    # 3. Fit scaler ONLY on training data (prevent data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(split.X_train)
    X_val = scaler.transform(split.X_val)
    X_test = scaler.transform(split.X_test)

    print(
        f"\n[ListAI] Dataset Loaded & Split: Train={split.train_size}, "
        f"Val={split.val_size}, Test={split.test_size} (Features: {len(feature_names)}, Classes: {len(class_names)})"
    )
    print(
        f"[ListAI] Architecture: 4 -> 16 (ReLU) -> 8 (ReLU) -> 3 (Softmax) | "
        f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}, Seed: {seed}\n"
    )

    list_report = None
    numpy_report = None
    list_time = 0.0
    numpy_time = 0.0

    # 4. Engine A: Pure Python Lists
    if engine in ("list", "both"):
        print(">>> Running Engine A: Pure Python List Engine...")
        list_model = ListNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=seed)
        list_trainer = ListTrainer(
            model=list_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            seed=seed,
            shuffle=True,
        )

        t0 = time.perf_counter()
        list_history = list_trainer.fit(
            X_train, split.y_train, X_val, split.y_val, verbose=verbose
        )
        list_time = time.perf_counter() - t0

        test_preds = list_model.predict(X_test)
        test_probs = list_model.predict_proba(X_test)
        from listai.data.preprocessing import one_hot_encode
        Y_test_oh = one_hot_encode(split.y_test, len(class_names))
        test_loss = list_cce(Y_test_oh, test_probs)

        list_report = evaluate_predictions(
            split.y_test, test_preds, loss=test_loss, num_classes=len(class_names)
        )
        print(f"    Training completed in {list_time:.4f}s")
        print(format_classification_report(list_report, class_names, engine_name="Engine A (Pure Python Lists)"))
        print()

    # 5. Engine B: NumPy Engine
    if engine in ("numpy", "both"):
        print(">>> Running Engine B: Independent NumPy Engine...")
        import numpy as np

        numpy_model = NumpyNeuralNetwork(layer_dims=(4, 16, 8, 3), seed=seed)
        numpy_trainer = NumpyTrainer(
            model=numpy_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            seed=seed,
            shuffle=True,
        )

        t0 = time.perf_counter()
        numpy_history = numpy_trainer.fit(
            X_train, split.y_train, X_val, split.y_val, verbose=verbose
        )
        numpy_time = time.perf_counter() - t0

        X_test_np = np.array(X_test, dtype=np.float64)
        from listai.data.preprocessing import one_hot_encode
        Y_test_np = np.array(one_hot_encode(split.y_test, len(class_names)), dtype=np.float64)

        test_preds_np = numpy_model.predict(X_test_np).tolist()
        test_probs_np = numpy_model.predict_proba(X_test_np)
        test_loss_np = np_cce(Y_test_np, test_probs_np)

        numpy_report = evaluate_predictions(
            split.y_test, test_preds_np, loss=test_loss_np, num_classes=len(class_names)
        )
        print(f"    Training completed in {numpy_time:.4f}s")
        print(format_classification_report(numpy_report, class_names, engine_name="Engine B (NumPy)"))
        print()

    # 6. Side-by-side comparison if both were run
    if engine == "both" and list_report and numpy_report:
        speedup = (list_time / numpy_time) if numpy_time > 0 else 1.0
        print("==================================================================")
        print("  Side-by-Side Comparison: Pure Python Lists vs NumPy")
        print("==================================================================")
        print(f"Metric                    Pure Python Lists         NumPy Engine")
        print("------------------------------------------------------------------")
        print(f"Training Time (s)         {list_time:>17.4f}s   {numpy_time:>18.4f}s")
        print(f"Test Accuracy             {list_report.accuracy * 100:>16.2f}%   {numpy_report.accuracy * 100:>17.2f}%")
        print(f"Test Loss                 {list_report.loss:>17.4f}    {numpy_report.loss:>18.4f}")
        print(f"Macro F1-Score            {list_report.macro_f1:>17.4f}    {numpy_report.macro_f1:>18.4f}")
        print(f"Speedup Factor (NumPy)    {speedup:>17.2f}x faster")
        print("==================================================================\n")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse CLI arguments and run pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer.")
    if args.lr <= 0.0:
        parser.error("--lr must be a positive float.")

    return run_pipeline(
        engine=args.engine,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
