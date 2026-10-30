"""Evaluation metrics for multi-class classification built without Scikit-learn."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from listai.data.preprocessing import one_hot_encode


@dataclass
class ClassificationReport:
    """Encapsulates comprehensive multi-class evaluation metrics."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_f1: List[float]
    support: List[int]
    confusion_matrix: List[List[int]]
    loss: float


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute classification accuracy."""
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must match.")
    if not y_true:
        return 0.0
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return float(correct) / float(len(y_true))


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: Optional[int] = None,
) -> List[List[int]]:
    """Compute confusion matrix where row i represents true class and col j represents predicted class.

    C[i][j] = count of instances with true class i predicted as class j.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must match.")

    if num_classes is None:
        max_true = max(y_true) if y_true else 0
        max_pred = max(y_pred) if y_pred else 0
        num_classes = max(max_true, max_pred) + 1

    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for yt, yp in zip(y_true, y_pred):
        if not (0 <= yt < num_classes and 0 <= yp < num_classes):
            raise ValueError(f"Label out of bounds [0, {num_classes - 1}]: yt={yt}, yp={yp}")
        cm[yt][yp] += 1

    return cm


def precision_recall_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: Optional[int] = None,
    average: str = "macro",
) -> Dict[str, Union[float, List[float]]]:
    """Compute precision, recall, and F1 score for multi-class classification.

    Supported averaging: 'macro', 'weighted', 'none'.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must match.")

    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    k = len(cm)

    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []
    supports: List[int] = []

    for c in range(k):
        tp = cm[c][c]
        fp = sum(cm[i][c] for i in range(k) if i != c)
        fn = sum(cm[c][j] for j in range(k) if j != c)
        support = sum(cm[c])
        supports.append(support)

        p = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        r = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * p * r) / (p + r) if (p + r) > 0.0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    total_samples = sum(supports)

    if average == "none":
        return {
            "precision": precisions,
            "recall": recalls,
            "f1": f1_scores,
            "support": supports,
        }
    elif average == "macro":
        return {
            "precision": sum(precisions) / k if k > 0 else 0.0,
            "recall": sum(recalls) / k if k > 0 else 0.0,
            "f1": sum(f1_scores) / k if k > 0 else 0.0,
        }
    elif average == "weighted":
        if total_samples == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        w_p = sum(p * s for p, s in zip(precisions, supports)) / total_samples
        w_r = sum(r * s for r, s in zip(recalls, supports)) / total_samples
        w_f1 = sum(f1 * s for f1, s in zip(f1_scores, supports)) / total_samples
        return {"precision": w_p, "recall": w_r, "f1": w_f1}
    else:
        raise ValueError(f"Unsupported average mode: {average}")


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    loss: float = 0.0,
    num_classes: Optional[int] = None,
) -> ClassificationReport:
    """Produce complete ClassificationReport from predicted and true labels."""
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    acc = accuracy_score(y_true, y_pred)
    per_class = precision_recall_f1(y_true, y_pred, num_classes=num_classes, average="none")
    macro = precision_recall_f1(y_true, y_pred, num_classes=num_classes, average="macro")
    weighted = precision_recall_f1(y_true, y_pred, num_classes=num_classes, average="weighted")

    return ClassificationReport(
        accuracy=acc,
        macro_precision=macro["precision"],
        macro_recall=macro["recall"],
        macro_f1=macro["f1"],
        weighted_precision=weighted["precision"],
        weighted_recall=weighted["recall"],
        weighted_f1=weighted["f1"],
        per_class_precision=per_class["precision"],
        per_class_recall=per_class["recall"],
        per_class_f1=per_class["f1"],
        support=per_class["support"],
        confusion_matrix=cm,
        loss=loss,
    )
