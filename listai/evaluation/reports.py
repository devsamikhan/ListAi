"""Formatted text and markdown reporting utilities for ListAI evaluation metrics."""

from typing import List, Sequence

from listai.evaluation.metrics import ClassificationReport


def format_confusion_matrix(
    cm: List[List[int]],
    class_names: Sequence[str],
) -> str:
    """Format confusion matrix as a clear text table."""
    col_width = max(len(name) for name in class_names)
    col_width = max(col_width, 10)

    header = " " * (col_width + 4) + "".join(f"{name:>{col_width}} " for name in class_names)
    divider = "-" * len(header)
    lines = [
        "Confusion Matrix (Rows: True, Columns: Predicted):",
        divider,
        header,
        divider,
    ]

    for idx, row in enumerate(cm):
        row_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        row_str = f"{row_name:<{col_width}} | " + "".join(f"{val:>{col_width}d} " for val in row)
        lines.append(row_str)

    lines.append(divider)
    return "\n".join(lines)


def format_classification_report(
    report: ClassificationReport,
    class_names: Sequence[str],
    engine_name: str = "Model",
) -> str:
    """Format full classification report table."""
    col_width = max(len(name) for name in class_names)
    col_width = max(col_width, 16)

    lines = [
        f"==================================================================",
        f"  {engine_name} Evaluation Summary",
        f"==================================================================",
        f"Test Loss: {report.loss:.4f}  |  Overall Accuracy: {report.accuracy * 100:.2f}%",
        f"------------------------------------------------------------------",
        f"{'Class':<{col_width}} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
        f"------------------------------------------------------------------",
    ]

    for idx, name in enumerate(class_names):
        p = report.per_class_precision[idx]
        r = report.per_class_recall[idx]
        f1 = report.per_class_f1[idx]
        sup = report.support[idx]
        lines.append(f"{name:<{col_width}} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {sup:>10d}")

    lines.extend([
        f"------------------------------------------------------------------",
        f"{'Macro Average':<{col_width}} {report.macro_precision:>10.4f} {report.macro_recall:>10.4f} {report.macro_f1:>10.4f} {sum(report.support):>10d}",
        f"{'Weighted Average':<{col_width}} {report.weighted_precision:>10.4f} {report.weighted_recall:>10.4f} {report.weighted_f1:>10.4f} {sum(report.support):>10d}",
        f"==================================================================",
        "",
        format_confusion_matrix(report.confusion_matrix, class_names),
    ])

    return "\n".join(lines)
