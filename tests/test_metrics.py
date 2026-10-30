"""Unit tests for Scikit-learn-free evaluation metrics."""

import unittest

from listai.evaluation.metrics import (
    accuracy_score,
    confusion_matrix,
    evaluate_predictions,
    precision_recall_f1,
)
from listai.evaluation.reports import format_classification_report, format_confusion_matrix


class TestMetrics(unittest.TestCase):
    """Test Accuracy, Precision, Recall, F1, and Confusion Matrix calculation."""

    def test_accuracy_score(self) -> None:
        """Verify accuracy calculation."""
        y_true = [0, 1, 2, 1, 0]
        y_pred = [0, 1, 2, 0, 0]
        self.assertEqual(accuracy_score(y_true, y_pred), 4.0 / 5.0)

    def test_confusion_matrix_and_f1(self) -> None:
        """Verify exact precision, recall, and F1 values on manual confusion table."""
        # Class 0: 2 samples, predicted [0, 1] -> TP=1, FN=1
        # Class 1: 2 samples, predicted [1, 1] -> TP=2, FN=0, FP=1 (from 0)
        # Class 2: 2 samples, predicted [2, 0] -> TP=1, FN=1, FP=0
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]

        cm = confusion_matrix(y_true, y_pred, num_classes=3)
        expected_cm = [
            [1, 1, 0],
            [0, 2, 0],
            [1, 0, 1],
        ]
        self.assertEqual(cm, expected_cm)

        per_class = precision_recall_f1(y_true, y_pred, num_classes=3, average="none")

        # Class 0: TP=1, FP=1, FN=1 -> P = 0.5, R = 0.5, F1 = 0.5
        self.assertAlmostEqual(per_class["precision"][0], 0.5)
        self.assertAlmostEqual(per_class["recall"][0], 0.5)
        self.assertAlmostEqual(per_class["f1"][0], 0.5)

        # Class 1: TP=2, FP=1, FN=0 -> P = 2/3, R = 1.0, F1 = 2*(2/3)*1 / (5/3) = 4/5 = 0.8
        self.assertAlmostEqual(per_class["precision"][1], 2.0 / 3.0)
        self.assertAlmostEqual(per_class["recall"][1], 1.0)
        self.assertAlmostEqual(per_class["f1"][1], 0.8)

        # Class 2: TP=1, FP=0, FN=1 -> P = 1.0, R = 0.5, F1 = 2*1*0.5 / 1.5 = 2/3
        self.assertAlmostEqual(per_class["precision"][2], 1.0)
        self.assertAlmostEqual(per_class["recall"][2], 0.5)
        self.assertAlmostEqual(per_class["f1"][2], 2.0 / 3.0)

    def test_report_generation(self) -> None:
        """Verify report string formatting produces expected sections."""
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        report = evaluate_predictions(y_true, y_pred, loss=0.05, num_classes=3)
        formatted = format_classification_report(report, ["setosa", "versicolor", "virginica"])

        self.assertIn("setosa", formatted)
        self.assertIn("Macro Average", formatted)
        self.assertIn("Confusion Matrix", formatted)
        self.assertEqual(report.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
