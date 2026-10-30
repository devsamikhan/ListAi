"""Phase 1 verification tests: project configuration, reproducibility, timing, and AST isolation."""

import ast
import math
from pathlib import Path
import unittest

from listai.config import (
    DATA_DIR,
    DOCS_DIR,
    FORBIDDEN_LIST_ENGINE_LIBRARIES,
    ListAIConfig,
    ModelConfig,
    PROJECT_ROOT,
    RESULTS_DIR,
)
from listai.utils.reproducibility import create_rng, he_normal_matrix, shuffled_indices
from listai.utils.timing import measure_repeated, timer
import main


class TestProjectFoundation(unittest.TestCase):
    """Verify Phase 1 repository structure, config, utilities, and strict isolation."""

    def test_directory_structure_exists(self) -> None:
        """All required directories must exist."""
        self.assertTrue(PROJECT_ROOT.is_dir())
        self.assertTrue(DATA_DIR.is_dir())
        self.assertTrue(DOCS_DIR.is_dir())
        self.assertTrue(RESULTS_DIR.is_dir())
        self.assertTrue((PROJECT_ROOT / "listai" / "engines" / "list_engine").is_dir())
        self.assertTrue((PROJECT_ROOT / "listai" / "engines" / "numpy_engine").is_dir())
        self.assertTrue((PROJECT_ROOT / "tests").is_dir())
        self.assertTrue((PROJECT_ROOT / "scripts").is_dir())

    def test_default_configuration(self) -> None:
        """Verify architecture (4 -> 16 -> 8 -> 3) and dataset split ratios."""
        cfg = ListAIConfig()
        self.assertEqual(cfg.model.layer_dims, (4, 16, 8, 3))
        self.assertAlmostEqual(
            cfg.dataset.train_ratio + cfg.dataset.val_ratio + cfg.dataset.test_ratio,
            1.0,
        )
        self.assertEqual(cfg.dataset.num_samples, 150)
        self.assertEqual(cfg.dataset.num_features, 4)
        self.assertEqual(cfg.dataset.num_classes, 3)

    def test_reproducibility_utilities(self) -> None:
        """Same seed must produce identical shuffled indices and He-initialized matrices."""
        rng1 = create_rng(42)
        rng2 = create_rng(42)
        self.assertEqual(shuffled_indices(20, rng1), shuffled_indices(20, rng2))

        w1 = he_normal_matrix(4, 16, fan_in=4, rng=create_rng(99))
        w2 = he_normal_matrix(4, 16, fan_in=4, rng=create_rng(99))
        self.assertEqual(len(w1), 4)
        self.assertEqual(len(w1[0]), 16)
        self.assertEqual(w1, w2)

        # Verify empirical std is near sqrt(2 / fan_in) for a larger sample
        large = he_normal_matrix(100, 100, fan_in=4, rng=create_rng(123))
        flat = [val for row in large for val in row]
        mean = sum(flat) / len(flat)
        variance = sum((x - mean) ** 2 for x in flat) / len(flat)
        expected_std = math.sqrt(2.0 / 4.0)
        self.assertAlmostEqual(math.sqrt(variance), expected_std, delta=0.05)

    def test_timing_utilities(self) -> None:
        """Verify timer context manager and repeated benchmark helper."""
        with timer() as rec:
            total = sum(range(1000))
        self.assertEqual(total, 499500)
        self.assertGreaterEqual(rec["elapsed_seconds"], 0.0)

        res, stats = measure_repeated(lambda: sum(range(500)), runs=3, measure_memory=True)
        self.assertEqual(res, sum(range(500)))
        self.assertEqual(stats["runs"], 3.0)
        self.assertIn("median_seconds", stats)
        self.assertIn("median_peak_kb", stats)

    def test_cli_argument_parsing(self) -> None:
        """Verify CLI accepts required flags."""
        exit_code = main.main(["--engine", "list", "--epochs", "10", "--batch-size", "8"])
        self.assertEqual(exit_code, 0)

    def test_list_engine_zero_forbidden_imports(self) -> None:
        """AST guard: Ensure no file in list_engine imports NumPy or any forbidden library."""
        list_engine_dir: Path = PROJECT_ROOT / "listai" / "engines" / "list_engine"
        py_files = list(list_engine_dir.rglob("*.py"))
        self.assertGreater(len(py_files), 0)

        for py_file in py_files:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_pkg = alias.name.split(".")[0].lower()
                        self.assertNotIn(
                            root_pkg,
                            FORBIDDEN_LIST_ENGINE_LIBRARIES,
                            f"Forbidden import '{alias.name}' found in {py_file}",
                        )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    root_pkg = node.module.split(".")[0].lower()
                    self.assertNotIn(
                        root_pkg,
                        FORBIDDEN_LIST_ENGINE_LIBRARIES,
                        f"Forbidden import from '{node.module}' found in {py_file}",
                    )


if __name__ == "__main__":
    unittest.main()
