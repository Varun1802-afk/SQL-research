"""Unit tests for model training pipeline and detection evaluation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.loader import DataSlice, SQLRecord
from src.features.manual_li import FEATURE_NAMES_32
from src.models.evaluate import evaluate_detection_pipeline
from src.models.train import (
    SQLDetectionPipeline,
    build_estimator,
    extract_features_matrix,
    train_model_pipeline,
)


class TestModelsAndEvaluation(unittest.TestCase):
    """Test model training, feature extraction, scaling, and evaluation metrics."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Synthetic training slice
        records_train = []
        for i in range(40):
            records_train.append(
                SQLRecord(
                    full_query=f"SELECT id, name FROM users WHERE id = {i};",
                    label=0,
                    split="train",
                    domain="test_src",
                )
            )
        for i in range(20):
            records_train.append(
                SQLRecord(
                    full_query=f"SELECT * FROM users WHERE id = {i} OR 1=1;",
                    label=1,
                    split="train",
                    domain="test_src",
                    attack_technique="boolean",
                )
            )
        self.train_slice = DataSlice("synthetic_train", records_train)

        # Synthetic evaluation slice
        records_eval = []
        for i in range(20):
            records_eval.append(
                SQLRecord(
                    full_query=f"SELECT col FROM table WHERE val = '{i}';",
                    label=0,
                    split="test",
                    domain="test_tgt",
                )
            )
        for i in range(10):
            records_eval.append(
                SQLRecord(
                    full_query=f"SELECT col FROM table WHERE val = '{i}' UNION SELECT 1, 2;",
                    label=1,
                    split="test",
                    domain="test_tgt",
                    attack_technique="union",
                )
            )
        self.eval_slice = DataSlice("synthetic_eval", records_eval)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_extract_features_matrix_shape_and_names(self) -> None:
        queries = ["SELECT 1;", "SELECT * FROM t WHERE id=1 OR 1=1;"]
        X = extract_features_matrix(queries)
        self.assertEqual(X.shape, (2, 32))
        self.assertTrue(np.all(np.isfinite(X)))

    def test_build_estimators(self) -> None:
        for family in ["logistic_regression", "random_forest", "xgboost"]:
            est, scaler = build_estimator(family, seed=42)
            self.assertIsNotNone(est)
            if family == "logistic_regression":
                self.assertIsNotNone(scaler)
            else:
                self.assertIsNone(scaler)

    def test_train_and_evaluate_logistic_regression(self) -> None:
        pipe = train_model_pipeline(
            self.train_slice, "logistic_regression", "lr_test", seed=42
        )
        self.assertEqual(pipe.model_family, "logistic_regression")
        self.assertIsNotNone(pipe.scaler)

        # Serialization
        save_path = self.temp_path / "lr.joblib"
        pipe.save(save_path)
        self.assertTrue(save_path.is_file())
        loaded_pipe = SQLDetectionPipeline.load(save_path)

        metrics = evaluate_detection_pipeline(loaded_pipe, self.eval_slice)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(metrics["total_eval_samples"], 30)

    def test_train_and_evaluate_random_forest(self) -> None:
        pipe = train_model_pipeline(
            self.train_slice, "random_forest", "rf_test", seed=42
        )
        metrics = evaluate_detection_pipeline(pipe, self.eval_slice)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["tp"] + metrics["confusion_matrix"]["fn"], 10)

    def test_train_and_evaluate_xgboost(self) -> None:
        pipe = train_model_pipeline(
            self.train_slice, "xgboost", "xgb_test", seed=42
        )
        metrics = evaluate_detection_pipeline(pipe, self.eval_slice)
        self.assertIn("technique_recalls", metrics)
        self.assertIn("union", metrics["technique_recalls"])
        self.assertEqual(metrics["technique_recalls"]["union"]["total"], 10)


if __name__ == "__main__":
    unittest.main()
