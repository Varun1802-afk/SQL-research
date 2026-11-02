"""Unit tests for SHAP, LIME, similarity metrics, and stability analysis."""

import unittest
import numpy as np

from src.analysis.stability import (
    analyze_cross_vs_within_stability,
    analyze_lime_repeat_stability,
    compute_distribution_summary,
    compute_signed_spearman_rank,
    compute_vector_cosine_similarity,
)
from src.data.loader import DataSlice, SQLRecord
from src.explainers.lime_explainer import generate_lime_attributions_with_repetitions
from src.explainers.shap_explainer import generate_shap_attributions
from src.features.manual_li import FEATURE_NAMES_32
from src.models.train import train_model_pipeline


class TestExplainersAndStability(unittest.TestCase):
    """Test SHAP, LIME, similarity edge cases, and decoupling analysis."""

    def setUp(self) -> None:
        records_train = []
        for i in range(50):
            records_train.append(
                SQLRecord(
                    full_query=f"SELECT col FROM t WHERE id = {i};",
                    label=0,
                    split="train",
                )
            )
        for i in range(30):
            records_train.append(
                SQLRecord(
                    full_query=f"SELECT col FROM t WHERE id = {i} OR 1=1;",
                    label=1,
                    split="train",
                    attack_technique="boolean",
                )
            )
        self.train_slice = DataSlice("train_exp", records_train)
        self.pipe = train_model_pipeline(self.train_slice, "xgboost", "xgb_exp_test", seed=42)

    def test_vector_cosine_similarity_edge_cases(self) -> None:
        # Both zero vectors
        z1 = np.zeros(32)
        z2 = np.zeros(32)
        self.assertEqual(compute_vector_cosine_similarity(z1, z2), 1.0)

        # One zero, one non-zero
        nz = np.ones(32)
        self.assertEqual(compute_vector_cosine_similarity(z1, nz), 0.0)
        self.assertEqual(compute_vector_cosine_similarity(nz, z1), 0.0)

        # Identical vectors
        self.assertAlmostEqual(compute_vector_cosine_similarity(nz, nz), 1.0)

        # Opposite vectors
        self.assertAlmostEqual(compute_vector_cosine_similarity(nz, -nz), -1.0)

    def test_signed_spearman_rank_correlation(self) -> None:
        u = np.arange(32, dtype=np.float64)
        v = np.arange(32, dtype=np.float64)
        self.assertAlmostEqual(compute_signed_spearman_rank(u, v), 1.0)

        v_rev = -u
        self.assertAlmostEqual(compute_signed_spearman_rank(u, v_rev), -1.0)

    def test_distribution_summary_with_bootstrap(self) -> None:
        vals = [0.8, 0.85, 0.9, 0.88, 0.92]
        summary = compute_distribution_summary(vals, n_bootstrap=100)
        self.assertIn("mean", summary)
        self.assertIn("median", summary)
        self.assertIn("iqr", summary)
        self.assertIn("ci_95_lower", summary)
        self.assertIn("ci_95_upper", summary)
        self.assertLessEqual(summary["ci_95_lower"], summary["mean"])
        self.assertGreaterEqual(summary["ci_95_upper"], summary["mean"])

    def test_shap_attributions_generation_and_shape(self) -> None:
        from src.models.train import extract_features_matrix
        queries = ["SELECT 1;", "SELECT * FROM users WHERE name='admin' OR 1=1;"]
        X_explain = extract_features_matrix(queries)

        shap_vals = generate_shap_attributions(self.pipe, X_explain)
        self.assertEqual(shap_vals.shape, (2, 32))
        self.assertTrue(np.all(np.isfinite(shap_vals)))

    def test_lime_attributions_repetition_tensor(self) -> None:
        from src.models.train import extract_features_matrix
        X_train = extract_features_matrix(self.train_slice.queries)
        queries = ["SELECT * FROM t WHERE id=1 OR 1=1;"]
        X_explain = extract_features_matrix(queries)

        lime_tensor = generate_lime_attributions_with_repetitions(
            self.pipe,
            X_train=X_train,
            X_explain=X_explain,
            n_repetitions=3,
            base_seed=42,
            num_samples=200,
        )
        self.assertEqual(lime_tensor.shape, (1, 3, 32))
        # Ensure attributions are non-trivial (not all zero or underflowed)
        self.assertGreater(float(np.max(np.abs(lime_tensor))), 1e-4)

        stability = analyze_lime_repeat_stability(lime_tensor)
        self.assertIn("lime_repeat_stability_summary", stability)
        self.assertIn("mean", stability["lime_repeat_stability_summary"])
        self.assertIn("sample_std", stability["lime_repeat_stability_summary"])

    def test_analyze_cross_vs_within_stability_decoupling(self) -> None:
        # 10 queries, 32 features
        attr_cross = np.random.RandomState(42).randn(10, 32)
        attr_within = attr_cross + np.random.RandomState(43).randn(10, 32) * 0.1

        y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y_cross = [0, 0, 0, 1, 0, 1, 1, 0, 1, 1]
        y_within = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        res = analyze_cross_vs_within_stability(
            attr_cross=attr_cross,
            attr_within=attr_within,
            y_true=y_true,
            y_pred_cross=y_cross,
            y_pred_within=y_within,
        )
        self.assertIn("overall_cosine_similarity", res)
        self.assertIn("decoupling_analysis", res)
        decoupling = res["decoupling_analysis"]
        self.assertIn("category_1_both_correct", decoupling)
        self.assertIn("category_2_cross_only_correct", decoupling)
        self.assertIn("category_3_within_only_correct", decoupling)
        self.assertIn("category_4_both_incorrect", decoupling)

        # Mutually exclusive and exhaustive: counts must sum to total queries
        total_strata_count = (
            decoupling["category_1_both_correct"]["count"]
            + decoupling["category_2_cross_only_correct"]["count"]
            + decoupling["category_3_within_only_correct"]["count"]
            + decoupling["category_4_both_incorrect"]["count"]
        )
        self.assertEqual(total_strata_count, 10)


if __name__ == "__main__":
    unittest.main()
