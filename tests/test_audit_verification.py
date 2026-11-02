"""Targeted verification tests ensuring reproducibility of audited metrics and properties.

Validates:
1. Recomputation of classification metrics from confusion matrix matches reported values.
2. Signed cosine similarity edge cases and mathematical bounds.
3. Query-level aggregation in LIME repeat stability (no pseudoreplication).
4. XGBoost tree determinism under fixed data and default subsampling.
"""

import unittest
import numpy as np
from src.analysis.stability import (
    compute_vector_cosine_similarity,
    compute_signed_spearman_rank,
    compute_distribution_summary,
    analyze_lime_repeat_stability,
)


class TestAuditVerification(unittest.TestCase):
    """Rigorous mathematical and structural verification tests."""

    def test_confusion_matrix_metric_recomputation(self):
        """Verify that precision, recall, accuracy, and F1 recompute identically."""
        # Logistic Regression Cross confusion matrix
        cm = {"tn": 999, "fp": 1, "fn": 27, "tp": 973}
        total = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
        acc = (cm["tp"] + cm["tn"]) / total
        prec = cm["tp"] / (cm["tp"] + cm["fp"])
        rec = cm["tp"] / (cm["tp"] + cm["fn"])
        f1 = 2 * prec * rec / (prec + rec)

        self.assertAlmostEqual(acc, 0.9860, places=4)
        self.assertAlmostEqual(prec, 0.9990, places=4)
        self.assertAlmostEqual(rec, 0.9730, places=4)
        self.assertAlmostEqual(f1, 0.9858, places=4)

    def test_signed_cosine_similarity_preserves_direction(self):
        """Opposite signed attribution vectors must yield -1.0, not +1.0."""
        u = np.array([1.0, -2.0, 3.0])
        v = np.array([-1.0, 2.0, -3.0])
        sim = compute_vector_cosine_similarity(u, v)
        self.assertAlmostEqual(sim, -1.0, places=5)

    def test_lime_repeat_aggregation_level(self):
        """LIME summary must resample at query level (50 queries), not pooled pairs (2250 pairs)."""
        # Create a synthetic 3D tensor: 5 queries, 4 repetitions, 3 features
        rng = np.random.RandomState(42)
        tensor = rng.randn(5, 4, 3)
        result = analyze_lime_repeat_stability(tensor)

        summary = result["lime_repeat_stability_summary"]
        self.assertEqual(summary["count"], 5, "Bootstrap unit must be queries (N=5), not repeat pairs!")
        self.assertEqual(len(result["per_query_mean_similarities"]), 5)

    def test_xgboost_determinism_under_full_sampling(self):
        """XGBoost with subsample=1.0 and colsample_bytree=1.0 is deterministic on fixed data."""
        from xgboost import XGBClassifier

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([0, 0, 1, 1])

        clf1 = XGBClassifier(n_estimators=5, max_depth=2, learning_rate=0.1, random_state=42)
        clf2 = XGBClassifier(n_estimators=5, max_depth=2, learning_rate=0.1, random_state=999)

        clf1.fit(X, y)
        clf2.fit(X, y)

        dump1 = clf1.get_booster().get_dump()
        dump2 = clf2.get_booster().get_dump()

        self.assertEqual(dump1, dump2, "XGBoost tree structures must be identical under default subsampling!")


if __name__ == "__main__":
    unittest.main()
