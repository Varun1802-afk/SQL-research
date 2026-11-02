"""Targeted tests auditing template-disjoint feasibility in Domain D.

Validates:
1. The locked 2,000-query evaluation set spans all 29 attack templates in Domain D.
2. The number of remaining attack templates disjoint from the evaluation set is strictly 0.
3. The number of eligible attack records with valid template IDs disjoint from the evaluation set is strictly 0.
4. Single-class supervised training limitations (demonstrating why binary classification is infeasible).
"""

import unittest
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data.loader import load_csv_records, sample_stratified_test


class TestTemplateDisjointFeasibility(unittest.TestCase):
    """Tests verifying the structural impossibility of template-disjoint training in Domain D."""

    @classmethod
    def setUpClass(cls):
        data_path = Path("data/d-d.csv")
        if not data_path.exists():
            raise unittest.SkipTest("data/d-d.csv not found locally.")

        cls.target_records = load_csv_records(data_path, domain="d", max_rows=150_000)
        cls.target_test_pool = [r for r in cls.target_records if r.original_split == "test" or r.split == "test"]
        cls.eval_slice = sample_stratified_test(cls.target_test_pool, n_total=2000, attack_ratio=0.50, seed=42)

    def test_eval_set_template_saturation(self):
        """Evaluation set must contain exactly 29 unique attack templates."""
        eval_attack_templates = set(r.query_template_id for r in self.eval_slice if r.label == 1 and r.query_template_id)
        self.assertEqual(len(eval_attack_templates), 29)

    def test_zero_disjoint_attack_templates_remaining(self):
        """All attack templates in the pool appear in the evaluation set (zero disjoint attack templates)."""
        pool_attack_templates = set(r.query_template_id for r in self.target_test_pool if r.label == 1 and r.query_template_id)
        eval_templates = set(self.eval_slice.template_ids)
        disjoint_attack_templates = pool_attack_templates - eval_templates

        self.assertEqual(
            len(disjoint_attack_templates),
            0,
            f"Expected 0 disjoint attack templates remaining, found {disjoint_attack_templates}"
        )

    def test_zero_disjoint_attack_records_with_valid_template(self):
        """No attack records with a valid template ID exist in the pool that do not share a template with the evaluation set."""
        eval_templates = set(self.eval_slice.template_ids)
        disjoint_attack_records = [
            r for r in self.target_test_pool
            if r.label == 1 and r.query_template_id and r.query_template_id not in eval_templates
        ]
        self.assertEqual(len(disjoint_attack_records), 0)

    def test_single_class_logistic_regression_raises_value_error(self):
        """Fitting LogisticRegression on single-class data raises ValueError."""
        X = np.ones((10, 5))
        y = np.zeros(10, dtype=int)
        with self.assertRaises(ValueError):
            LogisticRegression().fit(X, y)

    def test_single_class_roc_auc_undefined(self):
        """Evaluating ROC-AUC on single-class data issues UndefinedMetricWarning and returns NaN."""
        import warnings
        from sklearn.metrics._ranking import UndefinedMetricWarning

        y_true = np.zeros(10, dtype=int)
        y_pred = np.zeros(10, dtype=float)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            score = roc_auc_score(y_true, y_pred)
            self.assertTrue(np.isnan(score))
            self.assertTrue(any(issubclass(item.category, UndefinedMetricWarning) for item in w))


if __name__ == "__main__":
    unittest.main()
