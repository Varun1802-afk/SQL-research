"""SHAP explanation generator for SQL injection detection models.

Supports TreeExplainer for tree ensembles (Random Forest, XGBoost) and
LinearExplainer for linear models (Logistic Regression).
Outputs signed per-feature attributions aligned to class 1 (SQL injection).
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import shap

from src.features.manual_li import FEATURE_NAMES_32
from src.models.train import SQLDetectionPipeline

logger = logging.getLogger(__name__)


def generate_shap_attributions(
    pipeline: SQLDetectionPipeline,
    X_explain: np.ndarray,
    X_background: Optional[np.ndarray] = None,
    feature_names: Optional[list[str]] = None,
) -> np.ndarray:
    """Generate signed SHAP attribution matrix of shape (n_queries, 32).

    All values correspond to the contribution toward predicting class 1 (attack).

    Args:
        pipeline: Trained SQLDetectionPipeline.
        X_explain: Feature matrix of queries to explain (unscaled).
        X_background: Optional background reference samples (unscaled) for linear explainer.
        feature_names: Column names matching feature dimensions.

    Returns:
        np.ndarray of shape (len(X_explain), len(feature_names)) with finite signed values.
    """
    fnames = feature_names or FEATURE_NAMES_32
    estimator = pipeline.estimator
    family = pipeline.model_family

    logger.info(f"Generating SHAP attributions for {len(X_explain)} queries using {family}...")
    start_time = time.time()

    if family in ("random_forest", "xgboost"):
        # For tree ensembles, no background dataset is needed for exact TreeSHAP
        explainer = shap.TreeExplainer(estimator)
        # TreeExplainer expects features in the scale the tree was fitted on
        X_trans = pipeline.transform(X_explain)
        raw_values = explainer.shap_values(X_trans)

        if isinstance(raw_values, list):
            # scikit-learn binary RF often returns [shap_class_0, shap_class_1]
            if len(raw_values) == 2:
                shap_vals = np.array(raw_values[1], dtype=np.float64)
            else:
                shap_vals = np.array(raw_values[0], dtype=np.float64)
        elif isinstance(raw_values, np.ndarray):
            if raw_values.ndim == 3 and raw_values.shape[-1] == 2:
                # Shape (N, n_features, 2)
                shap_vals = raw_values[:, :, 1]
            else:
                # Shape (N, n_features) - e.g. XGBoost margin
                shap_vals = raw_values
        else:
            # shap.Explanation object in newer versions
            if hasattr(raw_values, "values"):
                vals = raw_values.values
                if vals.ndim == 3 and vals.shape[-1] == 2:
                    shap_vals = vals[:, :, 1]
                else:
                    shap_vals = vals
            else:
                shap_vals = np.array(raw_values, dtype=np.float64)

    elif family == "logistic_regression":
        # LinearExplainer uses transformed background dataset
        if X_background is not None:
            bg_trans = pipeline.transform(X_background)
            # Use k-means or subsample if background is large
            if len(bg_trans) > 100:
                rng = np.random.RandomState(pipeline.seed)
                bg_trans = bg_trans[rng.choice(len(bg_trans), 100, replace=False)]
        else:
            bg_trans = np.zeros((1, len(fnames)), dtype=np.float64)

        X_trans = pipeline.transform(X_explain)
        explainer = shap.LinearExplainer(estimator, bg_trans)
        raw_values = explainer.shap_values(X_trans)

        if isinstance(raw_values, list) and len(raw_values) == 2:
            shap_vals = np.array(raw_values[1], dtype=np.float64)
        elif hasattr(raw_values, "values"):
            shap_vals = np.array(raw_values.values, dtype=np.float64)
        else:
            shap_vals = np.array(raw_values, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported model family for SHAP: {family}")

    shap_vals = np.asarray(shap_vals, dtype=np.float64)

    # Sanity checks
    if shap_vals.shape != (len(X_explain), len(fnames)):
        raise ValueError(
            f"SHAP attribution shape mismatch: expected ({len(X_explain)}, {len(fnames)}), got {shap_vals.shape}"
        )
    if not np.all(np.isfinite(shap_vals)):
        logger.warning("Non-finite values detected in SHAP attributions! Imputing zeros.")
        shap_vals = np.nan_to_num(shap_vals, nan=0.0, posinf=0.0, neginf=0.0)

    elapsed = time.time() - start_time
    logger.info(f"SHAP explanation completed in {elapsed:.2f}s for {len(X_explain)} queries.")
    return shap_vals


def save_shap_attributions_to_csv(
    shap_matrix: np.ndarray,
    queries: Sequence[str],
    labels: Sequence[int],
    preds: Sequence[int],
    output_path: Path,
    feature_names: Optional[list[str]] = None,
) -> Path:
    """Save signed SHAP attributions and query metadata to CSV."""
    fnames = feature_names or FEATURE_NAMES_32
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_meta = pd.DataFrame({
        "query_index": list(range(len(queries))),
        "true_label": list(labels),
        "predicted_label": list(preds),
        "sql_query": list(queries),
    })

    df_attr = pd.DataFrame(shap_matrix, columns=fnames)
    df_combined = pd.concat([df_meta, df_attr], axis=1)

    df_combined.to_csv(output_path, index=False)
    logger.info(f"Saved SHAP attributions table ({len(df_combined)} rows) to {output_path}")
    return output_path
