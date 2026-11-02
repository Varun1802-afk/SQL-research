"""LIME explanation generator with repeated runs for explanation stability assessment.

Generates 10 independent repetitions per query using LimeTabularExplainer.
Outputs signed feature weights corresponding to class 1 (SQL injection).
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.features.manual_li import FEATURE_NAMES_32
from src.models.train import SQLDetectionPipeline

logger = logging.getLogger(__name__)


def generate_lime_attributions_with_repetitions(
    pipeline: SQLDetectionPipeline,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    n_repetitions: int = 10,
    base_seed: int = 42,
    feature_names: Optional[list[str]] = None,
    num_features: int = 32,
    num_samples: int = 1000,
) -> np.ndarray:
    """Generate repeated LIME explanations for a subset of queries.

    Args:
        pipeline: Trained SQLDetectionPipeline.
        X_train: Background training data for LIME distribution statistics (unscaled).
        X_explain: Queries to explain (unscaled).
        n_repetitions: Number of repeated runs per query (default 10).
        base_seed: Starting random seed for repetitions.
        feature_names: Canonical feature names (32).
        num_features: Number of features to explain (32).
        num_samples: Number of perturbation samples per explanation.

    Returns:
        np.ndarray of shape (n_queries, n_repetitions, 32) containing signed feature weights.
    """
    fnames = feature_names or FEATURE_NAMES_32
    n_queries = len(X_explain)
    out_matrix = np.zeros((n_queries, n_repetitions, len(fnames)), dtype=np.float64)

    # Prediction wrapper: LIME passes perturbed raw feature vectors
    # Pipeline handles scaling (if present) and outputs probability [P(y=0), P(y=1)]
    predict_fn = pipeline.predict_proba

    logger.info(
        f"Generating LIME explanations for {n_queries} queries x {n_repetitions} repetitions "
        f"({n_queries * n_repetitions} total explanations)..."
    )
    start_total = time.time()

    # For each repetition seed, instantiate an explainer with that random state
    for rep in range(n_repetitions):
        seed = base_seed + rep
        explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=fnames,
            class_names=["benign", "attack"],
            mode="classification",
            random_state=seed,
            discretize_continuous=False,  # Keep continuous features for consistent attribution space
            sample_around_instance=True,   # Sample continuous features centered around x_inst to prevent kernel distance underflow
        )

        rep_start = time.time()
        for q_idx in range(n_queries):
            x_inst = X_explain[q_idx]
            exp = explainer.explain_instance(
                data_row=x_inst,
                predict_fn=predict_fn,
                labels=(1,),  # Explain attack class
                num_features=num_features,
                num_samples=num_samples,
            )

            # Robust index-based mapping using exp.as_map()[1]
            if 1 in exp.as_map():
                for f_idx, weight in exp.as_map()[1]:
                    if 0 <= f_idx < len(fnames):
                        out_matrix[q_idx, rep, f_idx] = float(weight)
            else:
                # Fallback to feature name mapping
                weights_map = dict(exp.as_list(label=1))
                for f_idx, fname in enumerate(fnames):
                    out_matrix[q_idx, rep, f_idx] = float(weights_map.get(fname, 0.0))

        logger.info(f"Completed LIME repetition {rep+1}/{n_repetitions} (seed={seed}) in {time.time() - rep_start:.2f}s.")

    total_time = time.time() - start_total
    logger.info(f"All {n_queries * n_repetitions} LIME explanations finished in {total_time:.2f}s.")
    return out_matrix


def save_lime_attributions_to_csv(
    lime_tensor: np.ndarray,
    queries: Sequence[str],
    labels: Sequence[int],
    preds: Sequence[int],
    output_path: Path,
    base_seed: int = 42,
    feature_names: Optional[list[str]] = None,
) -> Path:
    """Save repeated LIME explanations to CSV."""
    fnames = feature_names or FEATURE_NAMES_32
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_queries, n_repetitions, n_feats = lime_tensor.shape
    rows = []

    for q_idx in range(n_queries):
        for rep in range(n_repetitions):
            seed = base_seed + rep
            row_dict: dict[str, Any] = {
                "query_index": q_idx,
                "repetition_idx": rep,
                "repetition_seed": seed,
                "true_label": int(labels[q_idx]),
                "predicted_label": int(preds[q_idx]),
                "sql_query": queries[q_idx],
            }
            for f_idx, fname in enumerate(fnames):
                row_dict[fname] = float(lime_tensor[q_idx, rep, f_idx])
            rows.append(row_dict)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved LIME repetitions table ({len(df)} rows) to {output_path}")
    return output_path
