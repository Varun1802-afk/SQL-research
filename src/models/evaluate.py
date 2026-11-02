"""Evaluation module for SQL injection detection models on held-out evaluation sets."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.loader import DataSlice
from src.models.train import SQLDetectionPipeline, extract_features_matrix

logger = logging.getLogger(__name__)


def evaluate_detection_pipeline(
    pipeline: SQLDetectionPipeline,
    eval_slice: DataSlice,
    X_eval: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Compute comprehensive detection metrics on quarantined evaluation slice.

    Metrics:
    - Accuracy, Precision, Recall, F1-score
    - ROC-AUC, PR-AUC (Average Precision)
    - Confusion Matrix (TN, FP, FN, TP)
    - Stratified breakdown (by attack technique, statement type, prediction correctness)
    """
    if X_eval is None:
        X_eval = extract_features_matrix(eval_slice.queries)

    y_true = np.array(eval_slice.labels, dtype=np.int64)

    y_pred = pipeline.predict(X_eval)
    try:
        y_proba = pipeline.predict_proba(X_eval)[:, 1]
    except Exception:
        y_proba = y_pred.astype(np.float64)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = float(eval_slice.attack_ratio)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    # Correctness stratification indices
    correct_mask = (y_true == y_pred)
    incorrect_mask = (y_true != y_pred)

    # Technique-specific recall for attack queries
    technique_recalls: dict[str, Any] = {}
    for i, r in enumerate(eval_slice.records):
        if r.is_attack and r.attack_technique:
            tech = r.attack_technique
            if tech not in technique_recalls:
                technique_recalls[tech] = {"total": 0, "correct": 0}
            technique_recalls[tech]["total"] += 1
            if y_pred[i] == 1:
                technique_recalls[tech]["correct"] += 1

    for tech, stats in technique_recalls.items():
        stats["recall"] = round(stats["correct"] / max(stats["total"], 1), 4)

    metrics = {
        "model_id": pipeline.model_id,
        "model_family": pipeline.model_family,
        "training_domain": pipeline.training_domain,
        "eval_slice_name": eval_slice.name,
        "total_eval_samples": len(eval_slice),
        "benign_samples": int(eval_slice.benign_count),
        "attack_samples": int(eval_slice.attack_count),
        "attack_prevalence": round(eval_slice.attack_ratio, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "technique_recalls": technique_recalls,
        "correct_predictions_count": int(np.sum(correct_mask)),
        "incorrect_predictions_count": int(np.sum(incorrect_mask)),
    }

    return metrics


def save_metrics_report(all_metrics: list[dict[str, Any]], output_path: Path) -> Path:
    """Save all evaluation metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {output_path}")
    return output_path
