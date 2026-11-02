"""Visualization module for producing publication-quality figures."""

import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.features.manual_li import FEATURE_NAMES_32

logger = logging.getLogger(__name__)

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def plot_detection_metrics_comparison(
    metrics_cross: dict[str, Any],
    metrics_within: dict[str, Any],
    model_family: str,
    output_path: Path,
) -> Path:
    """Plot bar comparison of Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]

    vals_cross = [metrics_cross.get(k, 0.0) for k in metric_keys]
    vals_within = [metrics_within.get(k, 0.0) for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, vals_cross, width, label="Cross-Domain (M_cross)", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x + width/2, vals_within, width, label="Within-Domain (M_within)", color="#c44e52", alpha=0.9)

    ax.set_ylabel("Score")
    ax.set_title(f"Detection Performance: Cross-Domain vs. Within-Domain ({model_family.replace('_', ' ').title()})")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")

    # Add values on top of bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved detection comparison plot to {output_path}")
    return output_path


def plot_decoupling_strata(
    decoupling_data: dict[str, Any],
    model_family: str,
    output_path: Path,
) -> Path:
    """Plot explanation stability across all prediction-correctness strata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cat1 = decoupling_data.get("category_1_both_correct", decoupling_data.get("strata_1_concordant_correct", {}))
    cat2 = decoupling_data.get("category_2_cross_only_correct", {})
    cat3 = decoupling_data.get("category_3_within_only_correct", decoupling_data.get("strata_2_accuracy_degraded", {}))
    cat4 = decoupling_data.get("category_4_both_incorrect", {})

    candidates = [
        ("Cat 1:\nBoth Correct", cat1, "#4c72b0"),
        ("Cat 2:\nCross Only Correct", cat2, "#55a868"),
        ("Cat 3:\nWithin Only Correct", cat3, "#dd8452"),
        ("Cat 4:\nBoth Incorrect", cat4, "#c44e52"),
    ]

    active = [(name, d, col) for name, d, col in candidates if d.get("count", 0) > 0]
    if not active:
        active = candidates[:2]

    strata_names = [item[0] for item in active]
    means = [item[1].get("mean", 0.0) for item in active]
    medians = [item[1].get("median", 0.0) for item in active]
    counts = [item[1].get("count", 0) for item in active]
    colors = [item[2] for item in active]

    err_low = [max(0.0, item[1].get("mean", 0.0) - item[1].get("ci_95_lower", item[1].get("mean", 0.0))) for item in active]
    err_high = [max(0.0, item[1].get("ci_95_upper", item[1].get("mean", 0.0)) - item[1].get("mean", 0.0)) for item in active]

    fig, ax = plt.subplots(figsize=(max(6, len(active) * 2.2), 5))
    x = np.arange(len(strata_names))
    width = 0.45

    ax.bar(
        x,
        means,
        width,
        yerr=[err_low, err_high],
        capsize=6,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_ylabel("Cosine Similarity (M_cross vs. M_within)")
    ax.set_title(f"Decoupling Analysis: Attribution Stability by Prediction Strata\n({model_family.replace('_', ' ').title()})")
    ax.set_xticks(x)
    ax.set_xticklabels(strata_names)
    ax.set_ylim(-0.1, 1.15)

    for i, (m, med, n) in enumerate(zip(means, medians, counts)):
        ax.annotate(
            f"Mean: {m:.3f}\nMed: {med:.3f}\n(N={n})",
            xy=(i, max(m, 0.08)),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved decoupling strata plot to {output_path}")
    return output_path


def plot_top_feature_importances(
    attr_cross: np.ndarray,
    attr_within: np.ndarray,
    model_family: str,
    output_path: Path,
    top_k: int = 10,
) -> Path:
    """Plot comparison of top K mean absolute feature attributions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_abs_cross = np.mean(np.abs(attr_cross), axis=0)
    mean_abs_within = np.mean(np.abs(attr_within), axis=0)

    # Rank by overall importance
    combined_imp = mean_abs_cross + mean_abs_within
    top_indices = np.argsort(combined_imp)[::-1][:top_k]

    top_names = [FEATURE_NAMES_32[i] for i in top_indices]
    top_cross = mean_abs_cross[top_indices]
    top_within = mean_abs_within[top_indices]

    y = np.arange(top_k)
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y - height/2, top_cross, height, label="Cross-Domain (M_cross)", color="#2b5c8f", alpha=0.9)
    ax.barh(y + height/2, top_within, height, label="Within-Domain (M_within)", color="#c44e52", alpha=0.9)

    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_title(f"Top {top_k} Features Comparison ({model_family.replace('_', ' ').title()})")
    ax.set_yticks(y)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved top feature importances plot to {output_path}")
    return output_path
