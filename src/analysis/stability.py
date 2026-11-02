"""Explanation stability and similarity analysis module.

Implements:
1. Signed vector cosine similarity with robust zero-vector handling.
2. Signed Spearman rank correlation preserving attribution direction.
3. Distribution summaries with bootstrap 95% confidence intervals.
4. LIME perturbation stability (mean pairwise similarity across 10 runs).
5. Repeat-fit model stability (estimator stochasticity isolation).
6. Cross-domain vs. within-domain explanation divergence.
7. Decoupling analysis across performance strata.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def compute_vector_cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two 1D signed attribution vectors.

    Edge case handling:
    - If both u and v are all zeros: returns 1.0 (perfect agreement on zero impact).
    - If one vector is all zeros and the other non-zero: returns 0.0.
    - Otherwise returns standard cosine similarity clamped to [-1.0, 1.0].
    """
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))

    if norm_u == 0.0 and norm_v == 0.0:
        return 1.0
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0

    dot = float(np.dot(u, v))
    cos = dot / (norm_u * norm_v)
    return float(np.clip(cos, -1.0, 1.0))


def compute_signed_spearman_rank(u: np.ndarray, v: np.ndarray) -> float:
    """Compute Spearman rank correlation on signed attribution vectors.

    Preserves direction: large positive features have highest ranks;
    large negative features have lowest ranks.
    """
    if np.all(u == u[0]) or np.all(v == v[0]):
        return 0.0
    corr, _ = spearmanr(u, v)
    if np.isnan(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def compute_distribution_summary(values: Sequence[float], n_bootstrap: int = 1000, seed: int = 42) -> dict[str, float]:
    """Compute mean, std (ddof=0), sample_std (ddof=1), median, IQR, and 95% bootstrap CI."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "sample_std": 0.0,
            "sem": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "ci_95_lower": 0.0,
            "ci_95_upper": 0.0,
        }

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=0))
    sample_std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sem_val = float(sample_std_val / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    median_val = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25, 75])
    iqr_val = float(q75 - q25)

    if len(arr) > 1:
        rng = np.random.RandomState(seed)
        boot_means = [
            np.mean(arr[rng.choice(len(arr), size=len(arr), replace=True)])
            for _ in range(n_bootstrap)
        ]
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
    else:
        ci_lower = mean_val
        ci_upper = mean_val

    return {
        "count": len(arr),
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "sample_std": round(sample_std_val, 4),
        "sem": round(sem_val, 4),
        "median": round(median_val, 4),
        "iqr": round(iqr_val, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
    }


def analyze_cross_vs_within_stability(
    attr_cross: np.ndarray,
    attr_within: np.ndarray,
    y_true: Sequence[int],
    y_pred_cross: Sequence[int],
    y_pred_within: Sequence[int],
    techniques: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Analyze explanation differences between M_cross and M_within on identical queries.

    Performs exhaustive 4-strata decoupling analysis:
    - Overall evaluation set
    - Category 1: Both Correct (y_cross == y_true and y_within == y_true)
    - Category 2: Cross Only Correct (y_cross == y_true and y_within != y_true)
    - Category 3: Within Only Correct (y_within == y_true and y_cross != y_true)
    - Category 4: Both Incorrect (y_cross != y_true and y_within != y_true)
    - Label strata: Benign vs. Attack
    - Technique-specific strata
    """
    n_queries = len(attr_cross)
    cos_sims = np.zeros(n_queries, dtype=np.float64)
    rank_sims = np.zeros(n_queries, dtype=np.float64)

    for i in range(n_queries):
        cos_sims[i] = compute_vector_cosine_similarity(attr_cross[i], attr_within[i])
        rank_sims[i] = compute_signed_spearman_rank(attr_cross[i], attr_within[i])

    y_t = np.asarray(y_true)
    y_c = np.asarray(y_pred_cross)
    y_w = np.asarray(y_pred_within)

    # Exhaustive and mutually exclusive 4-category partition
    mask_both_correct = (y_c == y_t) & (y_w == y_t)
    mask_cross_only = (y_c == y_t) & (y_w != y_t)
    mask_within_only = (y_w == y_t) & (y_c != y_t)
    mask_both_wrong = (y_c != y_t) & (y_w != y_t)

    mask_benign = (y_t == 0)
    mask_attack = (y_t == 1)

    overall_cos = compute_distribution_summary(cos_sims)
    overall_rank = compute_distribution_summary(rank_sims)

    strata_both_correct = compute_distribution_summary(cos_sims[mask_both_correct])
    strata_cross_only = compute_distribution_summary(cos_sims[mask_cross_only])
    strata_within_only = compute_distribution_summary(cos_sims[mask_within_only])
    strata_both_wrong = compute_distribution_summary(cos_sims[mask_both_wrong])

    strata_benign = compute_distribution_summary(cos_sims[mask_benign])
    strata_attack = compute_distribution_summary(cos_sims[mask_attack])

    technique_strata: dict[str, Any] = {}
    if techniques is not None:
        tech_arr = np.asarray(techniques)
        for tech in np.unique(tech_arr):
            if tech:
                mask_tech = (tech_arr == tech)
                technique_strata[str(tech)] = compute_distribution_summary(cos_sims[mask_tech])

    return {
        "overall_cosine_similarity": overall_cos,
        "overall_rank_correlation": overall_rank,
        "decoupling_analysis": {
            "category_1_both_correct": strata_both_correct,
            "category_2_cross_only_correct": strata_cross_only,
            "category_3_within_only_correct": strata_within_only,
            "category_4_both_incorrect": strata_both_wrong,
            # Backward-compatible aliases
            "strata_1_concordant_correct": strata_both_correct,
            "strata_2_accuracy_degraded": strata_within_only,
            "delta_mean_similarity": round(strata_both_correct["mean"] - strata_within_only["mean"], 4)
            if strata_both_correct["count"] and strata_within_only["count"] else None,
        },
        "label_strata": {
            "benign": strata_benign,
            "attack": strata_attack,
        },
        "technique_strata": technique_strata,
        "per_query_cosine_similarities": cos_sims.tolist(),
        "per_query_rank_correlations": rank_sims.tolist(),
    }


def analyze_lime_repeat_stability(lime_tensor: np.ndarray) -> dict[str, Any]:
    """Analyze LIME perturbation stability across 10 repeated runs for the same query.

    Args:
        lime_tensor: Shape (n_queries, n_repetitions, 32).

    Returns:
        Summary of pairwise cosine similarities across runs.
    """
    n_queries, n_reps, n_feats = lime_tensor.shape
    query_mean_similarities = np.zeros(n_queries, dtype=np.float64)

    for q in range(n_queries):
        pairwise_sims = []
        for i in range(n_reps):
            for j in range(i + 1, n_reps):
                sim = compute_vector_cosine_similarity(lime_tensor[q, i], lime_tensor[q, j])
                pairwise_sims.append(sim)
        query_mean_similarities[q] = np.mean(pairwise_sims) if pairwise_sims else 1.0

    summary = compute_distribution_summary(query_mean_similarities)
    return {
        "lime_repeat_stability_summary": summary,
        "per_query_mean_similarities": query_mean_similarities.tolist(),
    }


def analyze_repeat_fit_stability(attr_fit1: np.ndarray, attr_fit2: np.ndarray) -> dict[str, Any]:
    """Analyze repeat-fit model stability (estimator stochasticity isolation)."""
    n_queries = len(attr_fit1)
    cos_sims = np.zeros(n_queries, dtype=np.float64)
    for i in range(n_queries):
        cos_sims[i] = compute_vector_cosine_similarity(attr_fit1[i], attr_fit2[i])

    summary = compute_distribution_summary(cos_sims)
    return {
        "repeat_fit_stability_summary": summary,
        "per_query_similarities": cos_sims.tolist(),
    }
