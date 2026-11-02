"""Sensitivity analysis module evaluating protocol variations and split-provenance effects.

Compares:
1. Primary Provenance-Matched Protocol (both classes from comparable test partition stream).
2. Unbalanced / Split-Confounded Protocol (benign from official train, attack from official test).
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_protocol_sensitivity_comparison(
    primary_metrics: dict[str, Any],
    sensitivity_metrics: dict[str, Any],
    primary_stability: dict[str, Any],
    sensitivity_stability: dict[str, Any],
    model_family: str = "xgboost",
) -> dict[str, Any]:
    """Compare primary vs sensitivity protocol outcomes to evaluate robustness of conclusions."""
    primary_det = primary_metrics.get("detection", {})
    sens_det = sensitivity_metrics.get("detection", {})

    primary_stab = primary_stability.get("cross_vs_within", {})
    sens_stab = sensitivity_stability.get("cross_vs_within", {})

    # Extract detection differences
    f1_cross_prim = primary_det.get("cross_domain", {}).get("f1_score", 0.0)
    f1_within_prim = primary_det.get("within_domain", {}).get("f1_score", 0.0)
    delta_f1_prim = round(f1_cross_prim - f1_within_prim, 4)

    f1_cross_sens = sens_det.get("cross_domain", {}).get("f1_score", 0.0)
    f1_within_sens = sens_det.get("within_domain", {}).get("f1_score", 0.0)
    delta_f1_sens = round(f1_cross_sens - f1_within_sens, 4)

    # Extract attribution stability differences
    mean_cos_prim = primary_stab.get("overall_cosine_similarity", {}).get("mean", 0.0)
    mean_cos_sens = sens_stab.get("overall_cosine_similarity", {}).get("mean", 0.0)

    # Decoupling strata comparison
    decoupling_prim = primary_stab.get("decoupling_analysis", {})
    decoupling_sens = sens_stab.get("decoupling_analysis", {})

    strata1_prim = decoupling_prim.get("strata_1_concordant_correct", {}).get("mean", 0.0)
    strata2_prim = decoupling_prim.get("strata_2_accuracy_degraded", {}).get("mean", 0.0)
    delta_strata_prim = round(strata1_prim - strata2_prim, 4) if strata1_prim and strata2_prim else None

    strata1_sens = decoupling_sens.get("strata_1_concordant_correct", {}).get("mean", 0.0)
    strata2_sens = decoupling_sens.get("strata_2_accuracy_degraded", {}).get("mean", 0.0)
    delta_strata_sens = round(strata1_sens - strata2_sens, 4) if strata1_sens and strata2_sens else None

    comparison = {
        "model_family": model_family,
        "detection_performance": {
            "primary_provenance_matched": {
                "f1_cross": f1_cross_prim,
                "f1_within": f1_within_prim,
                "delta_f1": delta_f1_prim,
            },
            "sensitivity_unbalanced": {
                "f1_cross": f1_cross_sens,
                "f1_within": f1_within_sens,
                "delta_f1": delta_f1_sens,
            },
        },
        "explanation_stability": {
            "primary_provenance_matched": {
                "mean_cosine_similarity": mean_cos_prim,
                "strata_1_concordant_correct": strata1_prim,
                "strata_2_accuracy_degraded": strata2_prim,
                "delta_strata_stability": delta_strata_prim,
            },
            "sensitivity_unbalanced": {
                "mean_cosine_similarity": mean_cos_sens,
                "strata_1_concordant_correct": strata1_sens,
                "strata_2_accuracy_degraded": strata2_sens,
                "delta_strata_stability": delta_strata_sens,
            },
        },
        "robustness_verdict": {
            "stability_preserved_across_protocols": bool(abs(mean_cos_prim - mean_cos_sens) < 0.25),
            "decoupling_direction_consistent": bool(
                (delta_strata_prim is not None and delta_strata_sens is not None)
                and ((delta_strata_prim > 0 and delta_strata_sens > 0) or (delta_strata_prim < 0 and delta_strata_sens < 0))
            ),
        },
    }

    return comparison


def save_sensitivity_report(comparison: dict[str, Any], output_path: Path) -> Path:
    """Save sensitivity analysis report to JSON and Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    md_lines = [
        "# Protocol Sensitivity Analysis Report",
        "",
        "**Model Evaluated:** " + comparison.get("model_family", "").replace("_", " ").title(),
        "",
        "---",
        "",
        "## 1. Detection Performance Comparison",
        "",
        "| Protocol | M_cross F1 | M_within F1 | Δ F1 (Domain Shift Degradation) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Primary (Provenance-Matched)** | {comparison['detection_performance']['primary_provenance_matched']['f1_cross']:.4f} | {comparison['detection_performance']['primary_provenance_matched']['f1_within']:.4f} | {comparison['detection_performance']['primary_provenance_matched']['delta_f1']:+.4f} |",
        f"| **Sensitivity (Unbalanced / Split-Confounded)** | {comparison['detection_performance']['sensitivity_unbalanced']['f1_cross']:.4f} | {comparison['detection_performance']['sensitivity_unbalanced']['f1_within']:.4f} | {comparison['detection_performance']['sensitivity_unbalanced']['delta_f1']:+.4f} |",
        "",
        "---",
        "",
        "## 2. Explanation Attribution Stability Comparison",
        "",
        "| Protocol | Mean Cosine Sim | Strata 1 (Concordant) | Strata 2 (Degraded) | Δ Strata Sim |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Primary (Provenance-Matched)** | {comparison['explanation_stability']['primary_provenance_matched']['mean_cosine_similarity']:.4f} | {comparison['explanation_stability']['primary_provenance_matched']['strata_1_concordant_correct']:.4f} | {comparison['explanation_stability']['primary_provenance_matched']['strata_2_accuracy_degraded']:.4f} | {comparison['explanation_stability']['primary_provenance_matched']['delta_strata_stability']} |",
        f"| **Sensitivity (Unbalanced / Split-Confounded)** | {comparison['explanation_stability']['sensitivity_unbalanced']['mean_cosine_similarity']:.4f} | {comparison['explanation_stability']['sensitivity_unbalanced']['strata_1_concordant_correct']:.4f} | {comparison['explanation_stability']['sensitivity_unbalanced']['strata_2_accuracy_degraded']:.4f} | {comparison['explanation_stability']['sensitivity_unbalanced']['delta_strata_stability']} |",
        "",
        "---",
        "",
        "## 3. Robustness Verdict",
        "",
        f"- **Stability Preserved across Protocols:** {comparison['robustness_verdict']['stability_preserved_across_protocols']}",
        f"- **Decoupling Direction Consistent:** {comparison['robustness_verdict']['decoupling_direction_consistent']}",
        "",
    ]

    with open(output_path.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Saved sensitivity report to {output_path.with_suffix('.md')}")
    return output_path
