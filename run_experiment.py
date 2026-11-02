"""Main reproducible experiment orchestrator for SQL injection detection & explainability.

Executes the full experimental workflow:
1. Data loading and validation.
2. Protocol creation (provenance-matched primary + sensitivity unbalanced).
3. Model training (Logistic Regression, Random Forest, XGBoost).
4. Detection performance evaluation on quarantined evaluation set.
5. SHAP attribution generation.
6. LIME repeated attribution generation (10 repetitions per query).
7. Explanation stability and decoupling analysis.
8. Protocol sensitivity comparison.
9. Figure generation and results serialization.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import numpy as np

from src.data.loader import DataLoader, load_csv_records
from src.data.protocol import (
    create_provenance_matched_protocol,
    create_sensitivity_unbalanced_protocol,
    ProtocolSliceBundle,
)
from src.models.train import (
    SQLDetectionPipeline,
    extract_features_matrix,
    train_model_pipeline,
)
from src.models.evaluate import evaluate_detection_pipeline, save_metrics_report
from src.explainers.shap_explainer import (
    generate_shap_attributions,
    save_shap_attributions_to_csv,
)
from src.explainers.lime_explainer import (
    generate_lime_attributions_with_repetitions,
    save_lime_attributions_to_csv,
)
from src.analysis.stability import (
    analyze_cross_vs_within_stability,
    analyze_lime_repeat_stability,
    analyze_repeat_fit_stability,
)
from src.analysis.sensitivity import (
    run_protocol_sensitivity_comparison,
    save_sensitivity_report,
)
from src.visualization.plots import (
    plot_detection_metrics_comparison,
    plot_decoupling_strata,
    plot_top_feature_importances,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_experiment(
    data_dir: Path = Path("data"),
    results_dir: Path = Path("results"),
    models_to_run: list[str] = None,
    eval_size: int = 2000,
    train_size: int = 4000,
    shap_sample_size: int = 200,
    lime_sample_size: int = 50,
    lime_repetitions: int = 10,
    master_seed: int = 42,
) -> dict[str, Any]:
    """Execute end-to-end reproducible experiment."""
    models_to_run = models_to_run or ["logistic_regression", "random_forest", "xgboost"]
    logger.info("=" * 70)
    logger.info("STARTING SQL INJECTION EXPLANATION STABILITY EXPERIMENT")
    logger.info(f"Target models: {models_to_run}")
    logger.info(f"Evaluation size: {eval_size}, Training size per model: {train_size}")
    logger.info(f"SHAP subset: {shap_sample_size}, LIME subset: {lime_sample_size} (10 repeats)")
    logger.info(f"Master random seed: {master_seed}")
    logger.info("=" * 70)

    # 1. Locate dataset files
    target_path = data_dir / "d-d.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"Target dataset file {target_path} not found. Run src/data/download.py first.")

    # Locate source domain files: prioritize b-b.csv (Sakila) or a-a.csv or abc-d.csv
    source_path = None
    for candidate in ["b-b.csv", "a-a.csv", "abc-d.csv"]:
        cand_p = data_dir / candidate
        if cand_p.exists() and cand_p.stat().st_size > 100_000_000:
            source_path = cand_p
            break

    if source_path is None:
        raise FileNotFoundError(f"No source dataset file found in {data_dir}. Need b-b.csv, a-a.csv, or abc-d.csv.")

    logger.info(f"Loading target domain data from {target_path}...")
    target_records = load_csv_records(target_path, domain="d", max_rows=150_000)

    source_domain_id = "b" if "b-b" in source_path.name else ("a" if "a-a" in source_path.name else "abc")
    logger.info(f"Loading source domain data from {source_path} (domain={source_domain_id})...")
    source_records = load_csv_records(source_path, domain=source_domain_id, max_rows=150_000)

    # 2. Build Primary Protocol (Provenance-Matched)
    logger.info("\n--- STEP 2: CREATING PROVENANCE-MATCHED PROTOCOL SLICES ---")
    protocol_bundle = create_provenance_matched_protocol(
        target_records=target_records,
        source_records=source_records,
        target_domain="d",
        eval_size=eval_size,
        eval_attack_ratio=0.50,
        train_size=train_size,
        train_attack_ratio=0.50,
        seed=master_seed,
    )
    manifest_path = protocol_bundle.save_manifest(results_dir / "audit")

    # 3. Extract evaluation features once (shared across all models)
    eval_slice = protocol_bundle.eval_slice
    logger.info(f"\nExtracting features for {len(eval_slice)} evaluation queries...")
    X_eval = extract_features_matrix(eval_slice.queries)
    y_eval = np.array(eval_slice.labels, dtype=np.int64)

    # Bounded subset for SHAP explanations (stratified 50/50)
    rng_shap = np.random.RandomState(master_seed)
    benign_eval_indices = np.where(y_eval == 0)[0]
    attack_eval_indices = np.where(y_eval == 1)[0]
    n_half = shap_sample_size // 2
    selected_shap_indices = np.sort(np.concatenate([
        rng_shap.choice(benign_eval_indices, min(n_half, len(benign_eval_indices)), replace=False),
        rng_shap.choice(attack_eval_indices, min(n_half, len(attack_eval_indices)), replace=False),
    ]))
    X_shap_eval = X_eval[selected_shap_indices]
    queries_shap = [eval_slice.queries[i] for i in selected_shap_indices]
    labels_shap = [eval_slice.labels[i] for i in selected_shap_indices]

    # Subset for LIME (smaller subset due to 10 repetitions per query)
    selected_lime_indices = selected_shap_indices[:lime_sample_size]
    X_lime_eval = X_eval[selected_lime_indices]
    queries_lime = [eval_slice.queries[i] for i in selected_lime_indices]
    labels_lime = [eval_slice.labels[i] for i in selected_lime_indices]

    experiment_results: dict[str, Any] = {
        "metadata": {
            "master_seed": master_seed,
            "eval_size": len(eval_slice),
            "train_size": train_size,
            "shap_subset_size": len(selected_shap_indices),
            "lime_subset_size": len(selected_lime_indices),
            "target_domain": "d",
            "source_domain": source_domain_id,
            "source_file": str(source_path),
            "target_file": str(target_path),
        },
        "models": {},
    }

    all_detection_metrics = []

    # 4. Train and Evaluate each model family
    for family in models_to_run:
        logger.info(f"\n" + "=" * 50)
        logger.info(f"PROCESSING MODEL FAMILY: {family.upper()}")
        logger.info("=" * 50)

        # A. Train M_cross (Cross-Domain)
        pipe_cross = train_model_pipeline(
            train_slice=protocol_bundle.cross_train,
            model_family=family,
            model_id=f"{family}_cross",
            seed=master_seed,
            training_domain=source_domain_id,
        )
        pipe_cross.save(results_dir / "models" / f"{family}_cross.joblib")

        # B. Train M_within (Within-Domain Reference)
        pipe_within = train_model_pipeline(
            train_slice=protocol_bundle.within_train,
            model_family=family,
            model_id=f"{family}_within",
            seed=master_seed,
            training_domain="d",
        )
        pipe_within.save(results_dir / "models" / f"{family}_within.joblib")

        # C. Train M_cross_repeat (Repeat-Fit Reference with different seed)
        pipe_cross_repeat = train_model_pipeline(
            train_slice=protocol_bundle.repeat_cross_train,
            model_family=family,
            model_id=f"{family}_cross_repeat",
            seed=master_seed + 81,  # different estimator seed
            training_domain=source_domain_id,
        )
        pipe_cross_repeat.save(results_dir / "models" / f"{family}_cross_repeat.joblib")

        # Evaluate detection performance
        metrics_cross = evaluate_detection_pipeline(pipe_cross, eval_slice, X_eval=X_eval)
        metrics_within = evaluate_detection_pipeline(pipe_within, eval_slice, X_eval=X_eval)
        metrics_repeat = evaluate_detection_pipeline(pipe_cross_repeat, eval_slice, X_eval=X_eval)

        all_detection_metrics.extend([metrics_cross, metrics_within, metrics_repeat])

        # Plot detection comparison
        plot_detection_metrics_comparison(
            metrics_cross=metrics_cross,
            metrics_within=metrics_within,
            model_family=family,
            output_path=results_dir / "figures" / f"detection_metrics_{family}.png",
        )

        # 5. Generate SHAP attributions
        logger.info(f"\n--- GENERATING SHAP ATTRIBUTIONS ({family}) ---")
        X_bg = extract_features_matrix(protocol_bundle.cross_train.queries[:500])

        shap_cross = generate_shap_attributions(pipe_cross, X_shap_eval, X_background=X_bg)
        shap_within = generate_shap_attributions(pipe_within, X_shap_eval, X_background=X_bg)
        shap_repeat = generate_shap_attributions(pipe_cross_repeat, X_shap_eval, X_background=X_bg)

        preds_cross = pipe_cross.predict(X_shap_eval)
        preds_within = pipe_within.predict(X_shap_eval)

        save_shap_attributions_to_csv(
            shap_matrix=shap_cross,
            queries=queries_shap,
            labels=labels_shap,
            preds=preds_cross,
            output_path=results_dir / "explanations" / f"shap_attributions_{family}_cross.csv",
        )
        save_shap_attributions_to_csv(
            shap_matrix=shap_within,
            queries=queries_shap,
            labels=labels_shap,
            preds=preds_within,
            output_path=results_dir / "explanations" / f"shap_attributions_{family}_within.csv",
        )

        # 6. Generate LIME explanations (10 repetitions)
        logger.info(f"\n--- GENERATING LIME REPEATED EXPLANATIONS ({family}) ---")
        X_train_unscaled = extract_features_matrix(protocol_bundle.cross_train.queries[:1000])
        lime_cross = generate_lime_attributions_with_repetitions(
            pipeline=pipe_cross,
            X_train=X_train_unscaled,
            X_explain=X_lime_eval,
            n_repetitions=lime_repetitions,
            base_seed=master_seed,
        )
        preds_lime = pipe_cross.predict(X_lime_eval)
        save_lime_attributions_to_csv(
            lime_tensor=lime_cross,
            queries=queries_lime,
            labels=labels_lime,
            preds=preds_lime,
            output_path=results_dir / "explanations" / f"lime_attributions_{family}_cross.csv",
            base_seed=master_seed,
        )

        # 7. Analyze stability
        logger.info(f"\n--- ANALYZING EXPLANATION STABILITY ({family}) ---")
        techniques_shap = [eval_slice.records[i].attack_technique for i in selected_shap_indices]

        # A. Cross-Domain vs. Within-Domain Divergence & Decoupling
        stability_analysis = analyze_cross_vs_within_stability(
            attr_cross=shap_cross,
            attr_within=shap_within,
            y_true=labels_shap,
            y_pred_cross=preds_cross,
            y_pred_within=preds_within,
            techniques=techniques_shap,
        )

        # B. Repeat-Fit Model Stability (Estimator Stochasticity)
        repeat_fit_stability = analyze_repeat_fit_stability(shap_cross, shap_repeat)

        # C. LIME Perturbation Stability (10 repeated runs)
        lime_repeat_stability = analyze_lime_repeat_stability(lime_cross)

        # Plots
        plot_decoupling_strata(
            decoupling_data=stability_analysis["decoupling_analysis"],
            model_family=family,
            output_path=results_dir / "figures" / f"decoupling_strata_{family}.png",
        )
        plot_top_feature_importances(
            attr_cross=shap_cross,
            attr_within=shap_within,
            model_family=family,
            output_path=results_dir / "figures" / f"top_features_{family}.png",
        )

        experiment_results["models"][family] = {
            "detection": {
                "cross_domain": metrics_cross,
                "within_domain": metrics_within,
                "cross_repeat": metrics_repeat,
                "delta_f1": round(metrics_cross["f1_score"] - metrics_within["f1_score"], 4),
                "delta_accuracy": round(metrics_cross["accuracy"] - metrics_within["accuracy"], 4),
            },
            "stability": {
                "cross_vs_within": stability_analysis,
                "repeat_fit": repeat_fit_stability,
                "lime_repeat": lime_repeat_stability,
            },
        }

    # Save all detection metrics to JSON
    save_metrics_report(all_detection_metrics, results_dir / "metrics" / "detection_metrics.json")

    # 8. Protocol Sensitivity Analysis (Section 11)
    logger.info("\n--- STEP 8: RUNNING SENSITIVITY ANALYSIS (UNBALANCED / SPLIT-CONFOUNDED PROTOCOL) ---")
    sensitivity_bundle = create_sensitivity_unbalanced_protocol(
        target_records=target_records,
        source_records=source_records,
        target_domain="d",
        eval_size=eval_size,
        eval_attack_ratio=0.10,
        train_size=train_size,
        train_attack_ratio=0.10,
        seed=master_seed,
    )
    sensitivity_manifest = sensitivity_bundle.save_manifest(results_dir / "audit")

    # Run sensitivity comparison for primary model (e.g. xgboost or random_forest)
    sens_family = "xgboost" if "xgboost" in models_to_run else models_to_run[0]
    pipe_sens_cross = train_model_pipeline(
        sensitivity_bundle.cross_train, sens_family, f"{sens_family}_sens_cross", seed=master_seed
    )
    pipe_sens_within = train_model_pipeline(
        sensitivity_bundle.within_train, sens_family, f"{sens_family}_sens_within", seed=master_seed
    )

    X_sens_eval = extract_features_matrix(sensitivity_bundle.eval_slice.queries)
    sens_met_cross = evaluate_detection_pipeline(pipe_sens_cross, sensitivity_bundle.eval_slice, X_eval=X_sens_eval)
    sens_met_within = evaluate_detection_pipeline(pipe_sens_within, sensitivity_bundle.eval_slice, X_eval=X_sens_eval)

    # SHAP for sensitivity
    shap_sens_cross = generate_shap_attributions(pipe_sens_cross, X_sens_eval[:shap_sample_size])
    shap_sens_within = generate_shap_attributions(pipe_sens_within, X_sens_eval[:shap_sample_size])

    sens_stability = analyze_cross_vs_within_stability(
        attr_cross=shap_sens_cross,
        attr_within=shap_sens_within,
        y_true=sensitivity_bundle.eval_slice.labels[:shap_sample_size],
        y_pred_cross=pipe_sens_cross.predict(X_sens_eval[:shap_sample_size]),
        y_pred_within=pipe_sens_within.predict(X_sens_eval[:shap_sample_size]),
    )

    sens_comparison = run_protocol_sensitivity_comparison(
        primary_metrics=experiment_results["models"][sens_family],
        sensitivity_metrics={"detection": {"cross_domain": sens_met_cross, "within_domain": sens_met_within}},
        primary_stability=experiment_results["models"][sens_family]["stability"],
        sensitivity_stability={"cross_vs_within": sens_stability},
        model_family=sens_family,
    )
    save_sensitivity_report(sens_comparison, results_dir / "reports" / "sensitivity_analysis_report")
    experiment_results["sensitivity_analysis"] = sens_comparison

    # Save complete experiment results JSON
    results_json_path = results_dir / "reports" / "full_experiment_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    logger.info(f"\nEXPERIMENT COMPLETED SUCCESSFULLY! Full results saved to {results_json_path}")
    return experiment_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SQLi explanation stability experiment.")
    parser.add_argument("--eval-size", type=int, default=2000, help="Evaluation sample size")
    parser.add_argument("--train-size", type=int, default=4000, help="Training sample size per model")
    parser.add_argument("--shap-size", type=int, default=200, help="SHAP evaluation subset size")
    parser.add_argument("--lime-size", type=int, default=50, help="LIME evaluation subset size")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--results-dir", type=str, default="results", help="Results output directory")
    args = parser.parse_args()

    run_experiment(
        results_dir=Path(args.results_dir),
        eval_size=args.eval_size,
        train_size=args.train_size,
        shap_sample_size=args.shap_size,
        lime_sample_size=args.lime_size,
        master_seed=args.seed,
    )
