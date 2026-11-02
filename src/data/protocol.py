"""Experimental protocol definition and data partition coordinator.

Addresses Section 4: Split-provenance confounding elimination and evaluation isolation.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from src.data.loader import (
    DataSlice,
    SQLRecord,
    audit_source_domain_isolation,
    audit_within_domain_isolation,
    check_train_test_leakage,
    sample_stratified_test,
    sample_stratified_train,
)

logger = logging.getLogger(__name__)


def normalize_sql(q: str) -> str:
    """Normalize SQL string for strict deduplication."""
    s = q.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([;,()=])\s*", r"\1", s)
    return s.strip()


@dataclass
class ProtocolSliceBundle:
    """Encapsulates all datasets for an experimental run."""
    cross_train: DataSlice
    within_train: DataSlice
    repeat_cross_train: DataSlice  # identical training data to cross_train, for repeat-fit seed test
    eval_slice: DataSlice
    protocol_name: str
    metadata: dict[str, Any]

    def save_manifest(self, output_dir: Path) -> Path:
        """Export sample manifests and provenance metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / f"manifest_{self.protocol_name}.json"
        
        manifest_data = {
            "protocol_name": self.protocol_name,
            "metadata": self.metadata,
            "eval_summary": self.eval_slice.summary(),
            "cross_train_summary": self.cross_train.summary(),
            "within_train_summary": self.within_train.summary(),
            "repeat_cross_train_summary": self.repeat_cross_train.summary(),
            "eval_queries_count": len(self.eval_slice),
            "cross_train_queries_count": len(self.cross_train),
            "within_train_queries_count": len(self.within_train),
        }
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
            
        logger.info(f"Saved protocol manifest to {manifest_path}")
        return manifest_path


def create_provenance_matched_protocol(
    target_records: Sequence[SQLRecord],
    source_records: Sequence[SQLRecord],
    target_domain: str = "d",
    eval_size: int = 2000,
    eval_attack_ratio: float = 0.50,
    train_size: int = 4000,
    train_attack_ratio: float = 0.50,
    seed: int = 42,
) -> ProtocolSliceBundle:
    """Create primary provenance-matched protocol.

    Eliminates split-provenance confounding by drawing both benign and attack
    instances from the same original-partition generation stream (original_split == 'test').

    Protocol Steps:
    1. Pre-sampling isolation audits: Enforce zero target domain queries in source pool.
    2. Filter candidate pools to original_split == 'test' to eliminate partition confounding.
    3. Target Evaluation Reservation FIRST: Sample eval_size queries (e.g. 1000 benign, 1000 attack).
    4. Strict Evaluation Quarantine: Remove all evaluation queries (exact + normalized) from training pools.
    5. Within-Domain Training Sampling: Sample train_size queries (e.g. 2000 benign, 2000 attack) from quarantined target pool.
    6. Cross-Domain Training Sampling: Sample train_size queries from source domains A, B, C.
    7. Repeat-Fit Setup: repeat_cross_train shares identical records with cross_train.
    8. Disjointness & Leakage Assertions: Enforce 0 exact and normalized collisions between train and eval.
    """
    # 1. Pre-sampling isolation audits
    source_audit = audit_source_domain_isolation(source_records, target_domain=target_domain)
    if not source_audit["is_isolated"]:
        raise ValueError(
            f"Source records contain {source_audit['target_violations_count']} queries from target domain '{target_domain}'! Cross-domain evaluation breached."
        )

    within_audit = audit_within_domain_isolation(target_records, target_domain=target_domain)
    if not within_audit["is_pure"]:
        raise ValueError(
            f"Target records contain foreign domain queries: {within_audit['foreign_violations_count']} foreign records detected."
        )

    # 2. Filter to original_split == 'test' for provenance-matched protocol
    target_test_pool = [r for r in target_records if r.original_split == "test" or r.split == "test"]
    source_test_pool = [r for r in source_records if r.original_split == "test" or r.split == "test"]

    if not target_test_pool:
        target_test_pool = list(target_records)
    if not source_test_pool:
        source_test_pool = list(source_records)

    # 3. Target Evaluation Reservation FIRST
    eval_slice = sample_stratified_test(
        target_test_pool,
        n_total=eval_size,
        attack_ratio=eval_attack_ratio,
        seed=seed,
        slice_name=f"eval_provenance_matched_{target_domain}",
        require_is_test=False,
    )

    # 4. Strict Evaluation Quarantine
    eval_raw_set = {r.full_query for r in eval_slice.records}
    eval_norm_set = {normalize_sql(r.full_query) for r in eval_slice.records}

    target_train_candidates = [
        r for r in target_test_pool
        if r.full_query not in eval_raw_set and normalize_sql(r.full_query) not in eval_norm_set
    ]

    # 5. Within-Domain Training Sampling (M_within)
    within_train = sample_stratified_train(
        target_train_candidates,
        n_total=train_size,
        attack_ratio=train_attack_ratio,
        seed=seed,
        slice_name=f"train_within_provenance_matched_{target_domain}",
        require_is_train=False,
    )

    # 6. Cross-Domain Training Sampling (M_cross)
    # Ensure source records do not contain any evaluation queries
    source_train_candidates = [
        r for r in source_test_pool
        if r.full_query not in eval_raw_set and normalize_sql(r.full_query) not in eval_norm_set
    ]

    cross_train = sample_stratified_train(
        source_train_candidates,
        n_total=train_size,
        attack_ratio=train_attack_ratio,
        seed=seed,
        slice_name="train_cross_provenance_matched",
        require_is_train=False,
    )

    # 7. Repeat-Fit model setup: identical training instances to cross_train
    repeat_cross_train = cross_train

    # 8. Disjointness, Provenance & Leakage Assertions
    leakage_cross = check_train_test_leakage(cross_train.records, eval_slice.records)
    leakage_within = check_train_test_leakage(within_train.records, eval_slice.records)

    if leakage_cross.has_leakage:
        raise ValueError(f"Cross-domain leakage detected: {leakage_cross.query_overlap_count} exact, {leakage_cross.normalized_query_overlap_count} normalized.")
    if leakage_within.has_leakage:
        raise ValueError(f"Within-domain leakage detected: {leakage_within.query_overlap_count} exact, {leakage_within.normalized_query_overlap_count} normalized.")

    # Strict template-disjointness assertion for cross-domain transfer (B -> D)
    if leakage_cross.template_overlap_count > 0:
        raise ValueError(f"Cross-domain template overlap detected: {leakage_cross.template_overlap_count} overlapping templates between Domain B and Domain D!")

    # Strict label balance and integrity assertions
    for slc, expected_len, expected_ratio, name in [
        (eval_slice, eval_size, eval_attack_ratio, "eval_slice"),
        (cross_train, train_size, train_attack_ratio, "cross_train"),
        (within_train, train_size, train_attack_ratio, "within_train"),
    ]:
        if len(slc) != expected_len:
            raise ValueError(f"Slice {name} length mismatch: expected {expected_len}, got {len(slc)}")
        expected_attacks = int(round(expected_len * expected_ratio))
        expected_benign = expected_len - expected_attacks
        if slc.attack_count != expected_attacks or slc.benign_count != expected_benign:
            raise ValueError(f"Slice {name} class imbalance: expected {expected_benign} benign / {expected_attacks} attacks, got {slc.benign_count} benign / {slc.attack_count} attacks")
        if not all(r.label in (0, 1) for r in slc.records):
            raise ValueError(f"Slice {name} contains invalid labels outside {{0, 1}}!")
        if not all(r.original_split for r in slc.records):
            raise ValueError(f"Slice {name} contains records missing original_split provenance metadata!")

    metadata = {
        "protocol": "provenance_matched",
        "rationale": "Draws benign and attack instances from comparable test-partition pools to eliminate split-provenance confounding.",
        "target_domain": target_domain,
        "seed": seed,
        "eval_size": eval_size,
        "eval_attack_ratio": eval_attack_ratio,
        "train_size": train_size,
        "train_attack_ratio": train_attack_ratio,
        "target_pool_available": len(target_test_pool),
        "source_pool_available": len(source_test_pool),
        "leakage_audit_cross": leakage_cross.summary(),
        "leakage_audit_within": leakage_within.summary(),
    }

    return ProtocolSliceBundle(
        cross_train=cross_train,
        within_train=within_train,
        repeat_cross_train=repeat_cross_train,
        eval_slice=eval_slice,
        protocol_name="provenance_matched",
        metadata=metadata,
    )


def create_sensitivity_unbalanced_protocol(
    target_records: Sequence[SQLRecord],
    source_records: Sequence[SQLRecord],
    target_domain: str = "d",
    eval_size: int = 2000,
    eval_attack_ratio: float = 0.10,
    train_size: int = 4000,
    train_attack_ratio: float = 0.10,
    seed: int = 42,
) -> ProtocolSliceBundle:
    """Create sensitivity protocol using unbalanced (10% attack) split-confounded pools.

    Draws benign from official train partition and attack from official test partition.
    Used for Section 11 Sensitivity Analysis to evaluate how findings change when
    evaluating under the benchmark's original unbalanced class ratio.
    """
    # Isolation checks
    source_audit = audit_source_domain_isolation(source_records, target_domain=target_domain)
    if not source_audit["is_isolated"]:
        raise ValueError("Source isolation violation in sensitivity protocol.")

    # 1. Target Evaluation Reservation FIRST
    eval_slice = sample_stratified_test(
        target_records,
        n_total=eval_size,
        attack_ratio=eval_attack_ratio,
        seed=seed,
        slice_name=f"eval_sensitivity_unbalanced_{target_domain}",
        require_is_test=False,
    )

    # 2. Evaluation Quarantine
    eval_raw_set = {r.full_query for r in eval_slice.records}
    eval_norm_set = {normalize_sql(r.full_query) for r in eval_slice.records}

    target_train_candidates = [
        r for r in target_records
        if r.full_query not in eval_raw_set and normalize_sql(r.full_query) not in eval_norm_set
    ]

    # 3. Within-Domain Training Sampling
    within_train = sample_stratified_train(
        target_train_candidates,
        n_total=train_size,
        attack_ratio=train_attack_ratio,
        seed=seed,
        slice_name=f"train_within_sensitivity_unbalanced_{target_domain}",
        require_is_train=False,
    )

    # 4. Cross-Domain Training Sampling
    source_train_candidates = [
        r for r in source_records
        if r.full_query not in eval_raw_set and normalize_sql(r.full_query) not in eval_norm_set
    ]
    cross_train = sample_stratified_train(
        source_train_candidates,
        n_total=train_size,
        attack_ratio=train_attack_ratio,
        seed=seed,
        slice_name="train_cross_sensitivity_unbalanced",
        require_is_train=False,
    )

    repeat_cross_train = cross_train

    leakage_cross = check_train_test_leakage(cross_train.records, eval_slice.records)
    leakage_within = check_train_test_leakage(within_train.records, eval_slice.records)

    metadata = {
        "protocol": "sensitivity_unbalanced",
        "rationale": "Sensitivity baseline testing 10% attack prevalence and mixed original splits.",
        "target_domain": target_domain,
        "seed": seed,
        "eval_size": eval_size,
        "eval_attack_ratio": eval_attack_ratio,
        "train_size": train_size,
        "train_attack_ratio": train_attack_ratio,
        "leakage_audit_cross": leakage_cross.summary(),
        "leakage_audit_within": leakage_within.summary(),
    }

    return ProtocolSliceBundle(
        cross_train=cross_train,
        within_train=within_train,
        repeat_cross_train=repeat_cross_train,
        eval_slice=eval_slice,
        protocol_name="sensitivity_unbalanced",
        metadata=metadata,
    )
