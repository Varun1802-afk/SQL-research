"""Data loading and deterministic sampling for Superviz26-SQL.

Provides schema validation, label mapping, deterministic sampling,
train/test leakage auditing, and cross-domain / within-domain dataset slicing.
"""

from src.data.loader import (
    CANONICAL_COLUMNS,
    REQUIRED_COLUMNS,
    DataLoader,
    DataSlice,
    LeakageReport,
    MatchReport,
    SQLRecord,
    check_train_test_leakage,
    load_csv_records,
    load_dataset_splits,
    sample_benign_train,
    sample_stratified_test,
    sample_stratified_train,
    validate_header,
    verify_test_sets_match,
    audit_source_domain_isolation,
    audit_within_domain_isolation,
    create_supervised_lodo_slices,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "DataLoader",
    "DataSlice",
    "LeakageReport",
    "MatchReport",
    "SQLRecord",
    "check_train_test_leakage",
    "load_csv_records",
    "load_dataset_splits",
    "sample_benign_train",
    "sample_stratified_test",
    "sample_stratified_train",
    "validate_header",
    "verify_test_sets_match",
    "audit_source_domain_isolation",
    "audit_within_domain_isolation",
    "create_supervised_lodo_slices",
]
