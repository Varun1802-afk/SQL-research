"""Data loader and deterministic sampling module for Superviz26-SQL.

Provides schema validation, strict label encoding verification, deterministic
subsampling with fixed random seeds, train/test leakage detection, and
cross-domain / within-domain dataset slice generation.

Implemented strictly using the Python Standard Library (no third-party ML dependencies).
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import random
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

# Canonical column definitions established from official Superviz26-SQL specifications
# (see experiments/load_concept_drift.py and experiments/load_fews_shots.py in gquetel/adapt-sql).
REQUIRED_COLUMNS: list[str] = [
    "full_query",
    "label",
    "split",
]

CANONICAL_COLUMNS: list[str] = [
    "full_query",
    "label",
    "user_inputs",
    "attack_stage",
    "tamper_method",
    "attack_status",
    "statement_type",
    "query_template_id",
    "attack_id",
    "attack_technique",
    "split",
]

# Valid label encodings
LABEL_BENIGN: int = 0
LABEL_ATTACK: int = 1
VALID_LABELS: set[int] = {LABEL_BENIGN, LABEL_ATTACK}
VALID_SPLITS: set[str] = {"train", "test"}


@dataclass(frozen=True)
class SQLRecord:
    """Immutable representation of a single SQL query row from Superviz26-SQL."""

    full_query: str
    label: int
    split: str
    domain: str = ""
    source_file: str = ""
    original_split: str = ""
    user_inputs: str = ""
    attack_stage: str = ""
    tamper_method: str = ""
    attack_status: str = ""
    statement_type: str = ""
    query_template_id: str = ""
    attack_id: str = ""
    attack_technique: str = ""
    extra_fields: dict[str, str] = field(default_factory=dict)

    @property
    def is_benign(self) -> bool:
        """Return True if record is labeled as benign (label 0)."""
        return self.label == LABEL_BENIGN

    @property
    def is_attack(self) -> bool:
        """Return True if record is labeled as SQL injection (label 1)."""
        return self.label == LABEL_ATTACK

    @property
    def is_train(self) -> bool:
        """Return True if record belongs to the train split."""
        return self.split == "train"

    @property
    def is_test(self) -> bool:
        """Return True if record belongs to the test split."""
        return self.split == "test"

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a flat dictionary."""
        d: dict[str, Any] = {
            "full_query": self.full_query,
            "label": self.label,
            "split": self.split,
            "domain": self.domain,
            "source_file": self.source_file,
            "original_split": self.original_split,
            "user_inputs": self.user_inputs,
            "attack_stage": self.attack_stage,
            "tamper_method": self.tamper_method,
            "attack_status": self.attack_status,
            "statement_type": self.statement_type,
            "query_template_id": self.query_template_id,
            "attack_id": self.attack_id,
            "attack_technique": self.attack_technique,
        }
        d.update(self.extra_fields)
        return d


@dataclass
class DataSlice:
    """A collection of SQLRecords with descriptive statistics and slice metadata."""

    name: str
    records: list[SQLRecord] = field(default_factory=list)
    source_file: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[SQLRecord]:
        return iter(self.records)

    def __getitem__(self, index: int) -> SQLRecord:
        return self.records[index]

    @property
    def queries(self) -> list[str]:
        """List of raw SQL queries in order."""
        return [r.full_query for r in self.records]

    @property
    def labels(self) -> list[int]:
        """List of integer labels in order."""
        return [r.label for r in self.records]

    @property
    def benign_count(self) -> int:
        """Number of benign queries (label 0)."""
        return sum(1 for r in self.records if r.is_benign)

    @property
    def attack_count(self) -> int:
        """Number of SQL injection queries (label 1)."""
        return sum(1 for r in self.records if r.is_attack)

    @property
    def attack_ratio(self) -> float:
        """Proportion of SQL injection queries."""
        if not self.records:
            return 0.0
        return self.attack_count / len(self.records)

    @property
    def technique_distribution(self) -> dict[str, int]:
        """Distribution of attack techniques among malicious queries."""
        return dict(
            Counter(
                r.attack_technique for r in self.records if r.is_attack and r.attack_technique
            )
        )

    @property
    def statement_type_distribution(self) -> dict[str, int]:
        """Distribution of SQL statement types."""
        return dict(Counter(r.statement_type for r in self.records if r.statement_type))

    @property
    def template_ids(self) -> set[str]:
        """Set of unique query template IDs present in slice."""
        return {r.query_template_id for r in self.records if r.query_template_id}

    @property
    def domain_distribution(self) -> dict[str, int]:
        """Distribution of domain origins across records."""
        return dict(Counter(r.domain for r in self.records if r.domain))

    @property
    def original_split_distribution(self) -> dict[str, int]:
        """Distribution of original split origins across records."""
        return dict(Counter(r.original_split for r in self.records if r.original_split))

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary of slice statistics."""
        return {
            "name": self.name,
            "total_records": len(self.records),
            "benign_count": self.benign_count,
            "attack_count": self.attack_count,
            "attack_ratio": round(self.attack_ratio, 4),
            "unique_templates": len(self.template_ids),
            "technique_distribution": self.technique_distribution,
            "statement_type_distribution": self.statement_type_distribution,
            "domain_distribution": self.domain_distribution,
            "original_split_distribution": self.original_split_distribution,
            "source_file": str(self.source_file) if self.source_file else None,
            "metadata": self.metadata,
        }


@dataclass
class LeakageReport:
    """Audit report for overlap / leakage between two data slices."""

    train_size: int
    test_size: int
    query_overlap_count: int
    overlapping_queries_sample: list[str]
    normalized_query_overlap_count: int
    template_overlap_count: int
    overlapping_templates_sample: list[str]
    has_leakage: bool

    def summary(self) -> dict[str, Any]:
        return {
            "train_size": self.train_size,
            "test_size": self.test_size,
            "query_overlap_count": self.query_overlap_count,
            "has_leakage": self.has_leakage,
            "normalized_query_overlap_count": self.normalized_query_overlap_count,
            "template_overlap_count": self.template_overlap_count,
            "overlapping_queries_sample": self.overlapping_queries_sample[:5],
            "overlapping_templates_sample": self.overlapping_templates_sample[:5],
        }


@dataclass
class MatchReport:
    """Audit report comparing two test sets for identity."""

    size_a: int
    size_b: int
    identical_queries: bool
    identical_labels: bool
    identical_order: bool
    mismatch_count: int

    @property
    def is_exact_match(self) -> bool:
        return (
            self.size_a == self.size_b
            and self.identical_queries
            and self.identical_labels
            and self.identical_order
            and self.mismatch_count == 0
        )


def validate_header(header: Sequence[str]) -> list[str]:
    """Validate that the CSV header contains all required columns.

    Raises:
        ValueError: If any required column is missing.
    """
    if not header:
        raise ValueError("CSV header is empty or missing.")

    clean_header = [c.strip() for c in header]
    missing = [col for col in REQUIRED_COLUMNS if col not in clean_header]
    if missing:
        raise ValueError(
            f"CSV header missing required column(s): {missing}. Found columns: {clean_header}"
        )
    return clean_header


def parse_label(raw_label: Any, row_idx: int) -> int:
    """Parse and validate integer label encoding (0 = benign, 1 = attack).

    Raises:
        ValueError: If label is invalid or not in {0, 1}.
    """
    if raw_label is None:
        raise ValueError(f"Row {row_idx}: Label is None. Expected 0 (benign) or 1 (attack).")

    raw_str = str(raw_label).strip()
    if raw_str in ("0", "0.0", "benign", "normal"):
        return LABEL_BENIGN
    elif raw_str in ("1", "1.0", "attack", "sqli", "malicious"):
        # For Superviz26-SQL, the authoritative format is integer 0 or 1.
        if raw_str in ("0", "1"):
            return int(raw_str)
        # Accept text representations with strict mapping if needed, but primary is integer string
        return LABEL_ATTACK
    else:
        try:
            val = int(raw_str)
            if val in VALID_LABELS:
                return val
        except ValueError:
            pass
        raise ValueError(
            f"Row {row_idx}: Invalid label value '{raw_label}'. Expected 0 (benign) or 1 (attack)."
        )


def parse_split(raw_split: Any, row_idx: int) -> str:
    """Parse and validate split value ('train' or 'test').

    Raises:
        ValueError: If split is invalid.
    """
    if raw_split is None:
        raise ValueError(f"Row {row_idx}: Split is None. Expected 'train' or 'test'.")

    s = str(raw_split).strip().lower()
    if s not in VALID_SPLITS:
        raise ValueError(
            f"Row {row_idx}: Invalid split value '{raw_split}'. Expected 'train' or 'test'."
        )
    return s


def load_csv_records(
    file_path: Union[str, Path],
    domain: Optional[str] = None,
    max_rows: Optional[int] = None,
    encoding: str = "utf-8",
) -> list[SQLRecord]:
    """Load SQL records from a CSV file with full schema and label validation.

    Args:
        file_path: Path to the CSV file.
        domain: Optional domain identifier (defaults to file stem).
        max_rows: Optional maximum number of rows to read.
        encoding: Text encoding (default utf-8).

    Returns:
        List of validated SQLRecord instances.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If schema, label, or split validation fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            "Verify local path or obtain dataset from Zenodo (DOI: 10.5281/zenodo.19627322)."
        )

    records: list[SQLRecord] = []
    canonical_set = set(CANONICAL_COLUMNS)
    assigned_domain = domain if domain is not None else path.stem

    with open(path, mode="r", encoding=encoding, errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"Dataset file is completely empty: {path}")

        header = validate_header(raw_header)
        col_to_idx = {col: idx for idx, col in enumerate(header)}

        for row_idx, row in enumerate(reader, start=2):
            if not row or all(c == "" for c in row):
                continue  # skip empty lines

            # Pad row if fewer columns than header (e.g. trailing delimiters omitted)
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))

            full_query = row[col_to_idx["full_query"]]
            label = parse_label(row[col_to_idx["label"]], row_idx)
            split = parse_split(row[col_to_idx["split"]], row_idx)

            user_inputs = (
                row[col_to_idx["user_inputs"]] if "user_inputs" in col_to_idx else ""
            )
            attack_stage = (
                row[col_to_idx["attack_stage"]] if "attack_stage" in col_to_idx else ""
            )
            tamper_method = (
                row[col_to_idx["tamper_method"]] if "tamper_method" in col_to_idx else ""
            )
            attack_status = (
                row[col_to_idx["attack_status"]] if "attack_status" in col_to_idx else ""
            )
            statement_type = (
                row[col_to_idx["statement_type"]] if "statement_type" in col_to_idx else ""
            )
            query_template_id = (
                row[col_to_idx["query_template_id"]]
                if "query_template_id" in col_to_idx
                else ""
            )
            attack_id = (
                row[col_to_idx["attack_id"]] if "attack_id" in col_to_idx else ""
            )
            attack_technique = (
                row[col_to_idx["attack_technique"]]
                if "attack_technique" in col_to_idx
                else ""
            )

            extra: dict[str, str] = {}
            for col, idx in col_to_idx.items():
                if col not in canonical_set:
                    extra[col] = row[idx] if idx < len(row) else ""

            records.append(
                SQLRecord(
                    full_query=full_query,
                    label=label,
                    split=split,
                    domain=assigned_domain,
                    source_file=str(path),
                    original_split=split,
                    user_inputs=user_inputs,
                    attack_stage=attack_stage,
                    tamper_method=tamper_method,
                    attack_status=attack_status,
                    statement_type=statement_type,
                    query_template_id=query_template_id,
                    attack_id=attack_id,
                    attack_technique=attack_technique,
                    extra_fields=extra,
                )
            )

            if max_rows is not None and len(records) >= max_rows:
                break

    return records


def load_dataset_splits(
    file_path: Union[str, Path],
    domain: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> tuple[DataSlice, DataSlice]:
    """Load a Superviz26-SQL file and partition it into train and test DataSlices.

    Args:
        file_path: Path to the dataset CSV file.
        domain: Optional domain identifier.
        max_rows: Optional maximum rows to load.

    Returns:
        Tuple of (train_slice, test_slice).
    """
    path = Path(file_path)
    records = load_csv_records(path, domain=domain, max_rows=max_rows)

    train_records = [r for r in records if r.is_train]
    test_records = [r for r in records if r.is_test]

    train_slice = DataSlice(
        name=f"{path.stem}_train",
        records=train_records,
        source_file=path,
        metadata={"split": "train", "source_file": str(path), "domain": domain or path.stem},
    )
    test_slice = DataSlice(
        name=f"{path.stem}_test",
        records=test_records,
        source_file=path,
        metadata={"split": "test", "source_file": str(path), "domain": domain or path.stem},
    )

    return train_slice, test_slice


def sample_benign_train(
    records: Sequence[SQLRecord],
    n: int,
    seed: int = 42,
    slice_name: str = "benign_train_slice",
) -> DataSlice:
    """Deterministically sample n benign training records using an explicit random seed.

    Args:
        records: Source sequence of SQLRecord instances.
        n: Number of benign training queries to sample.
        seed: Random seed for deterministic subsampling.
        slice_name: Name for the resulting DataSlice.

    Returns:
        DataSlice containing exactly n benign training records.

    Raises:
        ValueError: If n <= 0 or if fewer than n benign train records are available.
    """
    if n <= 0:
        raise ValueError(f"Requested sample size n must be positive, got {n}.")

    benign_train_pool = [r for r in records if r.is_train and r.is_benign]
    available = len(benign_train_pool)

    if available < n:
        raise ValueError(
            f"Cannot sample {n} benign train records: only {available} available in pool."
        )

    # Deterministic sampling: index selection with fixed seed
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(available), n))
    sampled_records = [benign_train_pool[i] for i in selected_indices]

    return DataSlice(
        name=slice_name,
        records=sampled_records,
        metadata={
            "sample_type": "benign_train",
            "requested_size": n,
            "pool_available": available,
            "seed": seed,
        },
    )


def sample_stratified_train(
    records: Sequence[SQLRecord],
    n_total: int,
    attack_ratio: float = 0.10,
    seed: int = 42,
    slice_name: str = "stratified_train_slice",
    require_is_train: bool = True,
) -> DataSlice:
    """Deterministically sample n_total training queries with a fixed attack ratio.

    Crucial for supervised binary classification models (Logistic Regression,
    Random Forest, XGBoost), which require both positive (attack=1) and negative
    (benign=0) training instances to compute loss gradients and decision boundaries.

    Args:
        records: Source sequence of SQLRecord instances.
        n_total: Total number of training queries to sample.
        attack_ratio: Desired proportion of malicious queries (e.g. 0.10).
        seed: Random seed for deterministic subsampling.
        slice_name: Name for the resulting DataSlice.
        require_is_train: If True, only select from records where r.is_train is True.
            If False, candidate pool includes all records regardless of split flag.

    Returns:
        DataSlice containing exactly n_total training records with verified class distribution.

    Raises:
        ValueError: If n_total <= 0, attack_ratio invalid, or pools lack sufficient rows.
    """
    if n_total <= 0:
        raise ValueError(f"Requested total train sample size n_total must be positive, got {n_total}.")
    if not (0.0 < attack_ratio < 1.0):
        raise ValueError(f"Supervised attack ratio must be strictly between 0.0 and 1.0, got {attack_ratio}.")

    train_records = [r for r in records if r.is_train] if require_is_train else list(records)
    benign_pool = [r for r in train_records if r.is_benign]
    attack_pool = [r for r in train_records if r.is_attack]

    n_attack = round(n_total * attack_ratio)
    n_benign = n_total - n_attack

    # Ensure at least 1 sample per class
    if n_attack == 0:
        n_attack = 1
        n_benign = n_total - 1
    elif n_benign == 0:
        n_benign = 1
        n_attack = n_total - 1

    if len(benign_pool) < n_benign:
        raise ValueError(
            f"Cannot sample {n_benign} benign train records: only {len(benign_pool)} available in train pool."
        )
    if len(attack_pool) < n_attack:
        raise ValueError(
            f"Cannot sample {n_attack} attack train records: only {len(attack_pool)} available in train pool. "
            "Supervised models (LR, RF, XGBoost) require both classes to train. "
            "Note: Superviz26-SQL default split='train' was authored for one-class models and has 0 attacks."
        )

    rng_benign = random.Random(seed)
    rng_attack = random.Random(seed + 1)
    rng_shuffle = random.Random(seed + 2)

    benign_indices = sorted(rng_benign.sample(range(len(benign_pool)), n_benign))
    attack_indices = sorted(rng_attack.sample(range(len(attack_pool)), n_attack))

    def _ensure_train_record(r: SQLRecord) -> SQLRecord:
        if r.split == "train":
            return r
        return SQLRecord(
            full_query=r.full_query,
            label=r.label,
            split="train",
            domain=r.domain,
            source_file=r.source_file,
            original_split=r.original_split or r.split,
            user_inputs=r.user_inputs,
            attack_stage=r.attack_stage,
            tamper_method=r.tamper_method,
            attack_status=r.attack_status,
            statement_type=r.statement_type,
            query_template_id=r.query_template_id,
            attack_id=r.attack_id,
            attack_technique=r.attack_technique,
            extra_fields=r.extra_fields,
        )

    sampled_benign = [_ensure_train_record(benign_pool[i]) for i in benign_indices]
    sampled_attack = [_ensure_train_record(attack_pool[i]) for i in attack_indices]

    combined = sampled_benign + sampled_attack
    rng_shuffle.shuffle(combined)

    return DataSlice(
        name=slice_name,
        records=combined,
        metadata={
            "sample_type": "stratified_train",
            "requested_total": n_total,
            "target_attack_ratio": attack_ratio,
            "benign_count": n_benign,
            "attack_count": n_attack,
            "benign_pool_size": len(benign_pool),
            "attack_pool_size": len(attack_pool),
            "seed": seed,
        },
    )


def sample_stratified_test(
    records: Sequence[SQLRecord],
    n_total: int,
    attack_ratio: float = 0.10,
    seed: int = 42,
    slice_name: str = "stratified_test_slice",
    require_is_test: bool = True,
) -> DataSlice:
    """Deterministically sample n_total test queries with a fixed attack ratio.

    In Superviz26-SQL, the standard test attack ratio is ~10% (100k attacks / 1M queries).
    For a 5,000 pilot test set with attack_ratio=0.10, this produces exactly 4,500 benign
    queries and 500 SQL injection queries.

    Args:
        records: Source sequence of SQLRecord instances.
        n_total: Total number of test queries to sample.
        attack_ratio: Desired proportion of malicious queries (e.g. 0.10).
        seed: Random seed for deterministic subsampling.
        slice_name: Name for the resulting DataSlice.
        require_is_test: If True, only select from records where r.is_test is True.
            If False, falls back to full pool if test pool is empty.

    Returns:
        DataSlice containing exactly n_total test records.

    Raises:
        ValueError: If n_total <= 0, attack_ratio is invalid, or pools have insufficient rows.
    """
    if n_total <= 0:
        raise ValueError(f"Requested total test sample size n_total must be positive, got {n_total}.")
    if not (0.0 <= attack_ratio <= 1.0):
        raise ValueError(f"Attack ratio must be between 0.0 and 1.0, got {attack_ratio}.")

    test_records = [r for r in records if r.is_test]
    if not test_records and not require_is_test:
        test_records = list(records)

    benign_pool = [r for r in test_records if r.is_benign]
    attack_pool = [r for r in test_records if r.is_attack]

    n_attack = round(n_total * attack_ratio)
    n_benign = n_total - n_attack

    if len(benign_pool) < n_benign:
        raise ValueError(
            f"Cannot sample {n_benign} benign test records: only {len(benign_pool)} available in test pool."
        )
    if len(attack_pool) < n_attack:
        raise ValueError(
            f"Cannot sample {n_attack} attack test records: only {len(attack_pool)} available in test pool."
        )

    # Use distinct, reproducible seeds for benign and attack streams
    rng_benign = random.Random(seed)
    rng_attack = random.Random(seed + 1)
    rng_shuffle = random.Random(seed + 2)

    benign_indices = sorted(rng_benign.sample(range(len(benign_pool)), n_benign))
    attack_indices = sorted(rng_attack.sample(range(len(attack_pool)), n_attack))

    def _ensure_test_record(r: SQLRecord) -> SQLRecord:
        if r.split == "test":
            return r
        return SQLRecord(
            full_query=r.full_query,
            label=r.label,
            split="test",
            domain=r.domain,
            source_file=r.source_file,
            original_split=r.original_split or r.split,
            user_inputs=r.user_inputs,
            attack_stage=r.attack_stage,
            tamper_method=r.tamper_method,
            attack_status=r.attack_status,
            statement_type=r.statement_type,
            query_template_id=r.query_template_id,
            attack_id=r.attack_id,
            attack_technique=r.attack_technique,
            extra_fields=r.extra_fields,
        )

    sampled_benign = [_ensure_test_record(benign_pool[i]) for i in benign_indices]
    sampled_attack = [_ensure_test_record(attack_pool[i]) for i in attack_indices]

    combined = sampled_benign + sampled_attack
    # Deterministic final shuffle to prevent grouping by label
    rng_shuffle.shuffle(combined)

    return DataSlice(
        name=slice_name,
        records=combined,
        metadata={
            "sample_type": "stratified_test",
            "requested_total": n_total,
            "target_attack_ratio": attack_ratio,
            "benign_count": n_benign,
            "attack_count": n_attack,
            "benign_pool_size": len(benign_pool),
            "attack_pool_size": len(attack_pool),
            "seed": seed,
        },
    )


def check_train_test_leakage(
    train_records: Sequence[SQLRecord],
    test_records: Sequence[SQLRecord],
) -> LeakageReport:
    """Audit for train/test data leakage between training and testing queries.

    Checks:
        1. Exact string match between train query and test query.
        2. Normalized string match (trimmed, case-folded whitespace collapse).
        3. Query template overlap (shared query_template_id).

    Args:
        train_records: Training sequence of SQLRecords.
        test_records: Test sequence of SQLRecords.

    Returns:
        LeakageReport with counts and sample overlaps.
    """
    train_queries = {r.full_query for r in train_records}
    test_queries = {r.full_query for r in test_records}

    exact_overlap = train_queries.intersection(test_queries)

    # Normalized comparison: lowercased, multiple spaces collapsed, spaces around delimiters stripped
    def normalize_sql(q: str) -> str:
        s = q.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*([;,()=])\s*", r"\1", s)
        return s.strip()

    train_normalized = {normalize_sql(r.full_query): r.full_query for r in train_records}
    test_normalized = {normalize_sql(r.full_query): r.full_query for r in test_records}
    normalized_overlap = set(train_normalized.keys()).intersection(set(test_normalized.keys()))

    # Template comparison
    train_templates = {r.query_template_id for r in train_records if r.query_template_id}
    test_templates = {r.query_template_id for r in test_records if r.query_template_id}
    template_overlap = train_templates.intersection(test_templates)

    has_leakage = len(exact_overlap) > 0 or len(normalized_overlap) > 0

    return LeakageReport(
        train_size=len(train_records),
        test_size=len(test_records),
        query_overlap_count=len(exact_overlap),
        overlapping_queries_sample=sorted(list(exact_overlap))[:5],
        normalized_query_overlap_count=len(normalized_overlap),
        template_overlap_count=len(template_overlap),
        overlapping_templates_sample=sorted(list(template_overlap))[:5],
        has_leakage=has_leakage,
    )


def verify_test_sets_match(
    slice_a: Union[DataSlice, Sequence[SQLRecord]],
    slice_b: Union[DataSlice, Sequence[SQLRecord]],
) -> MatchReport:
    """Verify that two test slices (e.g. from abc-d.csv and d-d.csv) are identical.

    In the Superviz26-SQL generation protocol (generate_splits.py), test queries
    for domain D are sampled once and copied identically into both abc-d.csv and d-d.csv.
    This function verifies that instance-level alignment is preserved.

    Args:
        slice_a: First test slice or sequence of SQLRecords.
        slice_b: Second test slice or sequence of SQLRecords.

    Returns:
        MatchReport detailing query and label alignment.
    """
    recs_a = slice_a.records if isinstance(slice_a, DataSlice) else list(slice_a)
    recs_b = slice_b.records if isinstance(slice_b, DataSlice) else list(slice_b)

    size_a = len(recs_a)
    size_b = len(recs_b)

    if size_a != size_b:
        return MatchReport(
            size_a=size_a,
            size_b=size_b,
            identical_queries=False,
            identical_labels=False,
            identical_order=False,
            mismatch_count=abs(size_a - size_b),
        )

    mismatches = 0
    identical_queries = True
    identical_labels = True

    for i, (ra, rb) in enumerate(zip(recs_a, recs_b)):
        if ra.full_query != rb.full_query:
            identical_queries = False
            mismatches += 1
        if ra.label != rb.label:
            identical_labels = False

    return MatchReport(
        size_a=size_a,
        size_b=size_b,
        identical_queries=identical_queries,
        identical_labels=identical_labels,
        identical_order=identical_queries and identical_labels,
        mismatch_count=mismatches,
    )


def audit_source_domain_isolation(
    records: Sequence[SQLRecord],
    target_domain: str = "d",
) -> dict[str, Any]:
    """Verify that source-domain records never originate from the target domain."""
    target_clean = target_domain.strip().lower()
    target_aliases = {"d", "ohr"} if target_clean in ("d", "ohr") else {target_clean}
    violations = [
        r for r in records
        if r.domain and r.domain.strip().lower() in target_aliases
    ]
    return {
        "is_isolated": len(violations) == 0,
        "target_violations_count": len(violations),
        "target_domain": target_domain,
        "violation_samples": [v.to_dict() for v in violations[:5]],
    }


def audit_within_domain_isolation(
    records: Sequence[SQLRecord],
    target_domain: str = "d",
) -> dict[str, Any]:
    """Verify that within-domain records originate strictly from the target domain."""
    target_clean = target_domain.strip().lower()
    target_aliases = {"d", "ohr"} if target_clean in ("d", "ohr") else {target_clean}
    foreign = [
        r for r in records
        if r.domain and r.domain.strip().lower() not in target_aliases
    ]
    return {
        "is_pure": len(foreign) == 0,
        "foreign_violations_count": len(foreign),
        "target_domain": target_domain,
        "foreign_samples": [f.to_dict() for f in foreign[:5]],
    }


def create_supervised_lodo_slices(
    source_records: Sequence[SQLRecord],
    target_records: Sequence[SQLRecord],
    target_domain: str = "d",
    train_size: int = 30_000,
    train_attack_ratio: float = 0.10,
    test_size: int = 5_000,
    test_attack_ratio: float = 0.10,
    seed: int = 42,
    replicate_mode: str = "identical_data",
) -> dict[str, Any]:
    """Create supervised LODO training and evaluation slices with strict isolation.

    Protocol Commitments:
    1. Pre-sampling Isolation Audits: Validates source domain isolation and target purity FIRST.
    2. Evaluation Reservation FIRST: Reserves target evaluation slice Q_D^eval before any training sampling.
    3. Evaluation Quarantine: Excludes all Q_D^eval queries from target training pool.
    4. Source-Domain Restriction: Guarantees zero records from target_domain in M_cross training data.
    5. Within-Domain Restriction: Guarantees M_within training data comes only from target_domain.
    6. Disjointness Assertion: Verifies exact and normalized query overlap between train and eval is zero.
    7. Replicate Baseline: within_train_2 shares exact training slice (isolating estimator seed randomness).
    """
    # 1. Pre-sampling Domain Audits FIRST
    source_domain_audit = audit_source_domain_isolation(source_records, target_domain=target_domain)
    if not source_domain_audit["is_isolated"]:
        raise ValueError(
            f"Source records contain {source_domain_audit['target_violations_count']} queries from target domain '{target_domain}'! Cross-domain evaluation violated."
        )

    within_domain_audit_pre = audit_within_domain_isolation(target_records, target_domain=target_domain)
    if not within_domain_audit_pre["is_pure"]:
        raise ValueError(
            f"Target pool contains foreign domain records: {within_domain_audit_pre['foreign_violations_count']} violations found."
        )

    # 2. Target Evaluation Reservation FIRST
    eval_slice = sample_stratified_test(
        target_records,
        n_total=test_size,
        attack_ratio=test_attack_ratio,
        seed=seed,
        slice_name=f"supervised_eval_{target_domain}",
        require_is_test=False,
    )

    # 3. Evaluation Quarantine: remove all eval queries from target training pool
    eval_query_set = {r.full_query for r in eval_slice.records}
    def normalize_sql(q: str) -> str:
        s = q.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*([;,()=])\s*", r"\1", s)
        return s.strip()
    eval_norm_set = {normalize_sql(r.full_query) for r in eval_slice.records}

    target_train_pool = [
        r for r in target_records
        if r.full_query not in eval_query_set and normalize_sql(r.full_query) not in eval_norm_set
    ]

    # 4. Within-Domain Training Sampling (M_within)
    within_train_1 = sample_stratified_train(
        target_train_pool,
        n_total=train_size,
        attack_ratio=train_attack_ratio,
        seed=seed,
        slice_name=f"supervised_within_{target_domain}_seed1",
        require_is_train=False,
    )

    if replicate_mode == "identical_data":
        within_train_2 = within_train_1
    else:
        within_train_2 = sample_stratified_train(
            target_train_pool,
            n_total=train_size,
            attack_ratio=train_attack_ratio,
            seed=seed + 81,
            slice_name=f"supervised_within_{target_domain}_seed2",
            require_is_train=False,
        )

    # 5. Source-Domain Training Sampling (M_cross)
    cross_train = sample_stratified_train(
        source_records,
        n_total=train_size,
        attack_ratio=train_attack_ratio,
        seed=seed,
        slice_name="supervised_cross_train",
        require_is_train=False,
    )

    # 6. Overlap and Leakage Audits
    leakage_cross = check_train_test_leakage(cross_train.records, eval_slice.records)
    leakage_within = check_train_test_leakage(within_train_1.records, eval_slice.records)

    if leakage_cross.has_leakage:
        raise ValueError("Data leakage detected between cross-domain train and target evaluation queries!")
    if leakage_within.has_leakage:
        raise ValueError("Data leakage detected between within-domain train and target evaluation queries!")

    # 6. Provenance Reporting
    provenance = {
        "target_domain": target_domain,
        "eval_size": len(eval_slice),
        "eval_domain_distribution": eval_slice.domain_distribution,
        "eval_attack_ratio": eval_slice.attack_ratio,
        "cross_train_size": len(cross_train),
        "cross_train_domain_distribution": cross_train.domain_distribution,
        "cross_train_original_split_distribution": cross_train.original_split_distribution,
        "within_train_size": len(within_train_1),
        "within_train_domain_distribution": within_train_1.domain_distribution,
        "within_train_original_split_distribution": within_train_1.original_split_distribution,
        "leakage_cross": leakage_cross.summary(),
        "leakage_within": leakage_within.summary(),
        "source_domain_isolation": source_domain_audit,
        "within_domain_isolation": within_domain_audit_pre,
    }

    return {
        "cross_train": cross_train,
        "within_train_1": within_train_1,
        "within_train_2": within_train_2,
        "eval_slice": eval_slice,
        "provenance": provenance,
    }


class DataLoader:
    """High-level dataset coordinator for Superviz26-SQL pilot and full evaluations.

    Manages loading of cross-domain (abc-d) and within-domain (d-d) data slices,
    deterministic subsampling, schema validation, and leakage auditing.
    """

    def __init__(
        self,
        cross_domain_path: Optional[Union[str, Path]] = None,
        within_domain_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.cross_domain_path = Path(cross_domain_path) if cross_domain_path else None
        self.within_domain_path = Path(within_domain_path) if within_domain_path else None

    def check_files_exist(self) -> dict[str, bool]:
        """Check whether configured dataset files exist on disk."""
        return {
            "cross_domain_exists": bool(self.cross_domain_path and self.cross_domain_path.is_file()),
            "within_domain_exists": bool(self.within_domain_path and self.within_domain_path.is_file()),
        }

    def load_lodo_pilot_slices(
        self,
        train_size: int = 30_000,
        train_attack_ratio: Optional[float] = 0.10,
        test_size: int = 5_000,
        test_attack_ratio: float = 0.10,
        seed: int = 42,
        replicate_mode: str = "identical_data",
        train_benign_size: Optional[int] = None,
        attack_ratio: Optional[float] = None,
    ) -> dict[str, Any]:
        """Load and sample reproducible data slices for the pilot experiment.

        Supports both supervised binary classification models (LR, RF, XGBoost)
        with stratified training samples (both benign and attack queries), as well as
        one-class anomaly baselines (when train_attack_ratio=0.0).

        Replication modes for within-domain reference baseline:
            - 'identical_data' (Recommended): within_train_2 shares the exact same training
              instances as within_train_1, differing only in estimator seed (isolates model randomness).
            - 'resampled_data': within_train_2 draws an independent subsample using seed+81.

        Raises:
            FileNotFoundError: If either dataset file does not exist.
            ValueError: If sampling requirements or schema validations fail.
        """
        if self.cross_domain_path is None or not self.cross_domain_path.is_file():
            raise FileNotFoundError(
                f"Cross-domain dataset file not found: {self.cross_domain_path}. "
                "Download from Zenodo (DOI: 10.5281/zenodo.19627322) before executing."
            )
        if self.within_domain_path is None or not self.within_domain_path.is_file():
            raise FileNotFoundError(
                f"Within-domain dataset file not found: {self.within_domain_path}. "
                "Download from Zenodo (DOI: 10.5281/zenodo.19627322) before executing."
            )

        # Handle backwards compatibility
        actual_train_size = train_benign_size if train_benign_size is not None else train_size
        actual_train_attack_ratio = 0.0 if train_benign_size is not None else train_attack_ratio
        actual_test_attack_ratio = attack_ratio if attack_ratio is not None else test_attack_ratio

        # 1. Load both datasets
        cross_train_raw, cross_test_raw = load_dataset_splits(self.cross_domain_path)
        within_train_raw, within_test_raw = load_dataset_splits(self.within_domain_path)

        # 2. Audit test set identity between abc-d and d-d
        test_match_report = verify_test_sets_match(cross_test_raw, within_test_raw)

        # 3. Sample test set from target domain D
        test_slice = sample_stratified_test(
            cross_test_raw.records,
            n_total=test_size,
            attack_ratio=actual_test_attack_ratio,
            seed=seed,
            slice_name="pilot_test_d",
        )

        # 4. Sample training sets (supervised if train_attack_ratio > 0, benign-only otherwise)
        if actual_train_attack_ratio is not None and actual_train_attack_ratio > 0.0:
            cross_train_slice = sample_stratified_train(
                cross_train_raw.records,
                n_total=actual_train_size,
                attack_ratio=actual_train_attack_ratio,
                seed=seed,
                slice_name="pilot_cross_train_abc",
            )
            within_train_slice_1 = sample_stratified_train(
                within_train_raw.records,
                n_total=actual_train_size,
                attack_ratio=actual_train_attack_ratio,
                seed=seed,
                slice_name="pilot_within_train_d_seed1",
            )
            if replicate_mode == "identical_data":
                within_train_slice_2 = within_train_slice_1
            else:
                within_train_slice_2 = sample_stratified_train(
                    within_train_raw.records,
                    n_total=actual_train_size,
                    attack_ratio=actual_train_attack_ratio,
                    seed=seed + 81,
                    slice_name="pilot_within_train_d_seed2",
                )
        else:
            cross_train_slice = sample_benign_train(
                cross_train_raw.records,
                n=actual_train_size,
                seed=seed,
                slice_name="pilot_cross_train_abc",
            )
            within_train_slice_1 = sample_benign_train(
                within_train_raw.records,
                n=actual_train_size,
                seed=seed,
                slice_name="pilot_within_train_d_seed1",
            )
            if replicate_mode == "identical_data":
                within_train_slice_2 = within_train_slice_1
            else:
                within_train_slice_2 = sample_benign_train(
                    within_train_raw.records,
                    n=actual_train_size,
                    seed=seed + 81,
                    slice_name="pilot_within_train_d_seed2",
                )

        # 5. Audit leakage
        leakage_cross = check_train_test_leakage(cross_train_slice.records, test_slice.records)
        leakage_within = check_train_test_leakage(within_train_slice_1.records, test_slice.records)

        return {
            "cross_train": cross_train_slice,
            "within_train_1": within_train_slice_1,
            "within_train_2": within_train_slice_2,
            "test_slice": test_slice,
            "audit": {
                "test_match_report": test_match_report,
                "leakage_cross": leakage_cross,
                "leakage_within": leakage_within,
            },
        }
