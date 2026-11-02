"""Unit tests for src/data/loader.py.

Uses small synthetic CSV fixtures created in temporary directories to test:
- Schema validation (header checking, missing required columns, extra columns)
- Label mapping and strict validation (0=benign, 1=attack, reject invalid)
- Split validation ('train' or 'test', reject invalid)
- Deterministic sampling (exact repeatability, seed sensitivity)
- Class count exactness in stratified test sampling and benign train sampling
- Missing file and empty file handling
- Accidental train/test leakage detection (exact, normalized, template)
- Cross-domain vs within-domain test set identity verification
- DataLoader high-level orchestration
"""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from src.data.loader import (
    CANONICAL_COLUMNS,
    DataLoader,
    DataSlice,
    LeakageReport,
    MatchReport,
    SQLRecord,
    check_train_test_leakage,
    load_csv_records,
    load_dataset_splits,
    parse_label,
    parse_split,
    sample_benign_train,
    sample_stratified_test,
    sample_stratified_train,
    validate_header,
    verify_test_sets_match,
    audit_source_domain_isolation,
    audit_within_domain_isolation,
    create_supervised_lodo_slices,
)


class TestSchemaAndParsing(unittest.TestCase):
    """Test schema header validation, row parsing, and label mapping."""

    def test_validate_header_success(self) -> None:
        valid_header = ["full_query", "label", "split"]
        clean = validate_header(valid_header)
        self.assertEqual(clean, ["full_query", "label", "split"])

    def test_validate_header_with_all_canonical_columns(self) -> None:
        clean = validate_header(CANONICAL_COLUMNS)
        self.assertEqual(clean, CANONICAL_COLUMNS)

    def test_validate_header_missing_required_column_raises(self) -> None:
        # Missing 'split'
        with self.assertRaises(ValueError) as ctx:
            validate_header(["full_query", "label", "statement_type"])
        self.assertIn("missing required column", str(ctx.exception).lower())
        self.assertIn("split", str(ctx.exception))

    def test_validate_header_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_header([])

    def test_parse_label_valid(self) -> None:
        self.assertEqual(parse_label(0, 1), 0)
        self.assertEqual(parse_label("0", 2), 0)
        self.assertEqual(parse_label("benign", 3), 0)
        self.assertEqual(parse_label("normal", 4), 0)
        self.assertEqual(parse_label(1, 5), 1)
        self.assertEqual(parse_label("1", 6), 1)
        self.assertEqual(parse_label("attack", 7), 1)
        self.assertEqual(parse_label("sqli", 8), 1)

    def test_parse_label_invalid_raises(self) -> None:
        invalid_labels = [2, -1, "unknown", "", None, 3.5, "malware"]
        for inv in invalid_labels:
            with self.subTest(label=inv):
                with self.assertRaises(ValueError):
                    parse_label(inv, 1)

    def test_parse_split_valid(self) -> None:
        self.assertEqual(parse_split("train", 1), "train")
        self.assertEqual(parse_split("test", 2), "test")
        self.assertEqual(parse_split("TRAIN", 3), "train")
        self.assertEqual(parse_split("Test ", 4), "test")

    def test_parse_split_invalid_raises(self) -> None:
        invalid_splits = ["validation", "val", "dev", "", None, 123]
        for inv in invalid_splits:
            with self.subTest(split=inv):
                with self.assertRaises(ValueError):
                    parse_split(inv, 1)


class TestCSVLoading(unittest.TestCase):
    """Test reading CSV files with synthetic test fixtures."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_synthetic_csv(
        self,
        filename: str,
        rows: list[dict],
        fieldnames: list[str] | None = None,
    ) -> Path:
        filepath = self.temp_path / filename
        if fieldnames is None:
            fieldnames = CANONICAL_COLUMNS

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return filepath

    def test_missing_file_raises_filenotfound(self) -> None:
        missing_path = self.temp_path / "nonexistent.csv"
        with self.assertRaises(FileNotFoundError) as ctx:
            load_csv_records(missing_path)
        self.assertIn("Dataset file not found", str(ctx.exception))

    def test_empty_csv_file_raises_valueerror(self) -> None:
        empty_path = self.temp_path / "empty.csv"
        empty_path.touch()
        with self.assertRaises(ValueError) as ctx:
            load_csv_records(empty_path)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_load_valid_synthetic_dataset(self) -> None:
        rows = [
            {
                "full_query": "SELECT * FROM employees WHERE id = 1;",
                "label": "0",
                "split": "train",
                "statement_type": "select",
                "query_template_id": "T01",
                "attack_technique": "",
            },
            {
                "full_query": "SELECT * FROM employees WHERE id = 1 OR 1=1;",
                "label": "1",
                "split": "test",
                "statement_type": "select",
                "query_template_id": "T01",
                "attack_technique": "boolean",
            },
            {
                "full_query": "UPDATE salary SET amount = 5000 WHERE id = 2;",
                "label": "0",
                "split": "test",
                "statement_type": "update",
                "query_template_id": "T02",
                "attack_technique": "",
            },
        ]
        csv_path = self._create_synthetic_csv("valid.csv", rows)
        records = load_csv_records(csv_path)

        self.assertEqual(len(records), 3)
        self.assertTrue(records[0].is_train)
        self.assertTrue(records[0].is_benign)
        self.assertFalse(records[0].is_attack)
        self.assertEqual(records[0].statement_type, "select")

        self.assertTrue(records[1].is_test)
        self.assertTrue(records[1].is_attack)
        self.assertEqual(records[1].attack_technique, "boolean")

        self.assertTrue(records[2].is_test)
        self.assertTrue(records[2].is_benign)
        self.assertEqual(records[2].statement_type, "update")

    def test_load_dataset_splits(self) -> None:
        rows = [
            {"full_query": f"SELECT {i};", "label": "0", "split": "train"} for i in range(5)
        ] + [
            {"full_query": f"SELECT {i};", "label": "0", "split": "test"} for i in range(5, 8)
        ] + [
            {"full_query": f"SELECT {i} OR 1=1;", "label": "1", "split": "test"} for i in range(8, 10)
        ]
        csv_path = self._create_synthetic_csv("splits.csv", rows, fieldnames=["full_query", "label", "split"])
        train_slice, test_slice = load_dataset_splits(csv_path)

        self.assertEqual(len(train_slice), 5)
        self.assertEqual(len(test_slice), 5)
        self.assertEqual(train_slice.benign_count, 5)
        self.assertEqual(train_slice.attack_count, 0)
        self.assertEqual(test_slice.benign_count, 3)
        self.assertEqual(test_slice.attack_count, 2)


class TestDeterministicSampling(unittest.TestCase):
    """Test deterministic subsampling functions with exact seed behavior."""

    def setUp(self) -> None:
        # Build synthetic pool of 100 train records (all benign)
        # and 100 test records (80 benign, 20 attack)
        self.records: list[SQLRecord] = []
        for i in range(100):
            self.records.append(
                SQLRecord(
                    full_query=f"SELECT train_col FROM table WHERE id = {i};",
                    label=0,
                    split="train",
                    query_template_id=f"T_train_{i % 5}",
                )
            )
        for i in range(80):
            self.records.append(
                SQLRecord(
                    full_query=f"SELECT test_col FROM table WHERE val = {i};",
                    label=0,
                    split="test",
                    query_template_id=f"T_test_benign_{i % 4}",
                )
            )
        for i in range(20):
            self.records.append(
                SQLRecord(
                    full_query=f"SELECT test_col FROM table WHERE val = {i} UNION SELECT 1;",
                    label=1,
                    split="test",
                    attack_technique="union",
                    query_template_id=f"T_test_attack_{i % 2}",
                )
            )
        for i in range(25):
            self.records.append(
                SQLRecord(
                    full_query=f"SELECT train_atk FROM table WHERE id = {i} OR 1=1;",
                    label=1,
                    split="train",
                    attack_technique="boolean",
                    query_template_id=f"T_train_atk_{i % 3}",
                )
            )

    def test_sample_benign_train_exact_count(self) -> None:
        slice_result = sample_benign_train(self.records, n=30, seed=42)
        self.assertEqual(len(slice_result), 30)
        self.assertEqual(slice_result.benign_count, 30)
        self.assertEqual(slice_result.attack_count, 0)
        for r in slice_result:
            self.assertTrue(r.is_train)
            self.assertTrue(r.is_benign)

    def test_sample_stratified_train_exact_counts(self) -> None:
        # Request 40 train samples with 25% attack ratio -> 30 benign, 10 attack
        train_slice = sample_stratified_train(
            self.records, n_total=40, attack_ratio=0.25, seed=42
        )
        self.assertEqual(len(train_slice), 40)
        self.assertEqual(train_slice.benign_count, 30)
        self.assertEqual(train_slice.attack_count, 10)
        self.assertAlmostEqual(train_slice.attack_ratio, 0.25, places=4)
        for r in train_slice:
            self.assertTrue(r.is_train)

    def test_sample_stratified_train_repeatability(self) -> None:
        slice_1 = sample_stratified_train(self.records, n_total=30, attack_ratio=0.20, seed=42)
        slice_2 = sample_stratified_train(self.records, n_total=30, attack_ratio=0.20, seed=42)
        slice_3 = sample_stratified_train(self.records, n_total=30, attack_ratio=0.20, seed=999)
        self.assertEqual(slice_1.queries, slice_2.queries)
        self.assertEqual(slice_1.labels, slice_2.labels)
        self.assertNotEqual(slice_1.queries, slice_3.queries)

    def test_sample_stratified_train_pool_exhaustion_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            sample_stratified_train(self.records, n_total=60, attack_ratio=0.60, seed=42)
        self.assertIn("attack train records", str(ctx.exception).lower())

    def test_sample_stratified_train_zero_attacks_in_pool_raises(self) -> None:
        benign_only_records = [r for r in self.records if r.is_benign]
        with self.assertRaises(ValueError) as ctx:
            sample_stratified_train(benign_only_records, n_total=20, attack_ratio=0.10, seed=42)
        self.assertIn("only 0 available in train pool", str(ctx.exception).lower())

    def test_sample_benign_train_repeatability(self) -> None:
        slice_1 = sample_benign_train(self.records, n=25, seed=42)
        slice_2 = sample_benign_train(self.records, n=25, seed=42)
        slice_3 = sample_benign_train(self.records, n=25, seed=999)

        # Same seed must yield identical query sequence
        self.assertEqual(slice_1.queries, slice_2.queries)
        # Different seed must yield distinct sample
        self.assertNotEqual(slice_1.queries, slice_3.queries)

    def test_sample_benign_train_insufficient_pool_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            sample_benign_train(self.records, n=150, seed=42)
        self.assertIn("Cannot sample 150 benign train records", str(ctx.exception))

    def test_sample_benign_train_invalid_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            sample_benign_train(self.records, n=0, seed=42)
        with self.assertRaises(ValueError):
            sample_benign_train(self.records, n=-10, seed=42)

    def test_sample_stratified_test_exact_proportions(self) -> None:
        # Total 50 test samples with 20% attack ratio -> 40 benign, 10 attack
        test_slice = sample_stratified_test(
            self.records, n_total=50, attack_ratio=0.20, seed=42
        )
        self.assertEqual(len(test_slice), 50)
        self.assertEqual(test_slice.benign_count, 40)
        self.assertEqual(test_slice.attack_count, 10)
        self.assertAlmostEqual(test_slice.attack_ratio, 0.20, places=4)
        for r in test_slice:
            self.assertTrue(r.is_test)

    def test_sample_stratified_test_repeatability(self) -> None:
        slice_a = sample_stratified_test(self.records, n_total=40, attack_ratio=0.10, seed=123)
        slice_b = sample_stratified_test(self.records, n_total=40, attack_ratio=0.10, seed=123)
        slice_c = sample_stratified_test(self.records, n_total=40, attack_ratio=0.10, seed=456)

        self.assertEqual(slice_a.queries, slice_b.queries)
        self.assertEqual(slice_a.labels, slice_b.labels)
        self.assertNotEqual(slice_a.queries, slice_c.queries)

    def test_sample_stratified_test_pool_exhaustion_raises(self) -> None:
        # Only 20 attacks available; requesting 30 must raise
        with self.assertRaises(ValueError) as ctx:
            sample_stratified_test(self.records, n_total=60, attack_ratio=0.50, seed=42)
        self.assertIn("attack test records", str(ctx.exception).lower())

    def test_sample_stratified_test_invalid_args_raises(self) -> None:
        with self.assertRaises(ValueError):
            sample_stratified_test(self.records, n_total=-5, seed=42)
        with self.assertRaises(ValueError):
            sample_stratified_test(self.records, n_total=10, attack_ratio=1.5, seed=42)


class TestLeakageAndMatching(unittest.TestCase):
    """Test train/test leakage detection and test set identity verification."""

    def test_leakage_detection_no_leakage(self) -> None:
        train_recs = [
            SQLRecord(full_query="SELECT a FROM table1;", label=0, split="train", query_template_id="T1"),
            SQLRecord(full_query="SELECT b FROM table1;", label=0, split="train", query_template_id="T2"),
        ]
        test_recs = [
            SQLRecord(full_query="SELECT x FROM table2;", label=0, split="test", query_template_id="T3"),
            SQLRecord(full_query="SELECT y FROM table2 WHERE 1=1;", label=1, split="test", query_template_id="T4"),
        ]
        report = check_train_test_leakage(train_recs, test_recs)
        self.assertFalse(report.has_leakage)
        self.assertEqual(report.query_overlap_count, 0)
        self.assertEqual(report.normalized_query_overlap_count, 0)
        self.assertEqual(report.template_overlap_count, 0)

    def test_leakage_detection_exact_leakage_found(self) -> None:
        shared_query = "SELECT common_id FROM shared_table WHERE val = 1;"
        train_recs = [
            SQLRecord(full_query=shared_query, label=0, split="train"),
            SQLRecord(full_query="SELECT unique_train FROM t1;", label=0, split="train"),
        ]
        test_recs = [
            SQLRecord(full_query=shared_query, label=0, split="test"),
            SQLRecord(full_query="SELECT unique_test FROM t2;", label=0, split="test"),
        ]
        report = check_train_test_leakage(train_recs, test_recs)
        self.assertTrue(report.has_leakage)
        self.assertEqual(report.query_overlap_count, 1)
        self.assertEqual(report.overlapping_queries_sample, [shared_query])

    def test_leakage_detection_normalized_whitespace_leakage(self) -> None:
        train_recs = [
            SQLRecord(full_query="SELECT  id   FROM table;", label=0, split="train"),
        ]
        test_recs = [
            SQLRecord(full_query="select id from table ;", label=0, split="test"),
        ]
        report = check_train_test_leakage(train_recs, test_recs)
        self.assertTrue(report.has_leakage)
        self.assertEqual(report.normalized_query_overlap_count, 1)

    def test_verify_test_sets_match_exact_identity(self) -> None:
        test_recs_a = [
            SQLRecord(full_query="SELECT 1;", label=0, split="test"),
            SQLRecord(full_query="SELECT 2 UNION SELECT 3;", label=1, split="test"),
        ]
        test_recs_b = [
            SQLRecord(full_query="SELECT 1;", label=0, split="test"),
            SQLRecord(full_query="SELECT 2 UNION SELECT 3;", label=1, split="test"),
        ]
        report = verify_test_sets_match(test_recs_a, test_recs_b)
        self.assertTrue(report.is_exact_match)
        self.assertEqual(report.mismatch_count, 0)

    def test_verify_test_sets_match_mismatches_detected(self) -> None:
        test_recs_a = [
            SQLRecord(full_query="SELECT 1;", label=0, split="test"),
            SQLRecord(full_query="SELECT 2;", label=0, split="test"),
        ]
        test_recs_b = [
            SQLRecord(full_query="SELECT 1;", label=0, split="test"),
            SQLRecord(full_query="SELECT DIFFERENT;", label=0, split="test"),
        ]
        report = verify_test_sets_match(test_recs_a, test_recs_b)
        self.assertFalse(report.is_exact_match)
        self.assertEqual(report.mismatch_count, 1)


class TestDataLoaderOrchestrator(unittest.TestCase):
    """Test the high-level DataLoader class using synthetic LODO CSVs."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_check_files_exist(self) -> None:
        cross_path = self.temp_path / "abc-d.csv"
        within_path = self.temp_path / "d-d.csv"

        loader = DataLoader(cross_domain_path=cross_path, within_domain_path=within_path)
        status = loader.check_files_exist()
        self.assertFalse(status["cross_domain_exists"])
        self.assertFalse(status["within_domain_exists"])

        # Touch one file
        cross_path.touch()
        status_updated = loader.check_files_exist()
        self.assertTrue(status_updated["cross_domain_exists"])
        self.assertFalse(status_updated["within_domain_exists"])

    def test_load_lodo_pilot_slices_full_pipeline(self) -> None:
        # Create synthetic abc-d.csv and d-d.csv
        # In Superviz26-SQL, the test sets are identical between abc-d and d-d
        cross_path = self.temp_path / "abc-d.csv"
        within_path = self.temp_path / "d-d.csv"

        fieldnames = CANONICAL_COLUMNS

        # Test set queries (shared)
        test_rows = [
            {"full_query": f"SELECT test_d_{i} FROM t_d;", "label": "0", "split": "test", "statement_type": "select"}
            for i in range(50)
        ] + [
            {"full_query": f"SELECT test_d_{i} OR 1=1;", "label": "1", "split": "test", "attack_technique": "boolean"}
            for i in range(10)
        ]

        # Cross train rows (from domains A, B, C)
        cross_train_rows = [
            {"full_query": f"SELECT train_abc_{i} FROM t_src;", "label": "0", "split": "train", "statement_type": "select"}
            for i in range(80)
        ]

        # Within train rows (from domain D)
        within_train_rows = [
            {"full_query": f"SELECT train_d_{i} FROM t_d;", "label": "0", "split": "train", "statement_type": "select"}
            for i in range(80)
        ]

        def write_file(p: Path, rows: list[dict]) -> None:
            with open(p, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)

        write_file(cross_path, cross_train_rows + test_rows)
        write_file(within_path, within_train_rows + test_rows)

        loader = DataLoader(cross_domain_path=cross_path, within_domain_path=within_path)

        result = loader.load_lodo_pilot_slices(
            train_benign_size=40,
            test_size=30,
            attack_ratio=0.10,  # 3 attacks, 27 benign
            seed=42,
        )

        self.assertIn("cross_train", result)
        self.assertIn("within_train_1", result)
        self.assertIn("within_train_2", result)
        self.assertIn("test_slice", result)
        self.assertIn("audit", result)

        cross_train = result["cross_train"]
        within_train_1 = result["within_train_1"]
        within_train_2 = result["within_train_2"]
        test_slice = result["test_slice"]

        # Check sizes and counts
        self.assertEqual(len(cross_train), 40)
        self.assertEqual(cross_train.benign_count, 40)
        self.assertEqual(len(within_train_1), 40)
        self.assertEqual(len(within_train_2), 40)
        self.assertEqual(len(test_slice), 30)
        self.assertEqual(test_slice.attack_count, 3)
        self.assertEqual(test_slice.benign_count, 27)

        # Under default 'identical_data' mode, within_train_1 and within_train_2 share identical training queries
        self.assertEqual(within_train_1.queries, within_train_2.queries)

        # Audit reports
        audit = result["audit"]
        self.assertTrue(audit["test_match_report"].is_exact_match)
        self.assertFalse(audit["leakage_cross"].has_leakage)
        self.assertFalse(audit["leakage_within"].has_leakage)

    def test_load_lodo_pilot_slices_supervised_pipeline(self) -> None:
        """Test the supervised training pipeline with both benign and attack queries."""
        cross_path = self.temp_path / "abc-d_sup.csv"
        within_path = self.temp_path / "d-d_sup.csv"

        fieldnames = CANONICAL_COLUMNS

        # Shared test rows (from domain D)
        test_rows = [
            {"full_query": f"SELECT test_sup_{i} FROM t_d;", "label": "0", "split": "test", "statement_type": "select"}
            for i in range(50)
        ] + [
            {"full_query": f"SELECT test_sup_{i} OR 1=1;", "label": "1", "split": "test", "attack_technique": "boolean"}
            for i in range(10)
        ]

        # Cross train rows (benign + attack from source domains A, B, C)
        cross_train_rows = [
            {"full_query": f"SELECT train_abc_norm_{i} FROM t_src;", "label": "0", "split": "train", "statement_type": "select"}
            for i in range(60)
        ] + [
            {"full_query": f"SELECT train_abc_atk_{i} UNION SELECT 1;", "label": "1", "split": "train", "attack_technique": "union"}
            for i in range(20)
        ]

        # Within train rows (benign + attack from target domain D)
        within_train_rows = [
            {"full_query": f"SELECT train_d_norm_{i} FROM t_d;", "label": "0", "split": "train", "statement_type": "select"}
            for i in range(60)
        ] + [
            {"full_query": f"SELECT train_d_atk_{i} AND 1=1;", "label": "1", "split": "train", "attack_technique": "boolean"}
            for i in range(20)
        ]

        def write_file(p: Path, rows: list[dict]) -> None:
            with open(p, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)

        write_file(cross_path, cross_train_rows + test_rows)
        write_file(within_path, within_train_rows + test_rows)

        loader = DataLoader(cross_domain_path=cross_path, within_domain_path=within_path)

        result = loader.load_lodo_pilot_slices(
            train_size=40,
            train_attack_ratio=0.25,  # 10 attacks, 30 benign
            test_size=30,
            test_attack_ratio=0.10,   # 3 attacks, 27 benign
            seed=42,
            replicate_mode="identical_data",  # Recommended design: isolates model seed randomness
        )

        cross_train = result["cross_train"]
        within_train_1 = result["within_train_1"]
        within_train_2 = result["within_train_2"]
        test_slice = result["test_slice"]

        # Supervised train class counts: both classes must be present
        self.assertEqual(len(cross_train), 40)
        self.assertEqual(cross_train.benign_count, 30)
        self.assertEqual(cross_train.attack_count, 10)
        self.assertAlmostEqual(cross_train.attack_ratio, 0.25, places=4)

        self.assertEqual(len(within_train_1), 40)
        self.assertEqual(within_train_1.benign_count, 30)
        self.assertEqual(within_train_1.attack_count, 10)

        # In 'identical_data' mode, within_train_2 has the exact same queries as within_train_1
        self.assertEqual(within_train_1.queries, within_train_2.queries)

        # Test set counts
        self.assertEqual(len(test_slice), 30)
        self.assertEqual(test_slice.attack_count, 3)
        self.assertEqual(test_slice.benign_count, 27)

        # Audit reports
        audit = result["audit"]
        self.assertTrue(audit["test_match_report"].is_exact_match)
        self.assertFalse(audit["leakage_cross"].has_leakage)
        self.assertFalse(audit["leakage_within"].has_leakage)

    def test_load_lodo_pilot_slices_supervised_resampled_mode(self) -> None:
        """Test replicate_mode='resampled_data' draws distinct training samples."""
        cross_path = self.temp_path / "abc-d_resamp.csv"
        within_path = self.temp_path / "d-d_resamp.csv"
        fieldnames = CANONICAL_COLUMNS

        test_rows = [
            {"full_query": f"SELECT test_r_{i} FROM t_d;", "label": "0", "split": "test"} for i in range(20)
        ] + [
            {"full_query": f"SELECT test_r_{i} OR 1=1;", "label": "1", "split": "test"} for i in range(5)
        ]
        train_rows = [
            {"full_query": f"SELECT train_r_{i} FROM t;", "label": "0", "split": "train"} for i in range(50)
        ] + [
            {"full_query": f"SELECT train_r_{i} OR 1=1;", "label": "1", "split": "train"} for i in range(20)
        ]

        def write_file(p: Path, rows: list[dict]) -> None:
            with open(p, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)

        write_file(cross_path, train_rows + test_rows)
        write_file(within_path, train_rows + test_rows)

        loader = DataLoader(cross_domain_path=cross_path, within_domain_path=within_path)
        result = loader.load_lodo_pilot_slices(
            train_size=20,
            train_attack_ratio=0.20,
            test_size=10,
            test_attack_ratio=0.10,
            seed=42,
            replicate_mode="resampled_data",
        )
        within_1 = result["within_train_1"]
        within_2 = result["within_train_2"]
        self.assertNotEqual(within_1.queries, within_2.queries)


class TestSupervisedEvaluationIsolation(unittest.TestCase):
    """Test provenance tracking, source-domain restriction, and evaluation isolation."""

    def setUp(self) -> None:
        # Build synthetic source records from domains A, B, C (benign + attack)
        self.source_records: list[SQLRecord] = []
        for dom in ["a", "b", "c"]:
            for i in range(30):
                self.source_records.append(
                    SQLRecord(
                        full_query=f"SELECT src_{dom}_{i} FROM tbl_{dom};",
                        label=0,
                        split="train",
                        domain=dom,
                        original_split="train",
                    )
                )
            for i in range(10):
                self.source_records.append(
                    SQLRecord(
                        full_query=f"SELECT src_{dom}_atk_{i} FROM tbl_{dom} UNION SELECT 1;",
                        label=1,
                        split="train",
                        domain=dom,
                        original_split="test",  # original provenance from raw attack pool
                        attack_technique="union",
                    )
                )

        # Build synthetic target records from domain D (benign + attack)
        self.target_records: list[SQLRecord] = []
        for i in range(100):
            self.target_records.append(
                SQLRecord(
                    full_query=f"SELECT ohr_norm_{i} FROM employees WHERE id={i};",
                    label=0,
                    split="test",
                    domain="d",
                    original_split="test",
                )
            )
        for i in range(30):
            self.target_records.append(
                SQLRecord(
                    full_query=f"SELECT ohr_atk_{i} FROM employees WHERE id={i} OR 1=1;",
                    label=1,
                    split="test",
                    domain="d",
                    original_split="test",
                    attack_technique="boolean",
                )
            )

    def test_audit_source_domain_isolation_passes(self) -> None:
        audit = audit_source_domain_isolation(self.source_records, target_domain="d")
        self.assertTrue(audit["is_isolated"])
        self.assertEqual(audit["target_violations_count"], 0)

    def test_audit_source_domain_isolation_fails_on_leak(self) -> None:
        leaked = list(self.source_records) + [
            SQLRecord(full_query="SELECT leaked FROM ohr;", label=1, split="train", domain="d")
        ]
        audit = audit_source_domain_isolation(leaked, target_domain="d")
        self.assertFalse(audit["is_isolated"])
        self.assertEqual(audit["target_violations_count"], 1)

    def test_audit_within_domain_isolation_passes(self) -> None:
        audit = audit_within_domain_isolation(self.target_records, target_domain="d")
        self.assertTrue(audit["is_pure"])
        self.assertEqual(audit["foreign_violations_count"], 0)

    def test_audit_within_domain_isolation_fails_on_foreign_data(self) -> None:
        impure = list(self.target_records) + [
            SQLRecord(full_query="SELECT impure FROM sakila;", label=0, split="train", domain="b")
        ]
        audit = audit_within_domain_isolation(impure, target_domain="d")
        self.assertFalse(audit["is_pure"])
        self.assertEqual(audit["foreign_violations_count"], 1)

    def test_create_supervised_lodo_slices_end_to_end(self) -> None:
        result = create_supervised_lodo_slices(
            source_records=self.source_records,
            target_records=self.target_records,
            target_domain="d",
            train_size=40,
            train_attack_ratio=0.25,  # 10 attack, 30 benign
            test_size=20,
            test_attack_ratio=0.20,   # 4 attack, 16 benign
            seed=42,
            replicate_mode="identical_data",
        )

        cross_train = result["cross_train"]
        within_train_1 = result["within_train_1"]
        within_train_2 = result["within_train_2"]
        eval_slice = result["eval_slice"]
        provenance = result["provenance"]

        # Sizes
        self.assertEqual(len(cross_train), 40)
        self.assertEqual(cross_train.attack_count, 10)
        self.assertEqual(len(within_train_1), 40)
        self.assertEqual(within_train_1.attack_count, 10)
        self.assertEqual(len(eval_slice), 20)
        self.assertEqual(eval_slice.attack_count, 4)

        # Domain checks
        self.assertNotIn("d", cross_train.domain_distribution)
        self.assertIn("d", within_train_1.domain_distribution)
        self.assertEqual(within_train_1.domain_distribution.get("d"), 40)

        # Disjointness: within_train and eval_slice MUST have ZERO overlap
        overlap_exact = set(within_train_1.queries) & set(eval_slice.queries)
        self.assertEqual(len(overlap_exact), 0)

        # Replication: within_train_2 matches within_train_1 under identical_data mode
        self.assertEqual(within_train_1.queries, within_train_2.queries)

        # Provenance audit shows clean isolation
        self.assertTrue(provenance["source_domain_isolation"]["is_isolated"])
        self.assertTrue(provenance["within_domain_isolation"]["is_pure"])
        self.assertFalse(provenance["leakage_cross"]["has_leakage"])
        self.assertFalse(provenance["leakage_within"]["has_leakage"])

    def test_create_supervised_lodo_slices_rejects_target_leak_in_source(self) -> None:
        corrupted_source = list(self.source_records) + [
            SQLRecord(full_query="SELECT leaked FROM ohr;", label=1, split="train", domain="d")
        ]
        with self.assertRaises(ValueError) as ctx:
            create_supervised_lodo_slices(
                source_records=corrupted_source,
                target_records=self.target_records,
                target_domain="d",
                train_size=40,
                test_size=20,
                seed=42,
            )
        self.assertIn("Cross-domain evaluation violated", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
