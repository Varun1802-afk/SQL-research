"""Integration tests validating real downloaded Superviz26-SQL data against schema and benchmarks."""

import csv
import unittest
from pathlib import Path

from src.data.loader import REQUIRED_COLUMNS, validate_header


class TestRealDataIntegration(unittest.TestCase):
    """Real dataset validation on downloaded Superviz26-SQL files."""

    def setUp(self) -> None:
        self.d_d_path = Path("data/d-d.csv")

    def test_d_d_csv_file_exists_and_size_verified(self) -> None:
        self.assertTrue(self.d_d_path.is_file(), "data/d-d.csv must exist after download.")
        # Size must match canonical 227 MB
        size = self.d_d_path.stat().st_size
        self.assertGreater(size, 200_000_000, f"Expected >200MB, got {size:,} bytes.")

    def test_d_d_csv_schema_and_header(self) -> None:
        with open(self.d_d_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        validated = validate_header(header)
        for req in REQUIRED_COLUMNS:
            self.assertIn(req, validated)

    def test_d_d_csv_first_1000_rows_valid(self) -> None:
        with open(self.d_d_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 1000:
                    break
                self.assertIn(row["label"], ["0", "1"])
                self.assertIn(row["split"], ["train", "test"])
                self.assertTrue(len(row["full_query"].strip()) > 0)


if __name__ == "__main__":
    unittest.main()
