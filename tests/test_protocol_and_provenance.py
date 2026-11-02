"""Unit tests for experimental protocols, split-provenance handling, and evaluation quarantine."""

import unittest
from src.data.loader import SQLRecord
from src.data.protocol import (
    create_provenance_matched_protocol,
    create_sensitivity_unbalanced_protocol,
)


class TestProtocolAndProvenance(unittest.TestCase):
    """Test protocol creation, quarantine assertions, and split-provenance handling."""

    def setUp(self) -> None:
        # Build synthetic source records (Domain B / sakila)
        self.source_records = []
        for i in range(100):
            self.source_records.append(
                SQLRecord(
                    full_query=f"SELECT film_{i} FROM sakila_film;",
                    label=0,
                    split="test",
                    original_split="test",
                    domain="b",
                )
            )
        for i in range(100):
            self.source_records.append(
                SQLRecord(
                    full_query=f"SELECT film_{i} FROM sakila_film WHERE id={i} OR 1=1;",
                    label=1,
                    split="test",
                    original_split="test",
                    domain="b",
                    attack_technique="boolean",
                )
            )

        # Build synthetic target records (Domain D / OHR)
        self.target_records = []
        for i in range(150):
            self.target_records.append(
                SQLRecord(
                    full_query=f"SELECT emp_{i} FROM ohr_employees;",
                    label=0,
                    split="test",
                    original_split="test",
                    domain="d",
                )
            )
        for i in range(150):
            self.target_records.append(
                SQLRecord(
                    full_query=f"SELECT emp_{i} FROM ohr_employees WHERE id={i} UNION SELECT 1;",
                    label=1,
                    split="test",
                    original_split="test",
                    domain="d",
                    attack_technique="union",
                )
            )

    def test_provenance_matched_protocol_isolation_and_disjointness(self) -> None:
        bundle = create_provenance_matched_protocol(
            target_records=self.target_records,
            source_records=self.source_records,
            target_domain="d",
            eval_size=40,
            eval_attack_ratio=0.50,  # 20 benign, 20 attack
            train_size=60,
            train_attack_ratio=0.50, # 30 benign, 30 attack
            seed=42,
        )

        self.assertEqual(len(bundle.eval_slice), 40)
        self.assertEqual(bundle.eval_slice.benign_count, 20)
        self.assertEqual(bundle.eval_slice.attack_count, 20)

        self.assertEqual(len(bundle.within_train), 60)
        self.assertEqual(bundle.within_train.benign_count, 30)
        self.assertEqual(bundle.within_train.attack_count, 30)

        self.assertEqual(len(bundle.cross_train), 60)
        self.assertEqual(bundle.cross_train.benign_count, 30)
        self.assertEqual(bundle.cross_train.attack_count, 30)

        # Disjointness assertion: exactly 0 overlap between within_train and eval_slice
        eval_queries = set(bundle.eval_slice.queries)
        within_queries = set(bundle.within_train.queries)
        self.assertEqual(len(eval_queries.intersection(within_queries)), 0)

        # Repeat-fit training data identity
        self.assertEqual(bundle.cross_train.queries, bundle.repeat_cross_train.queries)

        # Provenance metadata
        self.assertEqual(bundle.protocol_name, "provenance_matched")
        self.assertTrue(bundle.metadata["leakage_audit_within"]["is_isolated"] if "is_isolated" in bundle.metadata["leakage_audit_within"] else not bundle.metadata["leakage_audit_within"]["has_leakage"])

    def test_protocol_rejects_target_domain_leak_in_source(self) -> None:
        leaked_source = list(self.source_records) + [
            SQLRecord(full_query="SELECT leaked FROM ohr_employees;", label=1, split="test", domain="d")
        ]
        with self.assertRaises(ValueError) as ctx:
            create_provenance_matched_protocol(
                target_records=self.target_records,
                source_records=leaked_source,
                target_domain="d",
                eval_size=20,
                train_size=30,
            )
        self.assertIn("Cross-domain evaluation breached", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
