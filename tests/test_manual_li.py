"""
tests/test_manual_li.py

Unit tests for handcrafted manual SQL injection feature extractor.
Note: Test queries in this suite are synthetic test cases used solely for
software verification and unit testing of feature extraction logic.
They do not constitute experimental research samples.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to sys.path to allow clean imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.manual_li import (
    FEATURE_NAMES_32,
    FEATURE_NAMES_35,
    DORMANT_FEATURE_NAMES,
    extract_features,
    extract_feature_vector,
    get_feature_names,
    has_tautology,
    get_escape_char_number,
    get_char_kinds_number,
)


class TestManualLiFeatureExtractor(unittest.TestCase):
    """Test suite for manual Li et al. (2019) feature extractor."""

    def test_feature_count_default_mode(self):
        """Default mode (benchmark_32) must return exactly 32 features."""
        query = "SELECT * FROM users WHERE id = 1"
        features = extract_features(query)
        self.assertEqual(len(features), 32)
        self.assertEqual(len(extract_feature_vector(query)), 32)
        self.assertEqual(len(get_feature_names()), 32)

    def test_feature_count_full_35_mode(self):
        """full_35 mode must return exactly 35 features."""
        query = "SELECT * FROM users WHERE id = 1"
        features = extract_features(query, mode="full_35")
        self.assertEqual(len(features), 35)
        self.assertEqual(len(extract_feature_vector(query, mode="full_35")), 35)
        self.assertEqual(len(get_feature_names(mode="full_35")), 35)

    def test_column_order_benchmark_32(self):
        """Column names and dictionary keys must strictly match canonical FEATURE_NAMES_32 order."""
        query = "SELECT * FROM airports WHERE ident = 'KLAX'"
        features = extract_features(query, mode="benchmark_32")
        keys = list(features.keys())
        self.assertEqual(keys, FEATURE_NAMES_32)
        self.assertEqual(get_feature_names("benchmark_32"), FEATURE_NAMES_32)

    def test_column_order_full_35(self):
        """Column names and dictionary keys in full_35 must match FEATURE_NAMES_35 order."""
        query = "SELECT * FROM airports WHERE ident = 'KLAX'"
        features = extract_features(query, mode="full_35")
        keys = list(features.keys())
        self.assertEqual(keys, FEATURE_NAMES_35)
        self.assertEqual(get_feature_names("full_35"), FEATURE_NAMES_35)

    def test_dormant_features_presence(self):
        """Dormant features must be excluded in benchmark_32 and present in full_35."""
        query = "SELECT 'nested \"quotes\"' FROM t"
        feats_32 = extract_features(query, mode="benchmark_32")
        feats_35 = extract_features(query, mode="full_35")

        for dormant in DORMANT_FEATURE_NAMES:
            self.assertNotIn(dormant, feats_32)
            self.assertIn(dormant, feats_35)

    def test_deterministic_output(self):
        """Repeated feature extractions must yield identical outputs."""
        query = "SELECT * FROM users WHERE user='admin' OR 1=1--"
        res1 = extract_features(query)
        res2 = extract_features(query)
        vec1 = extract_feature_vector(query)
        vec2 = extract_feature_vector(query)

        self.assertEqual(res1, res2)
        self.assertEqual(vec1, vec2)

    def test_empty_input(self):
        """Empty string must produce zeroed counts and length 0."""
        features = extract_features("")
        self.assertEqual(features["len_query"], 0)
        self.assertEqual(features["has_query_keywords"], 0)
        self.assertEqual(features["has_union"], 0)
        self.assertEqual(features["has_comment"], 0)
        self.assertEqual(features["c_num"], 0)
        self.assertEqual(features["c_space"], 0)
        self.assertEqual(features["c_comparison"], 0)
        self.assertTrue(all(v == 0 for v in features.values()))

    def test_none_input_handled_gracefully(self):
        """None input should be treated cleanly as empty string without crashing."""
        features = extract_features(None)
        self.assertEqual(features["len_query"], 0)
        self.assertTrue(all(v == 0 for v in features.values()))

    def test_mixed_case_keywords(self):
        """Keyword matching must be case-insensitive, while character counts preserve case."""
        query = "sElEcT uNiOn NuLl"
        features = extract_features(query)

        self.assertEqual(features["has_query_keywords"], 1)
        self.assertEqual(features["has_union"], 1)
        self.assertEqual(features["has_null"], 1)
        # Uppercase count in "sElEcT uNiOn NuLl": E, E, T, N, O, N, L -> exactly 7 uppercase characters
        self.assertEqual(features["c_upper"], 7)

    def test_comments_single_line(self):
        """Single-line comment tokens '--' and '#' must trigger has_comment."""
        q_dash = "SELECT 1 -- test comment"
        q_hash = "SELECT 1 # test comment"
        q_none = "SELECT 1"

        self.assertEqual(extract_features(q_dash)["has_comment"], 1)
        self.assertEqual(extract_features(q_hash)["has_comment"], 1)
        self.assertEqual(extract_features(q_none)["has_comment"], 0)

    def test_comments_multiline(self):
        """Multi-line comment tokens '/*' and '*/' must trigger has_multiline_comment.
        Note: In U_Li.py's single-pass character loop, the lookahead does not advance the
        index, so the '*' in '/*' and the '/' in '*/' fall into the c_arith branch (yielding 2).
        This test verifies exact replication of U_Li.py's counting behavior.
        """
        q_ml = "SELECT /* inline comment */ 1"
        features = extract_features(q_ml)

        self.assertEqual(features["has_multiline_comment"], 1)
        # Verifies U_Li.py verbatim behavior: c_arith is 2 due to unskipped trailing tokens in '/*' and '*/'
        self.assertEqual(features["c_arith"], 2)

    def test_brackets_counting(self):
        """Bracket counts for square [], round (), and curly {} brackets must be accurate."""
        query = "fn({a: [1, 2], b: (3)})"
        features = extract_features(query)

        self.assertEqual(features["c_square_brackets"], 2)  # '[' and ']'
        self.assertEqual(features["c_round_brackets"], 4)   # '(' and ')' twice
        self.assertEqual(features["c_curly_brackets"], 2)   # '{' and '}'

    def test_tautology_detection(self):
        """Tautological patterns (\\w+=\\1) must trigger has_tautology.
        Note: \\w matches word characters ([a-zA-Z0-9_]), so bare word/digit tautologies
        like 1=1, admin=admin, and x=x activate, whereas quoted strings ('a'='a')
        do not match directly adjacent to '=' without word boundary.
        """
        self.assertEqual(has_tautology("1=1"), 1)
        self.assertEqual(has_tautology("admin=admin"), 1)
        self.assertEqual(has_tautology("x=x"), 1)
        self.assertEqual(has_tautology("WHERE 1=1 --"), 1)
        self.assertEqual(has_tautology("'a'='a'"), 0)  # Verbatim U_Li.py behavior: quotes adjacent to '=' prevent \w+ match
        self.assertEqual(has_tautology("1=2"), 0)
        self.assertEqual(has_tautology("SELECT * FROM tbl"), 0)

    def test_comparison_operators(self):
        """Comparison operator count must track '=', '<', '>', '<=', '>=', '<>', '!='."""
        query = "a = 1 AND b < 2 AND c > 3 AND d <= 4"
        features = extract_features(query)
        # In models/U_Li.py:
        # query has:
        # '=' count: 2 (in 'a = 1' and 'd <= 4')
        # '<' count: 2 (in 'b < 2' and 'd <= 4')
        # '>' count: 1 (in 'c > 3')
        # '<=' count: 1 (in 'd <= 4')
        # Total comparison count = 2 + 2 + 1 + 1 = 6
        self.assertEqual(features["c_comparison"], 6)

    def test_escape_characters_dormant(self):
        """Hexadecimal and Unicode escape sequences must be counted in full_35 mode."""
        query = r"SELECT \x41 \x42 \u0043"
        features_35 = extract_features(query, mode="full_35")

        self.assertEqual(features_35["c_hex"], 2)       # \x41, \x42
        self.assertEqual(features_35["c_unicode"], 1)   # \u0043

    def test_quotes_within_quotes_dormant(self):
        """Single quote within double quote (or vice versa) must be counted in full_35."""
        query_s_in_d = 'SELECT "test \'nested\' string"'
        query_d_in_s = "SELECT 'test \"nested\" string'"

        f_s_in_d = extract_features(query_s_in_d, mode="full_35")
        f_d_in_s = extract_features(query_d_in_s, mode="full_35")

        self.assertEqual(f_s_in_d["c_quot_within_quot"], 2)
        self.assertEqual(f_d_in_s["c_quot_within_quot"], 2)

    def test_vector_conversion(self):
        """extract_feature_vector must match extract_features values in exact canonical order."""
        query = "SELECT id, name FROM users WHERE id = 10 AND length(name) > 3"
        for mode in ("benchmark_32", "full_35"):
            d = extract_features(query, mode=mode)
            vec = extract_feature_vector(query, mode=mode)
            names = get_feature_names(mode=mode)
            self.assertEqual(len(vec), len(names))
            self.assertEqual([d[name] for name in names], vec)

    def test_specific_sql_keywords_and_substring_rules(self):
        """Verify individual keyword activations and verbatim substring matching behaviors."""
        # Test word-boundary keywords
        self.assertEqual(extract_features("EXEC sp_help")["has_exec"], 1)
        self.assertEqual(extract_features("execution")["has_exec"], 0)  # \bexec\b

        self.assertEqual(extract_features("SELECT substring(col, 1, 2)")["has_string_functions"], 1)
        self.assertEqual(extract_features("SELECT substr(col, 1, 2)")["has_string_functions"], 1)

        self.assertEqual(extract_features("SELECT floor(rand() * 10)")["has_floor"], 1)
        self.assertEqual(extract_features("SELECT floor(rand() * 10)")["has_rand"], 1)

        self.assertEqual(extract_features("SELECT sleep(5)")["has_sleep"], 1)
        self.assertEqual(extract_features("SELECT count(*)")["has_count"], 1)
        self.assertEqual(extract_features("SELECT ascii('a')")["has_ascii"], 1)
        self.assertEqual(extract_features("SELECT concat('a', 'b')")["has_concat"], 1)
        self.assertEqual(extract_features("SELECT if(1=1, 1, 0)")["has_if"], 1)

        # Verifies substring match in U_Li.py for DATABASE_KEYWORDS
        self.assertEqual(extract_features("SELECT * FROM information_schema.tables")["has_database_keywords"], 1)
        self.assertEqual(extract_features("SELECT * FROM sysobjects")["has_database_keywords"], 1)

        # Verifies substring match in U_Li.py for FILE_KEYWORDS
        self.assertEqual(extract_features("SELECT load_file('/etc/passwd')")["has_file_keywords"], 1)
        self.assertEqual(extract_features("SELECT * INTO outfile '/tmp/out'")["has_file_keywords"], 1)



    def test_representative_benign_queries(self):
        """Verify feature extraction on representative benign domain queries (Sakila & OHR)."""
        # Sakila query
        q_sakila = "SELECT f.title, c.name FROM film f JOIN film_category fc ON f.film_id = fc.film_id JOIN category c ON fc.category_id = c.category_id WHERE f.rental_rate > 2.99;"
        f_sakila = extract_features(q_sakila)
        self.assertEqual(f_sakila["has_query_keywords"], 1)
        self.assertEqual(f_sakila["has_union"], 0)
        self.assertEqual(f_sakila["has_comment"], 0)
        self.assertEqual(f_sakila["has_tautology"], 0)
        self.assertEqual(f_sakila["c_comparison"], 3)  # 2 '=' in JOIN ON clauses + 1 '>' in WHERE clause
        self.assertGreater(f_sakila["len_query"], 50)

        # OHR query
        q_ohr = "SELECT employee_id, first_name, last_name, salary FROM employees WHERE department_id = 80 AND salary >= 10000;"
        f_ohr = extract_features(q_ohr)
        self.assertEqual(f_ohr["has_query_keywords"], 1)
        self.assertEqual(f_ohr["has_union"], 0)
        self.assertEqual(f_ohr["has_comment"], 0)
        # c_comparison: 2 '=' (in department_id = 80 and salary >= 10000) + 1 '>' (in >=) + 1 '>=' = 4
        self.assertEqual(f_ohr["c_comparison"], 4)

    def test_representative_malicious_queries(self):
        """Verify feature extraction on diverse SQL injection attack types."""
        # UNION attack
        q_union = "1' UNION ALL SELECT null, null, username, password FROM users--"
        f_union = extract_features(q_union)
        self.assertEqual(f_union["has_union"], 1)
        self.assertEqual(f_union["has_null"], 1)
        self.assertEqual(f_union["has_comment"], 1)

        # Boolean tautology attack
        q_taut = "admin' OR 1=1#"
        f_taut = extract_features(q_taut)
        self.assertEqual(f_taut["has_tautology"], 1)
        self.assertEqual(f_taut["has_connection_keywords"], 1)
        self.assertEqual(f_taut["has_comment"], 1)

        # Time-based blind attack
        q_time = "SELECT * FROM products WHERE id=1 AND sleep(5)--"
        f_time = extract_features(q_time)
        self.assertEqual(f_time["has_sleep"], 1)
        self.assertEqual(f_time["has_connection_keywords"], 1)

    def test_whitespace_and_casing_variations(self):
        """Verify robust behavior across irregular whitespace, tabs, and casing."""
        q_ws = "SELECT\t\t*\nFROM\r\nusers  WHERE   id   =\t1\nUNION\t\tSELECT\tnull"
        f_ws = extract_features(q_ws)
        self.assertEqual(f_ws["has_query_keywords"], 1)
        self.assertEqual(f_ws["has_union"], 1)
        self.assertEqual(f_ws["has_null"], 1)
        self.assertGreater(f_ws["c_space"], 5)

        q_case = "sElEcT * FrOm TaBlE wHeRe CoL = 'val' uNiOn SeLeCt 1"
        f_case = extract_features(q_case)
        self.assertEqual(f_case["has_query_keywords"], 1)
        self.assertEqual(f_case["has_union"], 1)

    def test_boundary_conditions(self):
        """Verify boundary cases: purely punctuation, non-ASCII Unicode, very long strings."""
        # Non-ASCII Unicode characters
        q_unicode = "SELECT * FROM users WHERE name = 'José' OR 'ünicöde' = 'ünicöde'"
        f_unicode = extract_features(q_unicode)
        self.assertEqual(f_unicode["has_query_keywords"], 1)
        self.assertGreater(f_unicode["len_query"], 0)
        self.assertTrue(all(np.isfinite(v) for v in extract_feature_vector(q_unicode)))

        # Pure punctuation string
        q_punct = ";;;===<<<>>>{}()[]"
        f_punct = extract_features(q_punct)
        self.assertEqual(f_punct["has_query_keywords"], 0)
        self.assertEqual(f_punct["c_curly_brackets"], 2)
        self.assertEqual(f_punct["c_round_brackets"], 2)
        self.assertEqual(f_punct["c_square_brackets"], 2)

    def test_invalid_mode_raises(self):
        """Invalid mode argument must raise ValueError."""
        with self.assertRaises(ValueError):
            extract_features("SELECT 1", mode="invalid_mode")
        with self.assertRaises(ValueError):
            extract_feature_vector("SELECT 1", mode="unknown")
        with self.assertRaises(ValueError):
            get_feature_names(mode="bad")


if __name__ == "__main__":
    unittest.main()

