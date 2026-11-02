"""
src/features/manual_li.py

Handcrafted lexical, syntactic, and structural SQL injection feature extractor
ported directly from the official benchmark repository:
    Repository: gquetel/sqlia-dataset-generator (formerly gquetel/sqlia-dataset)
    Source File: models/U_Li.py
    Attribution: Qi Li, Weishi Li, Junfeng Wang, Mingyu Cheng (IEEE Access 2019)
                 "A SQL Injection Detection Method Based on Adaptive Deep Forest"
                 DOI: 10.1109/ACCESS.2019.2944951

This port strictly preserves the original feature names, case handling, regular
expressions, string search behaviors, and counting semantics without altering
original quirks or heuristics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# Canonical feature names and ordering for the active benchmark representation (32 features)
# Matches the dictionary key insertion order of `result | get_char_kinds_number(query)` in models/U_Li.py
FEATURE_NAMES_32: List[str] = [
    # 23 features from initial `result` dict (U_Li.py lines 42-65)
    "len_query",
    "has_null",
    "has_comment",
    "has_query_keywords",
    "has_union",
    "has_database_keywords",
    "has_connection_keywords",
    "has_file_keywords",
    "has_exec",
    "has_string_functions",
    "c_comparison",
    "has_exist_keyword",
    "has_floor",
    "has_rand",
    "has_group",
    "has_order",
    "has_length",
    "has_ascii",
    "has_concat",
    "has_if",
    "has_count",
    "has_sleep",
    "has_tautology",
    # 9 features from `get_char_kinds_number` dict (U_Li.py lines 196-204)
    "c_num",
    "c_upper",
    "c_space",
    "c_special",
    "c_arith",
    "c_square_brackets",
    "c_round_brackets",
    "has_multiline_comment",
    "c_curly_brackets",
]

# 3 dormant features present in U_Li.py code but omitted from active benchmark output
DORMANT_FEATURE_NAMES: List[str] = [
    "c_quot_within_quot",  # Calculated in get_char_kinds_number loop, omitted from return dict
    "c_hex",               # Defined in get_escape_char_number, never called in extract_li_features
    "c_unicode",           # Defined in get_escape_char_number, never called in extract_li_features
]

# Canonical feature names and ordering for full 35-feature specification
FEATURE_NAMES_35: List[str] = FEATURE_NAMES_32 + DORMANT_FEATURE_NAMES

# Sets and constants defined verbatim from models/U_Li.py lines 29-39
_QUERY_KEYWORDS = {"select", "update", "drop", "insert", "create"}
_DATABASE_KEYWORDS = {
    "sysobjects",
    "msysobjects",
    "version",
    "information_schema",
}
_CONNECTION_KEYWORDS = {"inner join", "and", "or", "xor"}
_FILE_KEYWORDS = {"load_file", "infile", "outfile", "dumpfile"}
_STRING_FUNCTIONS = {"substr", "substring", "mid", "asc"}
_COMPARISON_OPERATORS = ("=", "<", ">", "<=", ">=", "<>", "!=")

# Compiled regular expressions for performance and exact behavioral fidelity
_P_HEX = re.compile(r"\\x[0-9a-fA-F]{2}")
_P_UNICODE = re.compile(r"\\u[0-9a-fA-F]{4}")
_P_TAUTOLOGY = re.compile(r"(\w+)=\1")
_P_NULL = re.compile(r"\bnull\b")
_P_UNION = re.compile(r"\bunion\b")
_P_EXEC = re.compile(r"\bexec\b")
_P_EXISTS = re.compile(r"\bexists\b")
_P_FLOOR = re.compile(r"\bfloor\b")
_P_RAND = re.compile(r"\brand\b")
_P_GROUP = re.compile(r"\bgroup\b")
_P_ORDER = re.compile(r"\border\b")
_P_LENGTH = re.compile(r"\blength\b")
_P_ASCII = re.compile(r"\bascii\b")
_P_CONCAT = re.compile(r"\bconcat\b")
_P_IF = re.compile(r"\bif\b")
_P_COUNT = re.compile(r"\bcount\b")
_P_SLEEP = re.compile(r"\bsleep\b")


def has_tautology(query: str) -> int:
    """Verbatim port of models/U_Li.py lines 125-129.

    Checks if query contains an identity comparison matching r"(\\w+)=\\1".
    Note: Evaluated against raw query string.
    """
    if _P_TAUTOLOGY.search(query) is None:
        return 0
    return 1


def get_escape_char_number(query: str) -> Dict[str, int]:
    """Verbatim port of models/U_Li.py lines 106-122.

    Counts hexadecimal and Unicode escape sequences in query string.
    """
    c_hex = len(_P_HEX.findall(query))
    c_unicode = len(_P_UNICODE.findall(query))
    return {"c_hex": c_hex, "c_unicode": c_unicode}


def get_char_kinds_number(query: str, include_dormant: bool = False) -> Dict[str, int]:
    """Verbatim port of models/U_Li.py lines 132-205.

    Analyzes character types and bracket counts via a single pass over query string.
    If include_dormant is True, includes 'c_quot_within_quot'.
    """
    c_num = 0
    c_upper = 0
    c_space = 0
    c_special = 0
    c_arith = 0
    c_square_brackets = 0
    c_round_brackets = 0
    c_curly_brackets = 0
    c_quot_within_quot = 0
    is_within_squot = False
    is_within_dquot = False
    has_multiline_comment = 0

    query_len = len(query)
    for i, c in enumerate(query):
        if c.isdigit():
            c_num += 1
        elif c.isupper():
            c_upper += 1
        elif c.isspace():
            c_space += 1
        elif c == "/":
            if i + 1 < query_len and query[i + 1] == "*":
                has_multiline_comment = 1
            else:
                c_arith += 1
        elif c == "*":
            if i + 1 < query_len and query[i + 1] == "/":
                has_multiline_comment = 1
            else:
                c_arith += 1
        elif c in ("+", "-"):
            c_arith += 1
        elif c == "[" or c == "]":
            c_square_brackets += 1
        elif c == "(" or c == ")":
            c_round_brackets += 1
        elif c == "{" or c == "}":
            c_curly_brackets += 1
        elif c == "'":
            if is_within_dquot:
                c_quot_within_quot += 1
            else:
                is_within_squot = True
            c_special += 1
        elif c == '"':
            if is_within_squot:
                c_quot_within_quot += 1
            else:
                is_within_dquot = True
            c_special += 1
        elif not c.isalnum():
            c_special += 1

    out = {
        "c_num": c_num,
        "c_upper": c_upper,
        "c_space": c_space,
        "c_special": c_special,
        "c_arith": c_arith,
        "c_square_brackets": c_square_brackets,
        "c_round_brackets": c_round_brackets,
        "has_multiline_comment": has_multiline_comment,
        "c_curly_brackets": c_curly_brackets,
    }
    if include_dormant:
        out["c_quot_within_quot"] = c_quot_within_quot
    return out


def extract_features(query: str, mode: str = "benchmark_32") -> Dict[str, int]:
    """Extract handcrafted SQL injection features from a SQL query.

    Preserves exact logic from models/U_Li.py (extract_li_features).

    Args:
        query: Raw SQL query string. If None, treated as empty string.
        mode: Feature representation mode:
              - 'benchmark_32' (default): 32 features actively emitted by U_Li.py.
              - 'full_35': 35 features, including the 3 dormant features authored
                in U_Li.py (c_quot_within_quot, c_hex, c_unicode).

    Returns:
        Dict mapping canonical feature names to integer counts/flags.

    Raises:
        ValueError: If mode is not 'benchmark_32' or 'full_35'.
    """
    if mode not in ("benchmark_32", "full_35"):
        raise ValueError(
            f"Unsupported feature mode '{mode}'. Choose 'benchmark_32' or 'full_35'."
        )

    if query is None:
        query = ""
    elif not isinstance(query, str):
        query = str(query)

    query_lower = query.lower()

    result: Dict[str, int] = {
        "len_query": len(query),
        "has_null": 0,
        "has_comment": int("--" in query_lower or "#" in query_lower),
        "has_query_keywords": 0,
        "has_union": 0,
        "has_database_keywords": 0,
        "has_connection_keywords": 0,
        "has_file_keywords": 0,
        "has_exec": int(_P_EXEC.search(query_lower) is not None),
        "has_string_functions": 0,
        "c_comparison": 0,
        "has_exist_keyword": int(_P_EXISTS.search(query_lower) is not None),
        "has_floor": int(_P_FLOOR.search(query_lower) is not None),
        "has_rand": int(_P_RAND.search(query_lower) is not None),
        "has_group": int(_P_GROUP.search(query_lower) is not None),
        "has_order": int(_P_ORDER.search(query_lower) is not None),
        "has_length": int(_P_LENGTH.search(query_lower) is not None),
        "has_ascii": int(_P_ASCII.search(query_lower) is not None),
        "has_concat": int(_P_CONCAT.search(query_lower) is not None),
        "has_if": int(_P_IF.search(query_lower) is not None),
        "has_count": int(_P_COUNT.search(query_lower) is not None),
        "has_sleep": int(_P_SLEEP.search(query_lower) is not None),
        "has_tautology": has_tautology(query),
    }

    if _P_NULL.search(query_lower):
        result["has_null"] = 1

    word_pattern = r"\b{}\b"
    for keyword in _QUERY_KEYWORDS:
        if re.search(word_pattern.format(keyword), query_lower):
            result["has_query_keywords"] = 1
            break

    if _P_UNION.search(query_lower):
        result["has_union"] = 1

    for keyword in _CONNECTION_KEYWORDS:
        if keyword in query_lower:
            result["has_connection_keywords"] = 1
            break

    for keyword in _FILE_KEYWORDS:
        if keyword in query_lower:
            result["has_file_keywords"] = 1
            break

    for keyword in _DATABASE_KEYWORDS:
        if keyword in query_lower:
            result["has_database_keywords"] = 1
            break

    for func in _STRING_FUNCTIONS:
        if re.search(word_pattern.format(func), query_lower):
            result["has_string_functions"] = 1
            break

    for op in _COMPARISON_OPERATORS:
        result["c_comparison"] += query_lower.count(op)

    char_kinds = get_char_kinds_number(query, include_dormant=(mode == "full_35"))
    features = result | char_kinds

    if mode == "full_35":
        escapes = get_escape_char_number(query)
        features |= escapes

    return features


def extract_feature_vector(query: str, mode: str = "benchmark_32") -> List[int]:
    """Extract features and return as an ordered list of integer values.

    Args:
        query: Raw SQL query string.
        mode: 'benchmark_32' (default) or 'full_35'.

    Returns:
        List of integers in canonical feature order.
    """
    d = extract_features(query, mode=mode)
    feature_names = FEATURE_NAMES_32 if mode == "benchmark_32" else FEATURE_NAMES_35
    return [d[name] for name in feature_names]


def get_feature_names(mode: str = "benchmark_32") -> List[str]:
    """Return canonical ordered list of feature names.

    Args:
        mode: 'benchmark_32' (default) or 'full_35'.
    """
    if mode == "benchmark_32":
        return list(FEATURE_NAMES_32)
    elif mode == "full_35":
        return list(FEATURE_NAMES_35)
    raise ValueError(
        f"Unsupported feature mode '{mode}'. Choose 'benchmark_32' or 'full_35'."
    )
