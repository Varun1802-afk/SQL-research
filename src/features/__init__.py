"""
Feature extraction modules for SQL injection detection.
"""

from .manual_li import (
    FEATURE_NAMES_32,
    FEATURE_NAMES_35,
    extract_features,
    extract_feature_vector,
    get_feature_names,
)

__all__ = [
    "FEATURE_NAMES_32",
    "FEATURE_NAMES_35",
    "extract_features",
    "extract_feature_vector",
    "get_feature_names",
]
