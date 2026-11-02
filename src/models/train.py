"""Model training pipeline for SQL injection detection.

Supports Logistic Regression (with training-fitted StandardScaler), Random Forest, and XGBoost.
Uses canonical benchmark_32 feature extractor from src/features/manual_li.py.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.loader import DataSlice
from src.features.manual_li import extract_feature_vector, FEATURE_NAMES_32

logger = logging.getLogger(__name__)


def extract_features_matrix(queries: Sequence[str]) -> np.ndarray:
    """Extract 32-dimensional feature matrix for a sequence of SQL queries.

    Uses the benchmark_32 manual feature extractor (Qi Li et al., 2019).
    """
    matrix = np.zeros((len(queries), len(FEATURE_NAMES_32)), dtype=np.float64)
    for i, q in enumerate(queries):
        matrix[i] = extract_feature_vector(q, mode="benchmark_32")
    return matrix


class SQLDetectionPipeline:
    """Encapsulates feature extraction, preprocessing scaler, and estimator."""

    def __init__(
        self,
        model_family: str,
        estimator: Any,
        scaler: Optional[StandardScaler] = None,
        feature_names: Optional[list[str]] = None,
        model_id: str = "",
        training_domain: str = "",
        seed: int = 42,
    ):
        self.model_family = model_family
        self.estimator = estimator
        self.scaler = scaler
        self.feature_names = feature_names or FEATURE_NAMES_32
        self.model_id = model_id
        self.training_domain = training_domain
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SQLDetectionPipeline":
        """Fit scaler (if present) and estimator on training data."""
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        self.estimator.fit(X_scaled, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned scaling transform without refitting."""
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels (0 = benign, 1 = attack)."""
        X_trans = self.transform(X)
        return self.estimator.predict(X_trans)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities [P(y=0), P(y=1)]."""
        X_trans = self.transform(X)
        return self.estimator.predict_proba(X_trans)

    def save(self, filepath: Path) -> Path:
        """Serialize pipeline to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved pipeline to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "SQLDetectionPipeline":
        """Load serialized pipeline from disk."""
        return joblib.load(filepath)


def build_estimator(model_family: str, seed: int = 42) -> tuple[Any, Optional[StandardScaler]]:
    """Build fresh estimator instance and associated preprocessing scaler."""
    if model_family == "logistic_regression":
        scaler = StandardScaler()  # Must be fitted ONLY on training data
        estimator = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        )
    elif model_family == "random_forest":
        scaler = None  # Tree-based estimators are invariant to monotonic scale
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_family == "xgboost":
        scaler = None  # Tree-based
        estimator = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    return estimator, scaler


def train_model_pipeline(
    train_slice: DataSlice,
    model_family: str,
    model_id: str,
    seed: int = 42,
    training_domain: str = "",
) -> SQLDetectionPipeline:
    """Train a detection pipeline from raw queries in a DataSlice."""
    logger.info(f"Extracting features for {len(train_slice)} queries in {train_slice.name}...")
    start_feat = time.time()
    X_train = extract_features_matrix(train_slice.queries)
    y_train = np.array(train_slice.labels, dtype=np.int64)
    logger.info(f"Feature extraction complete in {time.time() - start_feat:.2f}s. Shape: {X_train.shape}")

    estimator, scaler = build_estimator(model_family, seed=seed)
    pipeline = SQLDetectionPipeline(
        model_family=model_family,
        estimator=estimator,
        scaler=scaler,
        model_id=model_id,
        training_domain=training_domain,
        seed=seed,
    )

    start_train = time.time()
    logger.info(f"Fitting {model_family} ({model_id}) on {len(y_train)} samples...")
    pipeline.fit(X_train, y_train)
    logger.info(f"Training completed in {time.time() - start_train:.2f}s.")

    return pipeline
