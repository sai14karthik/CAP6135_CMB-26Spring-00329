"""Load NSL-KDD / CICIDS-style CSVs or synthetic data for smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_csv_supervised(
    csv_path: str | Path,
    label_column: str = "label",
    drop_columns: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expects numeric features + a binary or multiclass label column.
    Multiclass is collapsed to binary: benign vs attack (any non-benign label).
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    drop_columns = drop_columns or []
    if label_column not in df.columns:
        raise ValueError(f"Missing label column {label_column!r}")

    y_raw = df[label_column].astype(str).str.lower().str.strip()
    benign_tokens = {"normal", "benign", "0", "normal."}
    y = (~y_raw.isin(benign_tokens)).astype(np.int64)

    feature_df = df.drop(columns=[label_column] + [c for c in drop_columns if c in df.columns])
    # One-hot any remaining object columns
    feature_df = pd.get_dummies(feature_df, drop_first=False)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = feature_df.to_numpy(dtype=np.float32)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def make_synthetic_sequences(
    n_samples: int = 4000,
    seq_len: int = 8,
    n_features: int = 16,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Toy sequences for end-to-end smoke test without external CSVs.
    X_flat: (n_samples, seq_len * n_features) for DNN.
    X_seq: (n_samples, seq_len, n_features) for RNN/hybrid.
    """
    rng = np.random.default_rng(random_state)
    X_seq = rng.normal(size=(n_samples, seq_len, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples, dtype=np.int64)
    # Inject separable pattern: attacks get a positive shift on early timesteps
    mask = y.astype(bool)
    X_seq[mask, :3, :] += 1.25
    X_flat = X_seq.reshape(n_samples, seq_len * n_features)
    return train_test_split(X_flat, X_seq, y, test_size=0.2, random_state=random_state, stratify=y)
