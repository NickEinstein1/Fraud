"""Synthetic data mimicking the Kaggle Credit Card Fraud schema (V1–V28, Time, Amount, Class)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_creditcard_like_data(
    n_samples: int = 50_000,
    fraud_rate: float = 0.00172,
    random_state: int = 42,
) -> pd.DataFrame:
    """Approximate class imbalance (~0.17% fraud) of the Kaggle dataset."""
    rng = np.random.default_rng(random_state)
    n_fraud = max(1, int(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    def _block(n: int, mean_shift: float = 0.0) -> np.ndarray:
        base = rng.standard_normal((n, 28))
        if mean_shift:
            base += mean_shift
        return base

    legit_v = _block(n_legit)
    fraud_v = _block(n_fraud, mean_shift=1.5)
    V = np.vstack([legit_v, fraud_v])
    time = rng.integers(0, 172_792, size=n_samples)
    amount = np.abs(rng.lognormal(mean=3.0, sigma=1.2, size=n_samples))
    amount[-n_fraud:] *= rng.uniform(1.2, 3.0, size=n_fraud)
    labels = np.array([0] * n_legit + [1] * n_fraud)

    cols = {f"V{i}": V[:, i - 1] for i in range(1, 29)}
    df = pd.DataFrame(cols)
    df["Time"] = time
    df["Amount"] = amount
    df["Class"] = labels
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def ensure_dataset(path: str | Path, n_samples: int = 50_000) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_creditcard_like_data(n_samples=n_samples)
    df.to_csv(path, index=False)
    return path
