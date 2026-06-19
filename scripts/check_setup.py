#!/usr/bin/env python3
"""Preflight checks before train / API / UI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import dataset_kind
from src.utils.config import load_config

ISSUES: list[str] = []
WARNINGS: list[str] = []


def need(msg: str) -> None:
    ISSUES.append(msg)


def warn(msg: str) -> None:
    WARNINGS.append(msg)


def main() -> int:
    config = load_config(ROOT / "config.yaml")
    kind = dataset_kind(config)

    print("\n=== Fraud Platform — setup check ===\n")

    if kind == "credit_dt":
        cfg = config["data"]["credit_dt"]
        for label, key in (("Train", "train_path"), ("Test", "test_path")):
            p = ROOT / cfg[key]
            if p.exists():
                print(f"  [OK] {label}: {cfg[key]}")
            else:
                need(f"{label} CSV missing: {cfg[key]} — see data/credit_dt/README.md")
    else:
        p = ROOT / config["data"]["path"]
        if p.exists():
            print(f"  [OK] Dataset: {config['data']['path']}")
        else:
            warn(f"Kaggle CSV missing: {config['data']['path']} (synthetic fallback may run)")

    model = ROOT / config["paths"]["models"] / "catboost_fraud.cbm"
    if model.exists():
        print(f"  [OK] Trained model: {model.relative_to(ROOT)}")
    else:
        need("No trained model — run: python main.py")

    featurizer = ROOT / config["paths"]["artifacts"] / "credit_dt_featurizer.json"
    if kind == "credit_dt":
        if featurizer.exists():
            print(f"  [OK] Featurizer: {featurizer.relative_to(ROOT)}")
        else:
            warn("Featurizer manifest missing — created on first train or score")

    scaler = ROOT / config["paths"]["artifacts"] / "scaler.joblib"
    if scaler.exists():
        print(f"  [OK] Scaler: {scaler.relative_to(ROOT)}")
    elif model.exists():
        warn("Scaler missing but model exists — retrain recommended")

    if WARNINGS:
        print("\nWarnings:")
        for w in WARNINGS:
            print(f"  • {w}")

    if ISSUES:
        print("\nBlockers:")
        for i in ISSUES:
            print(f"  ✗ {i}")
        print("\nFix blockers, then: python main.py && python main.py --verify")
        return 1

    print("\nReady. Suggested next steps:")
    print("  python main.py --verify")
    print("  python main.py --api          # Sentinel UI at http://127.0.0.1:8000/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
