#!/usr/bin/env python3
"""CLI entrypoint — delegates to the AI World orchestration stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.architecture.layers import AIWorldStack
from src.orchestration.pipeline import FraudAIPipeline
from src.realtime.ai_engine import RealtimeFraudAIEngine
from src.serving.runtime import FraudServingRuntime
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fraud Detection AI Platform (top-down architecture)"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--architecture", action="store_true", help="Print AI layer stack")
    parser.add_argument("--serve", metavar="CSV", help="Batch scoring (L2 runtime)")
    parser.add_argument("--api", action="store_true", help="Start API + Sentinel web UI")
    parser.add_argument(
        "--website",
        action="store_true",
        help="Alias for --api (professional web interface)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run real-time AI stream in terminal (CatBoost + drift)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Streamlit real-time fraud dashboard",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Streamlit UI (manual entry, file path, upload)",
    )
    parser.add_argument("--max-batches", type=int, default=50, help="Realtime stream limit")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify trained models and scoring paths",
    )
    parser.add_argument(
        "--model-cards",
        action="store_true",
        help="Generate or regenerate model card documentation",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dashboard-port", type=int, default=8501)
    args = parser.parse_args()

    if args.architecture:
        print(AIWorldStack.describe())
        return

    if args.verify:
        import subprocess

        script = ROOT / "scripts" / "verify_models.py"
        raise SystemExit(subprocess.call([sys.executable, str(script)]))

    if args.model_cards:
        import json

        from src.registry.model_card import generate_model_cards
        from src.registry.model_registry import ModelRegistry

        config = load_config(args.config)
        registry = ModelRegistry(config["paths"]["registry"])
        prod = registry.get_production()
        if not prod:
            print("No production model. Run: python main.py")
            raise SystemExit(1)
        summary_path = ROOT / "artifacts" / "run_summary.json"
        drift_alerts, run_id, gov = [], None, "continue"
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            drift_alerts = s.get("drift_alerts", [])
            run_id = s.get("run_id")
            gov = s.get("governance_action", "continue")
        paths = generate_model_cards(
            config,
            version=prod["version"],
            metrics=prod.get("metrics", {}),
            drift_alerts=drift_alerts,
            governance_action=gov,
            run_id=run_id,
        )
        print("Model cards written:")
        for p in paths:
            print(f"  {p}")
        raise SystemExit(0)

    if args.api or args.website:
        import uvicorn

        uvicorn.run(
            "src.api.app:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return

    if args.ui:
        import subprocess

        ui_app = ROOT / "dashboard" / "app.py"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ui_app),
                "--server.port",
                str(args.dashboard_port),
            ],
            check=False,
        )
        return

    if args.dashboard:
        import subprocess

        dashboard = ROOT / "dashboard" / "realtime_dashboard.py"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard),
                "--server.port",
                str(args.dashboard_port),
            ],
            check=False,
        )
        return

    if args.realtime:
        engine = RealtimeFraudAIEngine(load_config(args.config))
        engine.load()
        print("Real-Time AI Engine loaded (CatBoost + Isolation Forest + KS drift)")
        print(f"Streaming up to {args.max_batches} batches…\n")

        def _print_tick(snap) -> None:
            print(
                f"[batch {snap.batch_id:04d}] processed={snap.total_processed} "
                f"fraud_prob={snap.mean_fraud_probability:.3f} "
                f"flagged={snap.fraud_detected} "
                f"drift_features={len(snap.drifted_features)} "
                f"gov={snap.governance_action}"
            )

        for snap in engine.stream(max_batches=args.max_batches, callback=_print_tick):
            if snap.blocked:
                print("\n*** GOVERNANCE BLOCK — drift threshold exceeded ***")
                break
        print("\nStream complete.", engine.status())
        return

    if args.serve:
        runtime = FraudServingRuntime(load_config(args.config))
        runtime.load()
        out, summary = runtime.score_csv(args.serve)
        print(out[["fraud_probability", "fraud_prediction"]].describe())
        print("Batch summary:", summary.to_dict())
        print("Health:", runtime.health().to_dict())
        return

    pipeline = FraudAIPipeline(load_config(args.config))
    pipeline.run_training(generate_if_missing=not args.no_synthetic)


if __name__ == "__main__":
    main()
