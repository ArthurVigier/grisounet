"""Generate MM256 day charts from the saved BigQuery predictions table."""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_logic.mm256_day_service import (
    DEFAULT_RISK_THRESHOLD,
    build_mm256_day_context,
    build_mm256_day_payload_from_context,
    save_mm256_day_assets,
)


def visualize_mm256_day(
    date_str: str,
    model_timestamp: str | None = None,
    run_timestamp: str | None = None,
    risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    period_padding_seconds: int = 180,
    upload_graphs: bool = False,
) -> dict:
    context = build_mm256_day_context(
        date_str=date_str,
        run_timestamp=run_timestamp or model_timestamp,
        risk_threshold=risk_threshold,
        period_padding_seconds=period_padding_seconds,
    )
    asset_paths = save_mm256_day_assets(context, upload_to_gcs=upload_graphs)
    payload = build_mm256_day_payload_from_context(context, include_points=False)

    print(f"BigQuery table -> {payload['bigquery_table']}")
    print(f"Whole-day graph -> {asset_paths['overlay_local_path']}")
    print(f"Event graph -> {asset_paths['event_local_path']}")
    if asset_paths["overlay_gcs_uri"]:
        print(f"Whole-day graph GCS -> {asset_paths['overlay_gcs_uri']}")
    if asset_paths["event_gcs_uri"]:
        print(f"Event graph GCS -> {asset_paths['event_gcs_uri']}")

    return {
        "date": date_str,
        "run_timestamp": payload["run_timestamp"],
        "model_timestamp": payload["model_timestamp"],
        "bigquery_table": payload["bigquery_table"],
        "summary": payload["summary"],
        **asset_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize MM256 day charts from BigQuery predictions")
    parser.add_argument("--date", required=True, help="Calendar day to analyze, format YYYY-MM-DD")
    parser.add_argument(
        "--model-timestamp",
        default=None,
        help="Saved model timestamp, for example mm256_20260318_174759",
    )
    parser.add_argument(
        "--run-timestamp",
        default=None,
        help="BigQuery predictions table suffix, for example 20260318_174759",
    )
    parser.add_argument("--risk-threshold", type=float, default=DEFAULT_RISK_THRESHOLD)
    parser.add_argument("--period-padding-seconds", type=int, default=180)
    parser.add_argument("--upload-graphs", action="store_true")
    args = parser.parse_args()

    visualize_mm256_day(
        date_str=args.date,
        model_timestamp=args.model_timestamp,
        run_timestamp=args.run_timestamp,
        risk_threshold=args.risk_threshold,
        period_padding_seconds=args.period_padding_seconds,
        upload_graphs=args.upload_graphs,
    )


if __name__ == "__main__":
    main()
