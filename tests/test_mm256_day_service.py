import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ml_logic.mm256_day_service as service


def _build_day_detail() -> pd.DataFrame:
    rows = []
    sample_specs = [
        {
            "sample_id": 1,
            "forecast_origin_time": pd.Timestamp("2024-01-01 11:59:59"),
            "values": [
                ("2024-01-01 12:00:00", 1.10, 1.05),
                ("2024-01-01 12:00:01", 1.40, 1.45),
                ("2024-01-01 12:00:02", 1.60, 1.55),
                ("2024-01-01 12:00:03", 1.70, 1.65),
            ],
        },
        {
            "sample_id": 2,
            "forecast_origin_time": pd.Timestamp("2024-01-01 12:00:00"),
            "values": [
                ("2024-01-01 12:00:01", 1.40, 1.35),
                ("2024-01-01 12:00:02", 1.60, 1.60),
                ("2024-01-01 12:00:03", 1.70, 1.68),
                ("2024-01-01 12:00:04", 1.20, 1.25),
            ],
        },
    ]

    for spec in sample_specs:
        for forecast_step, (target_time, actual, predicted) in enumerate(spec["values"]):
            rows.append(
                {
                    "sample_id": spec["sample_id"],
                    "forecast_step": forecast_step,
                    "actual_scaled": actual,
                    "predicted_scaled": predicted,
                    "residual_scaled": actual - predicted,
                    "forecast_origin_time": spec["forecast_origin_time"],
                    "target_start_time": pd.Timestamp(spec["values"][0][0]),
                    "target_time": pd.Timestamp(target_time),
                    "target_end_time": pd.Timestamp(spec["values"][-1][0]),
                    "target_date": "2024-01-01",
                    "run_timestamp": "20240101_120000",
                    "actual": actual,
                    "predicted": predicted,
                    "residual": actual - predicted,
                    "abs_residual": abs(actual - predicted),
                }
            )
    return pd.DataFrame(rows)


def _mock_day_bundle() -> dict:
    return {
        "sensor": "MM256",
        "date": "2024-01-01",
        "run_timestamp": "20240101_120000",
        "model_timestamp": "mm256_20240101_120000",
        "table_name": "predictions_mm256_final_20240101_120000",
        "table_ref": "project.dataset.predictions_mm256_final_20240101_120000",
        "detail_df": _build_day_detail(),
        "metadata": {
            "values_unscaled": True,
            "model_variant": "advanced",
            "pinball_quantile": 0.8,
            "feature_columns": ["MM256", "AN422"],
        },
        "values_unscaled": True,
    }


def test_build_mm256_day_context_aggregates_rows_and_selects_threshold_window(monkeypatch):
    monkeypatch.setattr(service, "fetch_mm256_day_detail", lambda **kwargs: _mock_day_bundle())

    context = service.build_mm256_day_context(
        date_str="2024-01-01",
        run_timestamp="20240101_120000",
        risk_threshold=1.5,
        period_padding_seconds=0,
    )

    assert len(context["day_agg"]) == 5
    assert context["summary"]["actual_reaches_risk_threshold"] is True
    assert context["summary"]["prediction_reaches_risk_threshold"] is True
    assert context["event_window"]["threshold_start_time"] == pd.Timestamp("2024-01-01 12:00:02")
    assert context["event_window"]["threshold_end_time"] == pd.Timestamp("2024-01-01 12:00:03")
    assert len(context["event_agg"]) == 2


def test_build_mm256_day_payload_exposes_actual_predicted_and_residual(monkeypatch):
    monkeypatch.setattr(service, "fetch_mm256_day_detail", lambda **kwargs: _mock_day_bundle())

    payload = service.build_mm256_day_payload(
        date_str="2024-01-01",
        run_timestamp="20240101_120000",
        risk_threshold=1.5,
        period_padding_seconds=0,
        include_points=True,
    )

    assert payload["values_unscaled"] is True
    assert payload["summary"]["likely_to_reach_threshold"] is True
    assert payload["whole_day"]["points"][0]["actual"] == 1.1
    assert "predicted" in payload["whole_day"]["points"][0]
    assert "residual" in payload["whole_day"]["points"][0]
