import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import api.fast as api_module
import ml_logic.mm256_day_service as service


def _build_context() -> dict:
    day_agg = pd.DataFrame(
        [
            {
                "target_time": pd.Timestamp("2024-01-01 12:00:00"),
                "target_date": "2024-01-01",
                "actual": 1.10,
                "predicted": 1.05,
                "residual": 0.05,
                "abs_residual": 0.05,
                "predicted_p10": 1.02,
                "predicted_p90": 1.08,
                "residual_p10": 0.03,
                "residual_p90": 0.07,
                "n_forecasts": 1,
            },
            {
                "target_time": pd.Timestamp("2024-01-01 12:00:01"),
                "target_date": "2024-01-01",
                "actual": 1.60,
                "predicted": 1.55,
                "residual": 0.05,
                "abs_residual": 0.05,
                "predicted_p10": 1.50,
                "predicted_p90": 1.60,
                "residual_p10": 0.03,
                "residual_p90": 0.07,
                "n_forecasts": 2,
            },
        ]
    )

    event_window = {
        "source": "actual",
        "threshold_reached": True,
        "threshold_start_time": pd.Timestamp("2024-01-01 12:00:01"),
        "threshold_end_time": pd.Timestamp("2024-01-01 12:00:01"),
        "window_start_time": pd.Timestamp("2024-01-01 12:00:01"),
        "window_end_time": pd.Timestamp("2024-01-01 12:00:01"),
        "peak_time": pd.Timestamp("2024-01-01 12:00:01"),
        "peak_value": 1.60,
        "n_points_above_threshold": 1,
    }

    return {
        "sensor": "MM256",
        "date": "2024-01-01",
        "run_timestamp": "20240101_120000",
        "model_timestamp": "mm256_20240101_120000",
        "table_ref": "project.dataset.predictions_mm256_final_20240101_120000",
        "values_unscaled": True,
        "metadata": {
            "model_variant": "advanced",
            "pinball_quantile": 0.8,
            "feature_columns": ["MM256", "AN422"],
        },
        "risk_threshold": 1.5,
        "detail_df": day_agg.copy(),
        "day_agg": day_agg,
        "event_agg": day_agg.iloc[1:].copy(),
        "event_window": event_window,
        "summary": {
            "risk_threshold": 1.5,
            "n_points": 2,
            "actual_reaches_risk_threshold": True,
            "prediction_reaches_risk_threshold": True,
            "likely_to_reach_threshold": True,
            "first_actual_threshold_time": "2024-01-01 12:00:01",
            "first_prediction_threshold_time": "2024-01-01 12:00:01",
            "n_seconds_actual_above_threshold": 1,
            "n_seconds_prediction_above_threshold": 1,
            "peak_actual": 1.6,
            "peak_actual_time": "2024-01-01 12:00:01",
            "peak_predicted": 1.55,
            "peak_predicted_time": "2024-01-01 12:00:01",
            "mean_residual": 0.05,
            "mean_abs_residual": 0.05,
            "rmse_residual": 0.05,
            "peak_abs_residual": 0.05,
            "peak_abs_residual_time": "2024-01-01 12:00:00",
            "mean_forecast_multiplicity": 1.5,
            "max_forecast_multiplicity": 2,
            "selected_event_source": "actual",
            "selected_event_threshold_reached": True,
            "selected_event_start_time": "2024-01-01 12:00:01",
            "selected_event_end_time": "2024-01-01 12:00:01",
            "selected_event_peak_time": "2024-01-01 12:00:01",
            "selected_event_peak_value": 1.6,
        },
    }


def test_mm256_day_endpoint_returns_bq_backed_payload(monkeypatch):
    monkeypatch.setattr(service, "build_mm256_day_context", lambda **kwargs: _build_context())
    client = TestClient(api_module.app)

    response = client.get(
        "/mm256/day",
        params={"date": "2024-01-01", "model_timestamp": "20240101_120000", "risk_threshold": 1.5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["bigquery_table"].endswith("predictions_mm256_final_20240101_120000")
    assert payload["summary"]["likely_to_reach_threshold"] is True
    assert payload["plots"]["whole_day_png"].startswith("/mm256/day/plot?")


def test_mm256_day_endpoint_can_return_saved_graph_locations(monkeypatch):
    monkeypatch.setattr(service, "build_mm256_day_context", lambda **kwargs: _build_context())
    monkeypatch.setattr(
        service,
        "save_mm256_day_assets",
        lambda context, upload_to_gcs=False: {
            "overlay_local_path": "/tmp/day_overlay.png",
            "event_local_path": "/tmp/day_event.png",
            "overlay_gcs_uri": "gs://bucket/day_overlay.png" if upload_to_gcs else None,
            "event_gcs_uri": "gs://bucket/day_event.png" if upload_to_gcs else None,
        },
    )
    client = TestClient(api_module.app)

    response = client.get(
        "/mm256/day",
        params={
            "date": "2024-01-01",
            "model_timestamp": "20240101_120000",
            "save_graphs": "true",
            "upload_graphs": "true",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["saved_assets"]["overlay_local_path"] == "/tmp/day_overlay.png"
    assert payload["saved_assets"]["overlay_gcs_uri"] == "gs://bucket/day_overlay.png"


def test_mm256_day_plot_endpoints_render_png(monkeypatch):
    monkeypatch.setattr(service, "build_mm256_day_context", lambda **kwargs: _build_context())
    client = TestClient(api_module.app)

    full_day = client.get("/mm256/day/plot", params={"date": "2024-01-01"})
    event_plot = client.get("/mm256/day/event-plot", params={"date": "2024-01-01"})

    assert full_day.status_code == 200
    assert event_plot.status_code == 200
    assert full_day.headers["content-type"] == "image/png"
    assert event_plot.headers["content-type"] == "image/png"
    assert len(full_day.content) > 0
    assert len(event_plot.content) > 0
