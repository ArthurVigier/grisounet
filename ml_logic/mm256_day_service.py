"""BigQuery-backed MM256 day analysis helpers for the API and demo scripts."""

from __future__ import annotations

import io
import json
import math
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

_TMP_CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMP_CACHE_ROOT / "xdg-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery
from google.cloud import storage

from ml_logic.secrets import get_secret

matplotlib.use("Agg")

TARGET_SENSOR = "MM256"
PREDICTIONS_TABLE_PREFIX = "predictions_mm256_final_"
DEFAULT_RISK_THRESHOLD = 1.5
DEFAULT_EVENT_PADDING_SECONDS = 180
DEFAULT_GRAPH_GCS_PREFIX = "graphs/mm256_day"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "results" / "models"
OUTPUT_DIR = PROJECT_ROOT / "results" / "day_analysis"


def normalize_run_timestamp(run_timestamp: str | None) -> str | None:
    """Accept either YYYYMMDD_HHMMSS or mm256_YYYYMMDD_HHMMSS."""
    if run_timestamp is None:
        return None
    return run_timestamp.removeprefix("mm256_")


def model_timestamp_from_run_timestamp(run_timestamp: str) -> str:
    return f"mm256_{normalize_run_timestamp(run_timestamp)}"


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _get_bq_refs() -> tuple[bigquery.Client, str, str]:
    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_OUTPUT_DATASET") or get_secret("BQ_DATASET")
    region = get_secret("BQ_REGION")
    if not project or not dataset:
        raise RuntimeError("Missing BigQuery configuration (GCP_PROJECT / dataset secret)")
    client = bigquery.Client(project=project, location=region)
    return client, project, dataset


def _bucket_name() -> str:
    bucket_name = get_secret("BUCKET_NAME")
    if not bucket_name:
        raise RuntimeError("Missing BUCKET_NAME secret for GCS access")
    return bucket_name


def resolve_predictions_table(run_timestamp: str | None = None) -> dict[str, str]:
    """Resolve one MM256 predictions table and its timestamp suffix."""
    client, project, dataset = _get_bq_refs()
    normalized = normalize_run_timestamp(run_timestamp)

    if normalized is not None:
        table_name = f"{PREDICTIONS_TABLE_PREFIX}{normalized}"
        return {
            "run_timestamp": normalized,
            "table_name": table_name,
            "table_ref": f"{project}.{dataset}.{table_name}",
        }

    tables = sorted(
        (
            table.table_id
            for table in client.list_tables(f"{project}.{dataset}")
            if table.table_id.startswith(PREDICTIONS_TABLE_PREFIX)
        ),
        reverse=True,
    )
    if not tables:
        raise FileNotFoundError(
            f"No BigQuery tables found with prefix {PREDICTIONS_TABLE_PREFIX}"
        )

    table_name = tables[0]
    return {
        "run_timestamp": table_name.removeprefix(PREDICTIONS_TABLE_PREFIX),
        "table_name": table_name,
        "table_ref": f"{project}.{dataset}.{table_name}",
    }


def _download_blob_best_effort(local_path: Path, blob_path: str) -> bool:
    """Download one GCS blob when it is not already cached locally."""
    if local_path.exists():
        return True

    local_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_name = _bucket_name()
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob(blob_path).download_to_filename(str(local_path))
        return True
    except (GoogleAPIError, OSError):
        return False


def _load_metadata(run_timestamp: str) -> dict[str, Any]:
    model_timestamp = model_timestamp_from_run_timestamp(run_timestamp)
    local_path = MODELS_DIR / f"model_{model_timestamp}_metadata.json"
    _download_blob_best_effort(local_path, f"models/model_{model_timestamp}_metadata.json")
    if not local_path.exists():
        return {}
    with open(local_path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_scaler_bundle(run_timestamp: str) -> tuple[dict[str, Any] | None, bool]:
    model_timestamp = model_timestamp_from_run_timestamp(run_timestamp)
    local_path = MODELS_DIR / f"model_{model_timestamp}_scalers.pkl"
    downloaded = _download_blob_best_effort(local_path, f"models/model_{model_timestamp}_scalers.pkl")
    if not downloaded and not local_path.exists():
        return None, False
    with open(local_path, "rb") as stream:
        return pickle.load(stream), True


def _inverse_with_scaler(values: pd.Series, scaler) -> np.ndarray:
    return scaler.inverse_transform(values.to_numpy(dtype=np.float32).reshape(-1, 1)).reshape(-1)


def _apply_saved_scale_context(
    detail_df: pd.DataFrame,
    run_timestamp: str,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    """Replace BQ-scaled values with unscaled values when the MM256 scaler is available."""
    metadata = _load_metadata(run_timestamp)
    scaler_bundle, scalers_available = _load_scaler_bundle(run_timestamp)

    detail = detail_df.copy()
    detail.rename(
        columns={
            "actual": "actual_scaled",
            "predicted": "predicted_scaled",
            "residual": "residual_scaled",
        },
        inplace=True,
    )

    values_unscaled = False
    target_scaler = None
    if scaler_bundle is not None:
        raw_scalers = scaler_bundle.get("raw_feature_scalers", scaler_bundle)
        target_scaler = raw_scalers.get(TARGET_SENSOR)

    if target_scaler is not None:
        detail["actual"] = _inverse_with_scaler(detail["actual_scaled"], target_scaler)
        detail["predicted"] = _inverse_with_scaler(detail["predicted_scaled"], target_scaler)
        detail["residual"] = detail["actual"] - detail["predicted"]
        values_unscaled = True
    else:
        detail["actual"] = detail["actual_scaled"].astype(float)
        detail["predicted"] = detail["predicted_scaled"].astype(float)
        detail["residual"] = detail["residual_scaled"].astype(float)

    detail["target_time"] = pd.to_datetime(detail["target_time"])
    detail["forecast_origin_time"] = pd.to_datetime(detail["forecast_origin_time"])
    detail["target_start_time"] = pd.to_datetime(detail["target_start_time"])
    detail["target_end_time"] = pd.to_datetime(detail["target_end_time"])
    detail["target_date"] = detail["target_time"].dt.strftime("%Y-%m-%d")
    detail["abs_residual"] = detail["residual"].abs()

    metadata["values_unscaled"] = values_unscaled
    metadata["scalers_available"] = bool(scalers_available)
    return detail, metadata, values_unscaled


def fetch_mm256_day_detail(
    date_str: str,
    run_timestamp: str | None = None,
) -> dict[str, Any]:
    """Fetch MM256 predictions for one day directly from BigQuery."""
    table_info = resolve_predictions_table(run_timestamp)
    client, _, _ = _get_bq_refs()

    query = f"""
        SELECT
          sample_id,
          forecast_step,
          CAST(actual AS FLOAT64) AS actual,
          CAST(predicted AS FLOAT64) AS predicted,
          CAST(residual AS FLOAT64) AS residual,
          forecast_origin_time,
          target_start_time,
          target_time,
          target_end_time,
          target_date,
          run_timestamp
        FROM `{table_info["table_ref"]}`
        WHERE sensor = @sensor
          AND target_date = @target_date
        ORDER BY target_time, forecast_step, sample_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("sensor", "STRING", TARGET_SENSOR),
            bigquery.ScalarQueryParameter("target_date", "DATE", pd.Timestamp(date_str).date()),
        ]
    )
    detail_df = client.query(query, job_config=job_config).result().to_dataframe(
        create_bqstorage_client=True
    )
    if detail_df.empty:
        raise ValueError(
            f"No MM256 prediction rows found in {table_info['table_name']} for {date_str}"
        )

    detail_df, metadata, values_unscaled = _apply_saved_scale_context(
        detail_df,
        run_timestamp=table_info["run_timestamp"],
    )
    return {
        "sensor": TARGET_SENSOR,
        "date": date_str,
        "run_timestamp": table_info["run_timestamp"],
        "model_timestamp": model_timestamp_from_run_timestamp(table_info["run_timestamp"]),
        "table_name": table_info["table_name"],
        "table_ref": table_info["table_ref"],
        "detail_df": detail_df,
        "metadata": metadata,
        "values_unscaled": values_unscaled,
    }


def _quantile(q: float):
    return lambda series: float(series.quantile(q))


def aggregate_day_predictions(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate overlapping MM256 forecasts by target timestamp."""
    grouped = (
        detail_df.groupby("target_time")
        .agg(
            actual=("actual", "mean"),
            predicted=("predicted", "mean"),
            residual=("residual", "mean"),
            abs_residual=("residual", lambda series: float(series.abs().mean())),
            predicted_p10=("predicted", _quantile(0.10)),
            predicted_p90=("predicted", _quantile(0.90)),
            residual_p10=("residual", _quantile(0.10)),
            residual_p90=("residual", _quantile(0.90)),
            n_forecasts=("predicted", "size"),
        )
        .reset_index()
    )
    grouped["target_date"] = grouped["target_time"].dt.strftime("%Y-%m-%d")
    return grouped


def _build_threshold_segments(day_agg: pd.DataFrame, value_col: str, risk_threshold: float) -> list[dict[str, Any]]:
    threshold_rows = day_agg.loc[day_agg[value_col] >= risk_threshold, ["target_time", value_col]].copy()
    if threshold_rows.empty:
        return []

    threshold_rows["segment_id"] = threshold_rows["target_time"].diff().ne(
        pd.Timedelta(seconds=1)
    ).cumsum()
    segments = []
    for _, segment in threshold_rows.groupby("segment_id"):
        peak_idx = segment[value_col].idxmax()
        peak_row = day_agg.loc[peak_idx]
        segments.append(
            {
                "start_time": pd.Timestamp(segment["target_time"].min()),
                "end_time": pd.Timestamp(segment["target_time"].max()),
                "peak_time": pd.Timestamp(peak_row["target_time"]),
                "peak_value": float(peak_row[value_col]),
                "n_points": int(len(segment)),
            }
        )
    return segments


def _select_event_window(
    day_agg: pd.DataFrame,
    risk_threshold: float,
    period_padding_seconds: int,
) -> dict[str, Any]:
    day_start = pd.Timestamp(day_agg["target_time"].min())
    day_end = pd.Timestamp(day_agg["target_time"].max())
    padding = pd.Timedelta(seconds=period_padding_seconds)

    for source in ("actual", "predicted"):
        segments = _build_threshold_segments(day_agg, source, risk_threshold)
        if not segments:
            continue
        best_segment = sorted(
            segments,
            key=lambda segment: (-segment["peak_value"], -segment["n_points"], segment["start_time"]),
        )[0]
        return {
            "source": source,
            "threshold_reached": True,
            "threshold_start_time": best_segment["start_time"],
            "threshold_end_time": best_segment["end_time"],
            "window_start_time": max(day_start, best_segment["start_time"] - padding),
            "window_end_time": min(day_end, best_segment["end_time"] + padding),
            "peak_time": best_segment["peak_time"],
            "peak_value": best_segment["peak_value"],
            "n_points_above_threshold": best_segment["n_points"],
        }

    peak_source = "actual" if float(day_agg["actual"].max()) >= float(day_agg["predicted"].max()) else "predicted"
    peak_row = day_agg.loc[day_agg[peak_source].idxmax()]
    peak_time = pd.Timestamp(peak_row["target_time"])
    return {
        "source": peak_source,
        "threshold_reached": False,
        "threshold_start_time": None,
        "threshold_end_time": None,
        "window_start_time": max(day_start, peak_time - padding),
        "window_end_time": min(day_end, peak_time + padding),
        "peak_time": peak_time,
        "peak_value": float(peak_row[peak_source]),
        "n_points_above_threshold": 0,
    }


def _timestamp_to_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _serialize_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    serializable = frame.copy()
    for column in serializable.select_dtypes(include=["datetime64[ns]"]).columns:
        serializable[column] = serializable[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    return json.loads(serializable.to_json(orient="records"))


def _build_day_summary(
    day_agg: pd.DataFrame,
    event_window: dict[str, Any],
    risk_threshold: float,
) -> dict[str, Any]:
    actual_threshold_rows = day_agg.loc[day_agg["actual"] >= risk_threshold, "target_time"]
    predicted_threshold_rows = day_agg.loc[day_agg["predicted"] >= risk_threshold, "target_time"]

    rmse = math.sqrt(float(np.mean(np.square(day_agg["residual"]))))
    peak_actual_row = day_agg.loc[day_agg["actual"].idxmax()]
    peak_predicted_row = day_agg.loc[day_agg["predicted"].idxmax()]
    peak_abs_residual_row = day_agg.loc[day_agg["residual"].abs().idxmax()]

    return {
        "risk_threshold": float(risk_threshold),
        "n_points": int(len(day_agg)),
        "actual_reaches_risk_threshold": bool(not actual_threshold_rows.empty),
        "prediction_reaches_risk_threshold": bool(not predicted_threshold_rows.empty),
        "likely_to_reach_threshold": bool(not predicted_threshold_rows.empty),
        "first_actual_threshold_time": _timestamp_to_string(actual_threshold_rows.iloc[0]) if not actual_threshold_rows.empty else None,
        "first_prediction_threshold_time": _timestamp_to_string(predicted_threshold_rows.iloc[0]) if not predicted_threshold_rows.empty else None,
        "n_seconds_actual_above_threshold": int((day_agg["actual"] >= risk_threshold).sum()),
        "n_seconds_prediction_above_threshold": int((day_agg["predicted"] >= risk_threshold).sum()),
        "peak_actual": float(peak_actual_row["actual"]),
        "peak_actual_time": _timestamp_to_string(peak_actual_row["target_time"]),
        "peak_predicted": float(peak_predicted_row["predicted"]),
        "peak_predicted_time": _timestamp_to_string(peak_predicted_row["target_time"]),
        "mean_residual": round(float(day_agg["residual"].mean()), 6),
        "mean_abs_residual": round(float(day_agg["residual"].abs().mean()), 6),
        "rmse_residual": round(rmse, 6),
        "peak_abs_residual": float(abs(peak_abs_residual_row["residual"])),
        "peak_abs_residual_time": _timestamp_to_string(peak_abs_residual_row["target_time"]),
        "mean_forecast_multiplicity": round(float(day_agg["n_forecasts"].mean()), 2),
        "max_forecast_multiplicity": int(day_agg["n_forecasts"].max()),
        "selected_event_source": event_window["source"],
        "selected_event_threshold_reached": bool(event_window["threshold_reached"]),
        "selected_event_start_time": _timestamp_to_string(event_window["window_start_time"]),
        "selected_event_end_time": _timestamp_to_string(event_window["window_end_time"]),
        "selected_event_peak_time": _timestamp_to_string(event_window["peak_time"]),
        "selected_event_peak_value": float(event_window["peak_value"]),
    }


def build_mm256_day_context(
    date_str: str,
    run_timestamp: str | None = None,
    risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    period_padding_seconds: int = DEFAULT_EVENT_PADDING_SECONDS,
) -> dict[str, Any]:
    """Build the full MM256 day context from BigQuery-saved predictions."""
    day_bundle = fetch_mm256_day_detail(date_str=date_str, run_timestamp=run_timestamp)
    day_detail = day_bundle["detail_df"]
    day_agg = aggregate_day_predictions(day_detail)
    event_window = _select_event_window(
        day_agg=day_agg,
        risk_threshold=risk_threshold,
        period_padding_seconds=period_padding_seconds,
    )
    event_agg = day_agg[
        (day_agg["target_time"] >= event_window["window_start_time"])
        & (day_agg["target_time"] <= event_window["window_end_time"])
    ].copy()
    summary = _build_day_summary(
        day_agg=day_agg,
        event_window=event_window,
        risk_threshold=risk_threshold,
    )

    return {
        **day_bundle,
        "risk_threshold": float(risk_threshold),
        "day_agg": day_agg,
        "event_agg": event_agg,
        "event_window": event_window,
        "summary": summary,
    }


def build_mm256_day_payload_from_context(
    context: dict[str, Any],
    include_points: bool = True,
) -> dict[str, Any]:
    payload = {
        "sensor": context["sensor"],
        "date": context["date"],
        "run_timestamp": context["run_timestamp"],
        "model_timestamp": context["model_timestamp"],
        "bigquery_table": context["table_ref"],
        "values_unscaled": bool(context["values_unscaled"]),
        "model_variant": context["metadata"].get("model_variant"),
        "pinball_quantile": context["metadata"].get("pinball_quantile"),
        "feature_columns": context["metadata"].get("feature_columns", []),
        "thresholds": {
            "demo_risk_threshold": float(context["risk_threshold"]),
        },
        "summary": context["summary"],
        "whole_day": {
            "start_time": _timestamp_to_string(context["day_agg"]["target_time"].min()),
            "end_time": _timestamp_to_string(context["day_agg"]["target_time"].max()),
            "n_points": int(len(context["day_agg"])),
            "n_raw_rows": int(len(context["detail_df"])),
        },
        "event_window": {
            "source": context["event_window"]["source"],
            "threshold_reached": bool(context["event_window"]["threshold_reached"]),
            "threshold_start_time": _timestamp_to_string(context["event_window"]["threshold_start_time"]),
            "threshold_end_time": _timestamp_to_string(context["event_window"]["threshold_end_time"]),
            "window_start_time": _timestamp_to_string(context["event_window"]["window_start_time"]),
            "window_end_time": _timestamp_to_string(context["event_window"]["window_end_time"]),
            "peak_time": _timestamp_to_string(context["event_window"]["peak_time"]),
            "peak_value": float(context["event_window"]["peak_value"]),
            "n_points": int(len(context["event_agg"])),
        },
    }

    if include_points:
        payload["whole_day"]["points"] = _serialize_frame(context["day_agg"])
        payload["event_window"]["points"] = _serialize_frame(context["event_agg"])

    return payload


def build_mm256_day_payload(
    date_str: str,
    run_timestamp: str | None = None,
    risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    period_padding_seconds: int = DEFAULT_EVENT_PADDING_SECONDS,
    include_points: bool = True,
) -> dict[str, Any]:
    context = build_mm256_day_context(
        date_str=date_str,
        run_timestamp=run_timestamp,
        risk_threshold=risk_threshold,
        period_padding_seconds=period_padding_seconds,
    )
    return build_mm256_day_payload_from_context(context, include_points=include_points)


def render_mm256_day_plot_png(context: dict[str, Any]) -> bytes:
    day_agg = context["day_agg"]
    event_window = context["event_window"]
    risk_threshold = float(context["risk_threshold"])

    plt = _get_pyplot()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1.5]},
    )

    top = axes[0]
    top.plot(day_agg["target_time"], day_agg["actual"], color="black", linewidth=2.1, label="Actual")
    top.plot(day_agg["target_time"], day_agg["predicted"], color="#1565c0", linewidth=1.8, label="Predicted")
    top.fill_between(
        day_agg["target_time"],
        day_agg["predicted_p10"],
        day_agg["predicted_p90"],
        color="#1565c0",
        alpha=0.18,
        label="Predicted p10-p90",
    )
    top.axhline(
        risk_threshold,
        color="#c62828",
        linewidth=1.5,
        linestyle=":",
        label=f"Threshold ({risk_threshold:.1f}%)",
    )
    top.axvspan(
        event_window["window_start_time"],
        event_window["window_end_time"],
        color="#c62828",
        alpha=0.08,
        label="Zoomed event period",
    )
    top.set_title(f"MM256 day view — {context['date']}")
    top.set_ylabel("Methane concentration" if context["values_unscaled"] else "Saved prediction scale")
    top.grid(alpha=0.25)
    top.legend(loc="upper right", ncol=3)

    bottom = axes[1]
    bottom.plot(day_agg["target_time"], day_agg["residual"], color="#455a64", linewidth=1.6, label="Residual")
    bottom.fill_between(
        day_agg["target_time"],
        day_agg["residual_p10"],
        day_agg["residual_p90"],
        color="#90a4ae",
        alpha=0.35,
        label="Residual p10-p90",
    )
    bottom.axhline(0.0, color="#37474f", linewidth=1.0, linestyle="--")
    bottom.set_ylabel("Residual")
    bottom.set_xlabel("Time")
    bottom.grid(alpha=0.25)
    bottom.legend(loc="upper right")

    locator = mdates.AutoDateLocator(minticks=8, maxticks=16)
    formatter = mdates.ConciseDateFormatter(locator)
    bottom.xaxis.set_major_locator(locator)
    bottom.xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return buffer.getvalue()


def render_mm256_event_plot_png(context: dict[str, Any]) -> bytes:
    event_agg = context["event_agg"]
    event_window = context["event_window"]
    risk_threshold = float(context["risk_threshold"])

    plt = _get_pyplot()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4]},
    )

    top = axes[0]
    top.plot(event_agg["target_time"], event_agg["actual"], color="black", linewidth=2.1, label="Actual")
    top.plot(event_agg["target_time"], event_agg["predicted"], color="#1565c0", linewidth=1.8, label="Predicted")
    top.axhline(
        risk_threshold,
        color="#c62828",
        linewidth=1.5,
        linestyle=":",
        label=f"Threshold ({risk_threshold:.1f}%)",
    )
    title_type = "threshold period" if event_window["threshold_reached"] else "peak period"
    top.set_title(f"MM256 {title_type} — {context['date']} ({event_window['source']})")
    top.set_ylabel("Methane concentration" if context["values_unscaled"] else "Saved prediction scale")
    top.grid(alpha=0.25)
    top.legend(loc="upper right")

    bottom = axes[1]
    bottom.plot(event_agg["target_time"], event_agg["residual"], color="#455a64", linewidth=1.6, label="Residual")
    bottom.axhline(0.0, color="#37474f", linewidth=1.0, linestyle="--")
    bottom.set_ylabel("Residual")
    bottom.set_xlabel("Time")
    bottom.grid(alpha=0.25)
    bottom.legend(loc="upper right")
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return buffer.getvalue()


def _upload_file_best_effort(local_path: Path, blob_path: str) -> str | None:
    try:
        bucket_name = _bucket_name()
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob(blob_path).upload_from_filename(str(local_path))
        return f"gs://{bucket_name}/{blob_path}"
    except (GoogleAPIError, OSError, RuntimeError):
        return None


def save_mm256_day_assets(
    context: dict[str, Any],
    output_dir: Path = OUTPUT_DIR,
    upload_to_gcs: bool = False,
    graph_gcs_prefix: str = DEFAULT_GRAPH_GCS_PREFIX,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"mm256_day_{context['date']}_{context['run_timestamp']}"

    overlay_local_path = output_dir / f"{base_name}_overlay.png"
    event_local_path = output_dir / f"{base_name}_event.png"
    overlay_local_path.write_bytes(render_mm256_day_plot_png(context))
    event_local_path.write_bytes(render_mm256_event_plot_png(context))

    assets = {
        "overlay_local_path": str(overlay_local_path),
        "event_local_path": str(event_local_path),
        "overlay_gcs_uri": None,
        "event_gcs_uri": None,
    }
    if upload_to_gcs:
        assets["overlay_gcs_uri"] = _upload_file_best_effort(
            overlay_local_path,
            f"{graph_gcs_prefix}/{overlay_local_path.name}",
        )
        assets["event_gcs_uri"] = _upload_file_best_effort(
            event_local_path,
            f"{graph_gcs_prefix}/{event_local_path.name}",
        )
    return assets
