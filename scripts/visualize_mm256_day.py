"""Visualize MM256 actuals vs baseline vs model prediction for one or more days.

This script loads the saved final MM256 model artifacts, rebuilds the holdout
test windows for the same preprocessing configuration, and generates:
  1. A day-long overlay of actual vs baseline vs model forecast.
  2. A representative single-window forecast around a selected event time.
  3. CSV exports for the detailed and aggregated day-level data.

The day-long overlay aggregates all overlapping forecasts targeting the same
timestamp by taking their mean, and also shows 10th/90th percentile bands to
surface forecast dispersion across overlapping windows.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.cv_time_series import build_last_input_baseline, compute_mm256_metrics
from scripts.preprocessor_MM256 import (
    TARGET_SENSOR,
    build_window_index_mm256,
    preprocess_mm256,
    slice_windows_mm256,
)


MODELS_DIR = Path(PROJECT_ROOT) / "results" / "models"
OUTPUT_DIR = Path(PROJECT_ROOT) / "results" / "day_analysis"


def _latest_metadata_path() -> Path:
    matches = sorted(MODELS_DIR.glob("model_mm256_*_metadata.json"))
    if not matches:
        raise FileNotFoundError("No MM256 metadata file found in results/models")
    return matches[-1]


def _load_artifacts(model_timestamp: str | None) -> tuple[str, dict, dict, str]:
    if model_timestamp is None:
        metadata_path = _latest_metadata_path()
        model_timestamp = metadata_path.name.removeprefix("model_").removesuffix("_metadata.json")
    else:
        metadata_path = MODELS_DIR / f"model_{model_timestamp}_metadata.json"

    model_path = MODELS_DIR / f"model_{model_timestamp}.keras"
    scalers_path = MODELS_DIR / f"model_{model_timestamp}_scalers.pkl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Missing scalers file: {scalers_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    with open(metadata_path, "r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    with open(scalers_path, "rb") as stream:
        scalers = pickle.load(stream)

    return model_timestamp, metadata, scalers, str(model_path)


def _split_temporal_holdout(data: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(data) * train_ratio)
    if split_idx <= 0 or split_idx >= len(data):
        raise ValueError("train_ratio produces an empty train or test split")
    return data.iloc[:split_idx].copy(), data.iloc[split_idx:].copy()


def _apply_saved_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    out = df.copy()
    for col, scaler in scalers.items():
        if col in out.columns:
            out[col] = scaler.transform(out[[col]]).astype(np.float32)
    return out


def _inverse_target(values: np.ndarray, scaler) -> np.ndarray:
    flat = values.reshape(-1, 1)
    restored = scaler.inverse_transform(flat).reshape(values.shape)
    return restored.astype(np.float32)


def _build_detailed_prediction_df(
    window_index: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
    target_scaler,
) -> pd.DataFrame:
    if len(window_index) != y_true.shape[0]:
        raise ValueError("Window index length does not match prediction sample count")

    horizon = y_true.shape[1]
    sample_count = y_true.shape[0]
    forecast_steps = np.tile(np.arange(horizon), sample_count)
    sample_ids = np.repeat(window_index["sample_id"].to_numpy(), horizon)
    input_end_times = np.repeat(pd.to_datetime(window_index["input_end_time"]).to_numpy(), horizon)
    target_start_times = np.repeat(pd.to_datetime(window_index["target_start_time"]).to_numpy(), horizon)
    target_times = pd.to_datetime(target_start_times) + pd.to_timedelta(forecast_steps, unit="s")

    actual_scaled = y_true.reshape(-1)
    predicted_scaled = y_pred.reshape(-1)
    baseline_scaled = y_baseline.reshape(-1)

    actual = _inverse_target(y_true, target_scaler).reshape(-1)
    predicted = _inverse_target(y_pred, target_scaler).reshape(-1)
    baseline = _inverse_target(y_baseline, target_scaler).reshape(-1)

    detail_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "input_end_time": pd.to_datetime(input_end_times),
            "forecast_step": forecast_steps,
            "target_time": pd.to_datetime(target_times),
            "actual_scaled": actual_scaled.astype(float),
            "predicted_scaled": predicted_scaled.astype(float),
            "baseline_scaled": baseline_scaled.astype(float),
            "actual": actual.astype(float),
            "predicted": predicted.astype(float),
            "baseline": baseline.astype(float),
        }
    )
    detail_df["target_date"] = detail_df["target_time"].dt.strftime("%Y-%m-%d")
    detail_df["abs_error_model"] = (detail_df["actual"] - detail_df["predicted"]).abs()
    detail_df["abs_error_baseline"] = (detail_df["actual"] - detail_df["baseline"]).abs()
    return detail_df


def load_mm256_prediction_bundle(
    model_timestamp: str | None = None,
    source: str = "local",
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
    train_ratio: float = 0.7,
) -> dict:
    """Load model artifacts, rebuild holdout predictions, and return detailed rows."""
    model_timestamp, metadata, scalers, model_path = _load_artifacts(model_timestamp)
    data, _, _ = preprocess_mm256(
        source=source,
        alert_rate=alert_rate,
        concentration_threshold=concentration_threshold,
        scale=False,
    )
    _, test_df = _split_temporal_holdout(data, train_ratio=train_ratio)
    scaled_test = _apply_saved_scalers(test_df, scalers)

    window_length = int(metadata["window_length"])
    forecast_horizon = int(metadata["forecast_horizon"])
    X_test, y_test = slice_windows_mm256(
        scaled_test,
        0,
        len(scaled_test),
        window_length,
        forecast_horizon,
    )
    window_index = build_window_index_mm256(
        scaled_test,
        0,
        len(scaled_test),
        window_length,
        forecast_horizon,
    )
    if X_test.shape[0] == 0:
        raise ValueError("No test windows available for visualization")

    feature_cols = [col for col in scaled_test.columns if col != "ALERT"]
    target_feature_idx = feature_cols.index(TARGET_SENSOR)

    model = load_model(model_path, compile=False)
    y_pred = model.predict(X_test, verbose=0)
    y_baseline = build_last_input_baseline(X_test, target_feature_idx, y_test.shape[1])
    target_scaler = scalers[TARGET_SENSOR]
    detail_df = _build_detailed_prediction_df(window_index, y_test, y_pred, y_baseline, target_scaler)

    return {
        "model_timestamp": model_timestamp,
        "metadata": metadata,
        "detail_df": detail_df,
        "forecast_horizon": forecast_horizon,
        "test_period": f"{test_df.index.min()} -> {test_df.index.max()}",
    }


def _quantile(q: float):
    return lambda series: float(series.quantile(q))


def _aggregate_day(detail_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        detail_df.groupby("target_time")
        .agg(
            actual=("actual", "mean"),
            predicted_mean=("predicted", "mean"),
            predicted_p10=("predicted", _quantile(0.10)),
            predicted_p90=("predicted", _quantile(0.90)),
            baseline_mean=("baseline", "mean"),
            baseline_p10=("baseline", _quantile(0.10)),
            baseline_p90=("baseline", _quantile(0.90)),
            n_forecasts=("predicted", "size"),
        )
        .reset_index()
    )
    grouped["abs_error_model"] = (grouped["actual"] - grouped["predicted_mean"]).abs()
    grouped["abs_error_baseline"] = (grouped["actual"] - grouped["baseline_mean"]).abs()
    return grouped


def _build_sample_index(day_detail: pd.DataFrame) -> pd.DataFrame:
    """Collapse detailed rows to one row per forecast sample."""
    sample_index = (
        day_detail.groupby("sample_id")
        .agg(
            input_end_time=("input_end_time", "first"),
            target_start_time=("target_time", "min"),
        )
        .reset_index()
    )
    return sample_index


def _pick_representative_sample(
    window_index: pd.DataFrame,
    aggregated_day: pd.DataFrame,
    anchor_time: str | None,
    forecast_horizon: int,
) -> int:
    if anchor_time is not None:
        anchor_ts = pd.Timestamp(anchor_time)
        distances = (pd.to_datetime(window_index["input_end_time"]) - anchor_ts).abs()
        return int(window_index.loc[distances.idxmin(), "sample_id"])

    peak_time = aggregated_day.loc[aggregated_day["actual"].idxmax(), "target_time"]
    target_mid = pd.to_datetime(window_index["target_start_time"]) + pd.to_timedelta(forecast_horizon // 2, unit="s")
    distances = (target_mid - peak_time).abs()
    return int(window_index.loc[distances.idxmin(), "sample_id"])


def _plot_day_overlay(day_agg: pd.DataFrame, date_str: str, output_path: Path):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    ax = axes[0]
    ax.plot(day_agg["target_time"], day_agg["actual"], color="black", linewidth=2.2, label="Actual")
    ax.plot(day_agg["target_time"], day_agg["predicted_mean"], color="#1565c0", linewidth=1.8, label="Model")
    ax.fill_between(
        day_agg["target_time"],
        day_agg["predicted_p10"],
        day_agg["predicted_p90"],
        color="#1565c0",
        alpha=0.18,
        label="Model p10-p90",
    )
    ax.plot(
        day_agg["target_time"],
        day_agg["baseline_mean"],
        color="#ef6c00",
        linewidth=1.8,
        linestyle="--",
        label="Baseline",
    )
    ax.fill_between(
        day_agg["target_time"],
        day_agg["baseline_p10"],
        day_agg["baseline_p90"],
        color="#ef6c00",
        alpha=0.16,
        label="Baseline p10-p90",
    )
    ax.set_title(f"MM256 day view — {date_str}")
    ax.set_ylabel("MM256 concentration")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=3)

    count_ax = axes[1]
    count_ax.plot(day_agg["target_time"], day_agg["n_forecasts"], color="#455a64", linewidth=1.5)
    count_ax.fill_between(day_agg["target_time"], 0, day_agg["n_forecasts"], color="#90a4ae", alpha=0.4)
    count_ax.set_ylabel("# forecasts")
    count_ax.set_xlabel("Time")
    count_ax.grid(alpha=0.25)

    locator = mdates.AutoDateLocator(minticks=8, maxticks=16)
    formatter = mdates.ConciseDateFormatter(locator)
    count_ax.xaxis.set_major_locator(locator)
    count_ax.xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_single_window(day_detail: pd.DataFrame, sample_id: int, output_path: Path):
    sample_df = day_detail[day_detail["sample_id"] == sample_id].sort_values("target_time")
    if sample_df.empty:
        raise ValueError(f"No rows found for sample_id={sample_id}")

    input_end_time = sample_df["input_end_time"].iloc[0]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample_df["target_time"], sample_df["actual"], color="black", linewidth=2.2, label="Actual")
    ax.plot(sample_df["target_time"], sample_df["predicted"], color="#1565c0", linewidth=1.8, label="Model")
    ax.plot(
        sample_df["target_time"],
        sample_df["baseline"],
        color="#ef6c00",
        linewidth=1.8,
        linestyle="--",
        label="Baseline",
    )
    ax.set_title(f"MM256 representative forecast — input ends at {input_end_time}")
    ax.set_xlabel("Forecast target time")
    ax.set_ylabel("MM256 concentration")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _summarize_day(date_str: str, day_detail: pd.DataFrame, day_agg: pd.DataFrame, representative_sample_id: int) -> dict:
    """Compute compact day-level summary metrics."""
    model_metrics = compute_mm256_metrics(
        day_agg["actual"].to_numpy().reshape(-1, 1, 1),
        day_agg["predicted_mean"].to_numpy().reshape(-1, 1, 1),
    )
    baseline_metrics = compute_mm256_metrics(
        day_agg["actual"].to_numpy().reshape(-1, 1, 1),
        day_agg["baseline_mean"].to_numpy().reshape(-1, 1, 1),
    )
    peak_row = day_agg.loc[day_agg["actual"].idxmax()]
    return {
        "date": date_str,
        "n_unique_target_times": int(len(day_agg)),
        "n_unique_samples": int(day_detail["sample_id"].nunique()),
        "peak_actual": float(peak_row["actual"]),
        "peak_time": str(peak_row["target_time"]),
        "model_MAE": model_metrics["MAE"],
        "baseline_MAE": baseline_metrics["MAE"],
        "improvement_vs_baseline_MAE": round(baseline_metrics["MAE"] - model_metrics["MAE"], 6),
        "model_RMSE": model_metrics["RMSE"],
        "baseline_RMSE": baseline_metrics["RMSE"],
        "improvement_vs_baseline_RMSE": round(baseline_metrics["RMSE"] - model_metrics["RMSE"], 6),
        "mean_forecast_multiplicity": round(float(day_agg["n_forecasts"].mean()), 2),
        "max_forecast_multiplicity": int(day_agg["n_forecasts"].max()),
        "representative_sample_id": int(representative_sample_id),
    }


def _write_day_report(report_path: Path, model_timestamp: str, test_period: str, summaries: list[dict], outputs: list[dict]):
    """Persist a compact markdown report for one batch of day analyses."""
    lines = [
        f"# MM256 Day Analysis Report ({model_timestamp})",
        "",
        f"- Test period: `{test_period}`",
        "",
    ]
    for summary, output in zip(summaries, outputs):
        lines.extend(
            [
                f"## {summary['date']}",
                "",
                f"- Peak actual: `{summary['peak_actual']}` at `{summary['peak_time']}`",
                f"- MAE: model `{summary['model_MAE']}` vs baseline `{summary['baseline_MAE']}`",
                f"- RMSE: model `{summary['model_RMSE']}` vs baseline `{summary['baseline_RMSE']}`",
                f"- Mean forecast multiplicity: `{summary['mean_forecast_multiplicity']}`",
                f"- Representative sample id: `{summary['representative_sample_id']}`",
                f"- Overlay: `{output['overlay_path']}`",
                f"- Representative window: `{output['window_path']}`",
                f"- Detail CSV: `{output['detail_path']}`",
                f"- Aggregate CSV: `{output['aggregate_path']}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def visualize_mm256_day(
    date_str: str,
    prediction_bundle: dict | None = None,
    model_timestamp: str | None = None,
    source: str = "local",
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
    train_ratio: float = 0.7,
    anchor_time: str | None = None,
) -> dict:
    if prediction_bundle is None:
        prediction_bundle = load_mm256_prediction_bundle(
            model_timestamp=model_timestamp,
            source=source,
            alert_rate=alert_rate,
            concentration_threshold=concentration_threshold,
            train_ratio=train_ratio,
        )

    model_timestamp = prediction_bundle["model_timestamp"]
    forecast_horizon = int(prediction_bundle["forecast_horizon"])
    detail_df = prediction_bundle["detail_df"]
    day_detail = detail_df[detail_df["target_date"] == date_str].copy()
    if day_detail.empty:
        raise ValueError(f"No forecast targets found for date {date_str}")

    day_agg = _aggregate_day(day_detail)
    sample_index = _build_sample_index(day_detail)
    representative_sample_id = _pick_representative_sample(
        sample_index,
        day_agg,
        anchor_time=anchor_time,
        forecast_horizon=forecast_horizon,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"mm256_day_{date_str}_{model_timestamp}"
    overlay_path = OUTPUT_DIR / f"{base_name}_overlay.png"
    window_path = OUTPUT_DIR / f"{base_name}_window.png"
    detail_path = OUTPUT_DIR / f"{base_name}_detail.csv"
    agg_path = OUTPUT_DIR / f"{base_name}_aggregate.csv"

    _plot_day_overlay(day_agg, date_str, overlay_path)
    _plot_single_window(day_detail, representative_sample_id, window_path)
    day_detail.to_csv(detail_path, index=False)
    day_agg.to_csv(agg_path, index=False)
    summary = _summarize_day(date_str, day_detail, day_agg, representative_sample_id)

    print(f"Day overlay -> {overlay_path}")
    print(f"Representative window -> {window_path}")
    print(f"Day detail CSV -> {detail_path}")
    print(f"Day aggregate CSV -> {agg_path}")

    return {
        "model_timestamp": model_timestamp,
        "date": date_str,
        "overlay_path": str(overlay_path),
        "window_path": str(window_path),
        "detail_path": str(detail_path),
        "aggregate_path": str(agg_path),
        "summary": summary,
        "representative_sample_id": int(representative_sample_id),
    }


def visualize_mm256_days(
    dates: list[str],
    model_timestamp: str | None = None,
    source: str = "local",
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
    train_ratio: float = 0.7,
) -> dict:
    """Generate visualizations and a markdown report for multiple days."""
    prediction_bundle = load_mm256_prediction_bundle(
        model_timestamp=model_timestamp,
        source=source,
        alert_rate=alert_rate,
        concentration_threshold=concentration_threshold,
        train_ratio=train_ratio,
    )
    outputs = []
    summaries = []
    for date_str in dates:
        output = visualize_mm256_day(date_str=date_str, prediction_bundle=prediction_bundle)
        outputs.append(output)
        summaries.append(output["summary"])

    report_path = OUTPUT_DIR / f"mm256_day_report_{prediction_bundle['model_timestamp']}.md"
    _write_day_report(
        report_path=report_path,
        model_timestamp=prediction_bundle["model_timestamp"],
        test_period=prediction_bundle["test_period"],
        summaries=summaries,
        outputs=outputs,
    )
    print(f"Day report -> {report_path}")

    return {
        "model_timestamp": prediction_bundle["model_timestamp"],
        "report_path": str(report_path),
        "days": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize MM256 predictions for one or more days")
    parser.add_argument("--date", action="append", required=True, help="Calendar day to analyze, format YYYY-MM-DD; repeat for multiple days")
    parser.add_argument("--model-timestamp", default=None, help="Saved model timestamp, for example mm256_20260317_181604")
    parser.add_argument("--source", choices=["bq", "cache", "local"], default="local")
    parser.add_argument("--alert-rate", type=float, default=1.0)
    parser.add_argument("--concentration-threshold", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--anchor-time", default=None, help="Optional input end time to force the representative window")
    args = parser.parse_args()

    if len(args.date) == 1:
        visualize_mm256_day(
            date_str=args.date[0],
            model_timestamp=args.model_timestamp,
            source=args.source,
            alert_rate=args.alert_rate,
            concentration_threshold=args.concentration_threshold,
            train_ratio=args.train_ratio,
            anchor_time=args.anchor_time,
        )
        return

    visualize_mm256_days(
        dates=args.date,
        model_timestamp=args.model_timestamp,
        source=args.source,
        alert_rate=args.alert_rate,
        concentration_threshold=args.concentration_threshold,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
