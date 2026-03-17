"""
Methane event detection and analysis pipeline.

Loads methane_data.csv, detects concentration events above a threshold,
generates event/day ID tables, produces visualizations, and optionally
pushes results to BigQuery.

Usage:
    python scripts/qn_methane_analysis.py [--threshold 0.7] [--gap 120] [--push-bq]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "..", "research", "analysis", "qn_analysis")
RAW_DATA_DIR = os.path.join(ANALYSIS_DIR, "raw_data")
CSV_PATH = os.path.join(RAW_DATA_DIR, "methane_data.csv")
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "processed_data")
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")

# All methane sensors (from attribute_information.txt)
ALL_METHANE_SENSORS = ["MM252", "MM261", "MM262", "MM263", "MM264", "MM256", "MM211", "CM861"]


# ---------------------------------------------------------------------------
# 1. Load & build datetime
# ---------------------------------------------------------------------------
def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute", "second"]]
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  -> {len(df):,} rows, {df['datetime'].min()} to {df['datetime'].max()}")
    return df


# ---------------------------------------------------------------------------
# 2. Preprocess: time indexes + event flag (threshold-only, no upper bound)
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame, sensor: str, threshold: float) -> pd.DataFrame:
    df = df.copy()
    # Robust day index (works across years)
    origin = df["datetime"].min().normalize()
    df["day_nb_ind"] = (df["datetime"] - origin).dt.days + 1
    df["hour_ind"] = (
        (df["datetime"] - origin).dt.total_seconds() // 3600
    ).astype(int)
    df["min_ind"] = (
        (df["datetime"] - origin).dt.total_seconds() // 60
    ).astype(int)
    df["sec_ind"] = (
        (df["datetime"] - origin).dt.total_seconds()
    ).astype(int)

    # Event flag: concentration >= threshold
    df["event"] = (df[sensor] >= threshold).astype(int)
    return df


# ---------------------------------------------------------------------------
# 3. Filter to event rows
# ---------------------------------------------------------------------------
def filter_events(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["event"] == 1].copy()


# ---------------------------------------------------------------------------
# 4. Assign event IDs (gap_threshold in seconds between consecutive rows
#    to split into separate events)
# ---------------------------------------------------------------------------
def assign_event_ids(df_events: pd.DataFrame, gap_threshold: int = 120) -> pd.DataFrame:
    df_events = df_events.sort_values("datetime").reset_index(drop=True)
    df_events["time_gap"] = df_events["datetime"].diff()
    df_events["event_id"] = (
        df_events["time_gap"] > pd.Timedelta(seconds=gap_threshold)
    ).cumsum() + 1
    return df_events


# ---------------------------------------------------------------------------
# 5. Day-to-event mapping
# ---------------------------------------------------------------------------
def build_day_event_map(df_events: pd.DataFrame) -> pd.DataFrame:
    return (
        df_events[["day_nb_ind", "event_id", "datetime"]]
        .assign(date=lambda d: d["datetime"].dt.date)
        .drop(columns="datetime")
        .drop_duplicates()
        .sort_values(["day_nb_ind", "event_id"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 6. Event summary
# ---------------------------------------------------------------------------
def build_event_summary(df_events: pd.DataFrame, sensor: str) -> pd.DataFrame:
    summary = (
        df_events.groupby("event_id")
        .agg(
            start_time=("datetime", "min"),
            end_time=("datetime", "max"),
            duration_seconds=("datetime", lambda x: (x.max() - x.min()).total_seconds()),
            max_concentration=(sensor, "max"),
            mean_concentration=(sensor, "mean"),
            num_measurements=("datetime", "count"),
        )
        .reset_index()
    )
    summary["date"] = summary["start_time"].dt.date
    return summary


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
def plot_full_timeseries(df: pd.DataFrame, sensor: str, threshold: float, out_dir: str):
    """Full timeseries with threshold line."""
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(df["datetime"], df[sensor], linewidth=0.3, alpha=0.7, label=sensor)
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold = {threshold}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Methane concentration (%CH4)")
    ax.set_title(f"Methane concentration over time — {sensor}")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(out_dir, f"timeseries_{sensor}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_event_durations(summary: pd.DataFrame, sensor: str, out_dir: str):
    """Bar chart of event durations."""
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(
        summary["event_id"],
        summary["duration_seconds"] / 60,
        color="steelblue",
        edgecolor="navy",
        alpha=0.8,
    )
    ax.set_xlabel("Event ID")
    ax.set_ylabel("Duration (minutes)")
    ax.set_title(f"Duration of detected methane events — {sensor}")
    plt.tight_layout()
    path = os.path.join(out_dir, f"event_durations_{sensor}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_event_max_concentration(summary: pd.DataFrame, sensor: str, threshold: float, out_dir: str):
    """Bar chart of max concentration per event."""
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["crimson" if v >= threshold * 2 else "orange" if v >= threshold * 1.5 else "steelblue"
              for v in summary["max_concentration"]]
    ax.bar(summary["event_id"], summary["max_concentration"], color=colors, edgecolor="navy", alpha=0.8)
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold = {threshold}")
    ax.set_xlabel("Event ID")
    ax.set_ylabel("Max concentration (%CH4)")
    ax.set_title(f"Peak concentration per methane event — {sensor}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"event_max_conc_{sensor}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_top_events(df: pd.DataFrame, df_events: pd.DataFrame, summary: pd.DataFrame,
                    sensor: str, threshold: float, out_dir: str, n_top: int = 6):
    """Zoom into the top-N longest events showing concentration evolution."""
    top = summary.nlargest(n_top, "duration_seconds")
    n_cols = 2
    n_rows = (n_top + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, (_, row) in enumerate(top.iterrows()):
        eid = row["event_id"]
        mask = df_events["event_id"] == eid
        evt = df_events[mask]
        # add 5 min padding
        pad = pd.Timedelta(minutes=5)
        ctx = df[(df["datetime"] >= evt["datetime"].min() - pad) &
                 (df["datetime"] <= evt["datetime"].max() + pad)]

        ax = axes[i]
        ax.plot(ctx["datetime"], ctx[sensor], linewidth=0.8, color="steelblue")
        ax.fill_between(evt["datetime"], evt[sensor], alpha=0.3, color="orange")
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=0.8)
        dur_min = row["duration_seconds"] / 60
        ax.set_title(f"Event {eid} — {dur_min:.1f} min, peak {row['max_concentration']:.2f}%")
        ax.set_ylabel("%CH4")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Top {n_top} longest methane events — {sensor}", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f"top_events_{sensor}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_daily_event_heatmap(df_events: pd.DataFrame, summary: pd.DataFrame, sensor: str, out_dir: str):
    """Events per day as a bar chart."""
    daily = df_events.groupby(df_events["datetime"].dt.date).agg(
        n_events=("event_id", "nunique"),
        total_minutes=("datetime", lambda x: len(x) / 60),
        max_conc=(sensor, "max"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    axes[0].bar(daily.index, daily["n_events"], color="steelblue")
    axes[0].set_ylabel("# events")
    axes[0].set_title(f"Daily methane event overview — {sensor}")

    axes[1].bar(daily.index, daily["total_minutes"], color="orange")
    axes[1].set_ylabel("Total event minutes")

    axes[2].bar(daily.index, daily["max_conc"], color="crimson")
    axes[2].set_ylabel("Max concentration (%CH4)")
    axes[2].set_xlabel("Date")

    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(out_dir, f"daily_overview_{sensor}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# 8. Build consolidated cross-sensor table
# ---------------------------------------------------------------------------
def build_consolidated_table(
    all_summaries: dict[str, pd.DataFrame],
    all_day_maps: dict[str, pd.DataFrame],
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a consolidated event summary and day-event map across all sensors.

    Event summary: one row per (sensor, event_id) with columns prefixed by sensor.
    Consolidated day table: one row per date, with event_id_<sensor> and day_id_<sensor> columns.
    """
    # --- Consolidated event summary (long format with sensor column) ---
    parts = []
    for sensor, summary in all_summaries.items():
        s = summary.copy()
        s["sensor"] = sensor
        s.rename(columns={"event_id": f"event_id_{sensor}"}, inplace=True)
        parts.append(s)
    consolidated_events = pd.concat(parts, ignore_index=True)

    # --- Consolidated day table (wide format: one row per date) ---
    # Get all unique dates from the dataset
    all_dates = sorted(
        set().union(*(dm["date"].unique() for dm in all_day_maps.values()))
    )
    day_rows = []
    for date in all_dates:
        row = {"date": date}
        # Find the day_nb_ind for this date from any sensor that has it
        for sensor, dm in all_day_maps.items():
            match = dm[dm["date"] == date]
            if not match.empty:
                row["day_nb_ind"] = match["day_nb_ind"].iloc[0]
                break
        for sensor, dm in all_day_maps.items():
            match = dm[dm["date"] == date]
            if not match.empty:
                # List of event IDs for this sensor on this date
                event_ids = sorted(int(x) for x in match["event_id"].unique())
                row[f"event_id_{sensor}"] = str(event_ids)
                row[f"n_events_{sensor}"] = len(event_ids)
            else:
                row[f"event_id_{sensor}"] = "[]"
                row[f"n_events_{sensor}"] = 0
        day_rows.append(row)
    consolidated_days = pd.DataFrame(day_rows)

    return consolidated_events, consolidated_days


# ---------------------------------------------------------------------------
# 9. BigQuery push
# ---------------------------------------------------------------------------
def _get_bq_client():
    """Return a BigQuery client using .env credentials."""
    from dotenv import load_dotenv
    from google.cloud import bigquery

    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        import subprocess
        main_root = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=SCRIPT_DIR, text=True
        ).strip().replace("/.git", "")
        env_path = os.path.join(main_root, ".env")
    load_dotenv(env_path)

    project = os.environ["GCP_PROJECT"]
    dataset = os.environ["BQ_DATASET"]
    region = os.environ["BQ_REGION"]
    client = bigquery.Client(project=project, location=region)
    return client, project, dataset


def push_to_bigquery(event_summary: pd.DataFrame, day_event_map: pd.DataFrame,
                     table_suffix: str = ""):
    """Push event summary and day-event map to BigQuery for a single sensor."""
    from google.cloud import bigquery

    client, project, dataset = _get_bq_client()
    suffix = f"_{table_suffix}" if table_suffix else ""
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")

    # -- Event summary table --
    table_events = f"{project}.{dataset}.methane_events_summary{suffix}"
    es = event_summary.copy()
    es["start_time"] = es["start_time"].astype(str)
    es["end_time"] = es["end_time"].astype(str)
    es["date"] = es["date"].astype(str)
    client.load_table_from_dataframe(es, table_events, job_config=job_config).result()
    print(f"  -> Pushed event summary ({len(es)} rows) to {table_events}")

    # -- Day-event map table --
    table_days = f"{project}.{dataset}.methane_day_event_map{suffix}"
    dem = day_event_map.copy()
    dem["date"] = dem["date"].astype(str)
    client.load_table_from_dataframe(dem, table_days, job_config=job_config).result()
    print(f"  -> Pushed day-event map ({len(dem)} rows) to {table_days}")


def push_consolidated_to_bigquery(consolidated_events: pd.DataFrame,
                                  consolidated_days: pd.DataFrame):
    """Push the cross-sensor consolidated tables to BigQuery."""
    from google.cloud import bigquery

    client, project, dataset = _get_bq_client()
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")

    # -- Consolidated events (long format) --
    table_ce = f"{project}.{dataset}.methane_events_all_sensors"
    ce = consolidated_events.copy()
    for col in ["start_time", "end_time"]:
        if col in ce.columns:
            ce[col] = ce[col].astype(str)
    if "date" in ce.columns:
        ce["date"] = ce["date"].astype(str)
    client.load_table_from_dataframe(ce, table_ce, job_config=job_config).result()
    print(f"  -> Pushed consolidated events ({len(ce)} rows) to {table_ce}")

    # -- Consolidated day map (wide format) --
    table_cd = f"{project}.{dataset}.methane_day_event_map_all_sensors"
    cd = consolidated_days.copy()
    if "date" in cd.columns:
        cd["date"] = cd["date"].astype(str)
    client.load_table_from_dataframe(cd, table_cd, job_config=job_config).result()
    print(f"  -> Pushed consolidated day map ({len(cd)} rows) to {table_cd}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_single_sensor(df: pd.DataFrame, sensor: str, threshold: float,
                      gap: int, push_bq: bool, skip_plots: bool = False):
    """Run the full pipeline for a single sensor. Returns (summary, day_map) or None."""
    print(f"\n{'='*60}")
    print(f"Sensor: {sensor}, Threshold: {threshold}, Gap: {gap}s")
    print(f"{'='*60}")

    df_proc = preprocess(df, sensor, threshold)
    df_events = filter_events(df_proc)
    print(f"Event rows: {len(df_events):,} / {len(df):,} ({100*len(df_events)/len(df):.2f}%)")

    if len(df_events) == 0:
        print(f"No events detected for {sensor}. Skipping.")
        return None, None

    df_events = assign_event_ids(df_events, gap_threshold=gap)
    n_events = df_events["event_id"].nunique()
    print(f"Distinct events: {n_events}")

    day_map = build_day_event_map(df_events)
    print(f"Days with events: {day_map['day_nb_ind'].nunique()}")

    summary = build_event_summary(df_events, sensor)

    # Print top events
    print(f"\n--- Top 5 events by duration ({sensor}) ---")
    top5 = summary.nlargest(5, "duration_seconds")[
        ["event_id", "date", "start_time", "end_time", "duration_seconds",
         "max_concentration", "mean_concentration", "num_measurements"]
    ]
    print(top5.to_string(index=False))

    # Save CSVs
    summary.to_csv(os.path.join(OUTPUT_DIR, f"methane_events_summary_{sensor}.csv"), index=False)
    day_map.to_csv(os.path.join(OUTPUT_DIR, f"methane_day_event_map_{sensor}.csv"), index=False)

    # Plots
    if not skip_plots:
        print(f"Generating plots for {sensor}...")
        plot_full_timeseries(df, sensor, threshold, PLOTS_DIR)
        plot_event_durations(summary, sensor, PLOTS_DIR)
        plot_event_max_concentration(summary, sensor, threshold, PLOTS_DIR)
        plot_top_events(df, df_events, summary, sensor, threshold, PLOTS_DIR)
        plot_daily_event_heatmap(df_events, summary, sensor, PLOTS_DIR)

    # Per-sensor BQ push
    if push_bq:
        print(f"Pushing {sensor} tables to BigQuery...")
        push_to_bigquery(summary, day_map, table_suffix=sensor)

    return summary, day_map


def main():
    parser = argparse.ArgumentParser(description="Methane event detection pipeline")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Methane concentration threshold (default: 0.7)")
    parser.add_argument("--gap", type=int, default=120,
                        help="Gap in seconds to split events (default: 120)")
    parser.add_argument("--sensor", type=str, default="MM264",
                        help="Target sensor column (default: MM264)")
    parser.add_argument("--all-sensors", action="store_true",
                        help="Run for all methane sensors")
    parser.add_argument("--push-bq", action="store_true",
                        help="Push results to BigQuery")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation (faster)")
    parser.add_argument("--csv", type=str, default=CSV_PATH,
                        help="Path to methane CSV")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load data once
    df = load_data(args.csv)

    sensors = ALL_METHANE_SENSORS if args.all_sensors else [args.sensor]

    all_summaries = {}
    all_day_maps = {}

    for sensor in sensors:
        summary, day_map = run_single_sensor(
            df, sensor, args.threshold, args.gap, args.push_bq, args.skip_plots
        )
        if summary is not None:
            all_summaries[sensor] = summary
            all_day_maps[sensor] = day_map

    # Build and save consolidated table when multiple sensors
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print("Building consolidated cross-sensor tables")
        print(f"{'='*60}")

        consolidated_events, consolidated_days = build_consolidated_table(
            all_summaries, all_day_maps, df
        )

        # Save CSVs
        ce_path = os.path.join(OUTPUT_DIR, "methane_events_all_sensors.csv")
        consolidated_events.to_csv(ce_path, index=False)
        print(f"Saved consolidated events -> {ce_path}")

        cd_path = os.path.join(OUTPUT_DIR, "methane_day_event_map_all_sensors.csv")
        consolidated_days.to_csv(cd_path, index=False)
        print(f"Saved consolidated day map -> {cd_path}")

        print(f"\nConsolidated events: {len(consolidated_events):,} rows across {len(all_summaries)} sensors")
        print(f"Consolidated day map: {len(consolidated_days)} days")
        print(f"\nSensors included: {list(all_summaries.keys())}")
        print(f"\nConsolidated day map columns: {list(consolidated_days.columns)}")
        print(f"\n--- Consolidated day map (first 10 rows) ---")
        print(consolidated_days.head(10).to_string(index=False))

        if args.push_bq:
            print("\nPushing consolidated tables to BigQuery...")
            push_consolidated_to_bigquery(consolidated_events, consolidated_days)

    print("\nDone!")


if __name__ == "__main__":
    main()
