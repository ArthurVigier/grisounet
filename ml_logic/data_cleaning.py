# Data cleaning — detect and remove abnormal / inconsistent sensor values

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Physical validity ranges for each sensor family.
# Values outside these bounds are considered sensor errors or sentinel codes.
# Ranges are intentionally generous to avoid discarding real extreme events.
# ---------------------------------------------------------------------------
SENSOR_BOUNDS = {
    # Airflow (m/s) — negative values can indicate reverse flow, but large
    # negatives (e.g. -266) are clearly sensor errors / sentinel codes.
    "AN311":  (-10.0, 20.0),
    "AN422":  (-10.0, 20.0),
    "AN423":  (-10.0, 20.0),
    # Methane concentration (% vol) — LEL is 5 %, anything above ~5 % in a
    # ventilated mine gallery is suspect; above 10 % is physically unrealistic.
    "MM252":  (0.0, 10.0),
    "MM261":  (0.0, 10.0),
    "MM262":  (0.0, 10.0),
    "MM263":  (0.0, 10.0),
    "MM264":  (0.0, 10.0),
    "MM256":  (0.0, 10.0),
    "MM211":  (0.0, 10.0),
    "CM861":  (0.0, 100.0),   # high-concentration sensor, wider range
    # Temperature (°C) — mine galleries are typically 10–40 °C
    "TP1721": (-10.0, 60.0),
    "TP1711": (-10.0, 60.0),
    # Humidity (%) — physical range 0–100
    "RH1722": (0.0, 100.0),
    "RH1712": (0.0, 100.0),
    # Barometric pressure (hPa) — reasonable surface/underground range
    "BA1723": (850.0, 1100.0),
    "BA1713": (850.0, 1100.0),
    # Pipeline / drainage sensors
    "CR863":  (-500.0, 500.0),   # pressure diff (Pa)
    "P_864":  (0.0, 1000.0),     # pipeline pressure (kPa)
    "TC862":  (-10.0, 60.0),     # pipeline temperature (°C)
    "WM868":  (0.0, 500.0),      # flow rate
    # Motor currents (A) — should be non-negative
    "AMP1_IR": (0.0, 500.0),
    "AMP2_IR": (0.0, 500.0),
    "DMP3_IR": (0.0, 500.0),
    "DMP4_IR": (0.0, 500.0),
    "AMP5_IR": (0.0, 500.0),
    # Discrete / categorical-like
    "F_SIDE":  (0.0, 1.0),
    "V":       (0.0, 50.0),
}


def flag_out_of_range(
    df: pd.DataFrame,
    bounds: Optional[dict] = None,
) -> pd.DataFrame:
    """Flag rows where any sensor reading falls outside its physical validity range.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (original column names, i.e. AN311, MM263, etc.).
    bounds : dict, optional
        {column_name: (low, high)}. Defaults to SENSOR_BOUNDS.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an extra boolean column ``out_of_range`` (True = at
        least one sensor value is outside bounds) and one boolean column per
        checked sensor named ``oor_<col>`` for per-sensor diagnostics.
    """
    bounds = bounds or SENSOR_BOUNDS
    df = df.copy()
    oor_mask = pd.Series(False, index=df.index)

    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        col_mask = (df[col] < lo) | (df[col] > hi)
        df[f"oor_{col}"] = col_mask
        oor_mask = oor_mask | col_mask

    df["out_of_range"] = oor_mask
    return df


def flag_frozen_sensors(
    df: pd.DataFrame,
    sensor_cols: Optional[list] = None,
    window: int = 300,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Flag periods where a sensor reports the exact same value for too long.

    A sensor stuck at a constant value for `window` consecutive seconds
    (default 5 min) likely indicates a malfunction or communication loss.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by time at 1-second resolution.
    sensor_cols : list[str], optional
        Columns to check. Defaults to all methane + airflow sensors.
    window : int
        Number of consecutive identical readings to trigger the flag.
    threshold : float
        Minimum absolute value of the rolling std to consider "varying".
        Default 0.0 means strictly zero variance triggers the flag.

    Returns
    -------
    pd.DataFrame
        Copy with a boolean ``frozen_sensor`` column.
    """
    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith(("AN", "MM"))]

    df = df.copy()
    frozen = pd.Series(False, index=df.index)

    for col in sensor_cols:
        if col not in df.columns:
            continue
        rolling_std = df[col].rolling(window, min_periods=window).std()
        frozen = frozen | (rolling_std <= threshold)

    df["frozen_sensor"] = frozen
    return df


def flag_timestamp_gaps(
    df: pd.DataFrame,
    max_gap_seconds: int = 2,
    time_col: str = "time",
) -> pd.DataFrame:
    """Flag rows immediately following a gap in the time series.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a DatetimeIndex or a datetime column named *time_col*.
    max_gap_seconds : int
        Maximum allowed gap between consecutive rows (inclusive). Rows after
        a gap larger than this are flagged.

    Returns
    -------
    pd.DataFrame
        Copy with a boolean ``after_gap`` column.
    """
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index.to_series()
    else:
        ts = df[time_col]

    gaps = ts.diff().dt.total_seconds()
    df["after_gap"] = gaps > max_gap_seconds
    return df


def flag_sensor_disagreement(
    df: pd.DataFrame,
    groups: Optional[dict] = None,
    z_threshold: float = 4.0,
) -> pd.DataFrame:
    """Flag rows where co-located sensors diverge abnormally.

    For each group of sensors that should measure similar quantities (e.g.
    two temperature probes), flag rows where the difference between any pair
    exceeds *z_threshold* standard deviations of that difference.

    Parameters
    ----------
    groups : dict, optional
        ``{group_name: [col1, col2, ...]}``.  Defaults to paired
        temperature, humidity, and pressure probes.
    z_threshold : float
        Number of standard deviations beyond which the difference is flagged.

    Returns
    -------
    pd.DataFrame
        Copy with a boolean ``sensor_disagreement`` column.
    """
    if groups is None:
        groups = {
            "temperature": ["TP1721", "TP1711"],
            "humidity":    ["RH1722", "RH1712"],
            "pressure":    ["BA1723", "BA1713"],
        }

    df = df.copy()
    disagree = pd.Series(False, index=df.index)

    for name, cols in groups.items():
        present = [c for c in cols if c in df.columns]
        if len(present) < 2:
            continue
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                diff = df[present[i]] - df[present[j]]
                mu, sigma = diff.mean(), diff.std()
                if sigma > 0:
                    disagree = disagree | ((diff - mu).abs() > z_threshold * sigma)

    df["sensor_disagreement"] = disagree
    return df


# ---------------------------------------------------------------------------
# High-level wrapper: detect everything and optionally drop flagged rows
# ---------------------------------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    drop: bool = False,
    bounds: Optional[dict] = None,
    frozen_window: int = 300,
    gap_seconds: int = 2,
    z_threshold: float = 4.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run all anomaly detectors and optionally remove flagged rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (original column names).
    drop : bool
        If True, rows flagged by *any* detector are removed.
        If False, only the flag columns are added (for inspection).
    verbose : bool
        Print a summary of how many rows each detector flags.

    Returns
    -------
    pd.DataFrame
    """
    n = len(df)

    df = flag_out_of_range(df, bounds=bounds)
    df = flag_frozen_sensors(df, window=frozen_window)
    df = flag_timestamp_gaps(df, max_gap_seconds=gap_seconds)
    df = flag_sensor_disagreement(df, z_threshold=z_threshold)

    flag_cols = ["out_of_range", "frozen_sensor", "after_gap", "sensor_disagreement"]
    df["any_anomaly"] = df[flag_cols].any(axis=1)

    if verbose:
        print("=== Data cleaning summary ===")
        for flag in flag_cols:
            count = df[flag].sum()
            print(f"  {flag:25s}: {count:>10,} rows  ({count / n * 100:.3f} %)")
        total = df["any_anomaly"].sum()
        print(f"  {'any_anomaly':25s}: {total:>10,} rows  ({total / n * 100:.3f} %)")
        print(f"  {'total rows':25s}: {n:>10,}")

    if drop:
        # Remove per-sensor diagnostic columns to keep output clean
        oor_cols = [c for c in df.columns if c.startswith("oor_")]
        df = df[~df["any_anomaly"]].drop(columns=flag_cols + oor_cols + ["any_anomaly"])

    return df
