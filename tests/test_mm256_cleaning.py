import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_logic.data_cleaning import clean_dataframe
from scripts.preprocessor_MM256 import identify_active_days


def test_clean_dataframe_disables_sensor_disagreement_by_default():
    df = pd.DataFrame(
        {
            "MM256": [0.2] * 6,
            "TP1721": [20.0, 20.0, 20.0, 20.0, 20.0, 50.0],
            "TP1711": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        },
        index=pd.date_range("2024-01-01", periods=6, freq="s"),
    )

    cleaned = clean_dataframe(df, drop=False, verbose=False)

    assert "sensor_disagreement" in cleaned.columns
    assert not cleaned["sensor_disagreement"].any()


def test_identify_active_days_excludes_only_saturated_mm256_days():
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:01",
            "2024-01-02 00:00:00",
            "2024-01-02 00:00:01",
            "2024-01-03 00:00:00",
            "2024-01-03 00:00:01",
        ]
    )
    df = pd.DataFrame(
        {
            "MM256": [0.2, 1.5, 0.3, 30.0, 0.4, 2.2],
        },
        index=idx,
    )

    active_days = identify_active_days(df, concentration_threshold=1.0)

    assert active_days["date"].astype(str).tolist() == ["2024-01-01", "2024-01-03"]


def test_identify_active_days_can_keep_saturated_days_when_requested():
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:01",
            "2024-01-02 00:00:00",
            "2024-01-02 00:00:01",
        ]
    )
    df = pd.DataFrame(
        {
            "MM256": [0.2, 1.5, 0.3, 30.0],
        },
        index=idx,
    )

    active_days = identify_active_days(
        df,
        concentration_threshold=1.0,
        max_daily_peak_mm256=None,
    )

    assert active_days["date"].astype(str).tolist() == ["2024-01-01", "2024-01-02"]
