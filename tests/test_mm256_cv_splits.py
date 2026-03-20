import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.cv_time_series import build_window_balanced_folds_mm256
from scripts.preprocessor_MM256 import slice_windows_mm256


def _build_synthetic_mm256_df(n_rows: int = 2000) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n_rows, freq="s")
    return pd.DataFrame(
        {
            "MM256": [1.2] * n_rows,
            "ALERT": [1] * n_rows,
        },
        index=index,
    )


def test_window_balanced_folds_keep_validation_sizes_nearly_equal():
    df = _build_synthetic_mm256_df()

    folds = build_window_balanced_folds_mm256(
        train_df=df,
        n_splits=4,
        gap=300,
        window_length=300,
        forecast_horizon=120,
    )

    val_sizes = [fold["n_val_windows"] for fold in folds]

    assert len(folds) == 4
    assert max(val_sizes) - min(val_sizes) <= 1
    assert min(val_sizes) > 0


def test_window_balanced_fold_support_rows_rebuild_exact_window_counts():
    df = _build_synthetic_mm256_df()

    folds = build_window_balanced_folds_mm256(
        train_df=df,
        n_splits=3,
        gap=300,
        window_length=300,
        forecast_horizon=120,
    )

    for fold in folds:
        fold_train_df = df.loc[: fold["train_end_time"]].copy()
        fold_val_df = df.loc[fold["val_start_time"] : fold["val_end_time"]].copy()

        X_train, _ = slice_windows_mm256(
            fold_train_df,
            0,
            len(fold_train_df),
            300,
            120,
        )
        X_val, _ = slice_windows_mm256(
            fold_val_df,
            0,
            len(fold_val_df),
            300,
            120,
        )

        assert X_train.shape[0] == fold["n_train_windows"]
        assert X_val.shape[0] == fold["n_val_windows"]
