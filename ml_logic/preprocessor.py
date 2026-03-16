# Data pre-processing

import gc

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data_local() -> pd.DataFrame:
    """loads the dataset from local repertory raw_data into a DataFrame"""
    return pd.read_csv('raw_data/methane_data.csv')


def preprocess_split(df: pd.DataFrame, test_size: float =0.3, alert_rate : float =1.0) -> tuple:
    """ Preprocess function --> datetime / a single current intensity feature / train_test split / scaled values.
        Takes as input a DataFrame and a test_size (defaulting to 0.3 when no test_size argument is passed).
        Applies a MinMax scaler to each feature column and stores the corresponding scaler in a dictionary of scalers.
        The dictionary is needed to be able to inverse_transform each column with its own scaler later on.
        The function returns a tuple containing two scaled dateasets (train_data and test_data) AND a dictionary of scalers."""

    # Create a copy of the initial DataFrame before editing it
    df = df.copy()

    # Create datetime indication for time, drop previous time indications
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    df.set_index('time', inplace=True)
    df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)

    # Replace 5 columns on current intensity in 5 motors by one average indication
    df['AMP_AVG'] = df[['AMP1_IR', 'AMP2_IR', 'DMP3_IR', 'DMP4_IR', 'AMP5_IR']].mean(axis=1)
    df.drop(columns=['AMP1_IR', 'AMP2_IR', 'DMP3_IR', 'DMP4_IR', 'AMP5_IR'], inplace=True)

    # Create a binary flag for observations in which methane concentration exceeds the alert_rate: 1 means observation above alert_rate
    df["ALERT"] = df[["MM256", "MM263", "MM264"]].ge(alert_rate).any(axis=1).astype(int)

    # Split dataframe into train and test datasets - not separating targets from features for now (model-dependent need)
    train_data, test_data = train_test_split(df, test_size=test_size, shuffle=False)

    # Scale numerical features without data leakage
    # features_to_scale = ['AN311', 'AN422', 'AN423', 'TP1721', 'RH1722', 'BA1723', 'TP1711', 'RH1712', 'BA1713', 'MM252', 'MM261', 'MM262', 'MM263',\
    #                  'MM264', 'MM256', 'MM211', 'CM861','CR863', 'P_864', 'TC862', 'WM868', 'AMP_AVG', 'F_SIDE', 'V']
    features_to_scale = train_data.select_dtypes(include='number')
    scalers = {}
    for feature in features_to_scale :
        mm_scaler = MinMaxScaler()
        train_data[feature] = mm_scaler.fit_transform(train_data[[feature]])
        test_data[feature] =  mm_scaler.transform(test_data[[feature]])
        scalers[feature] = mm_scaler

    return train_data, test_data, scalers


def preprocess_max(df: pd.DataFrame, test_size: float =0.3, alert_rate : float =1.0) -> tuple:
    """ Preprocess function --> datetime / max reading of MM256, MM263 MM264 / single current intensity feature / train_test split / scaled values.
        Takes as input a DataFrame and a test_size (defaulting to 0.3 when no test_size argument is passed).
        Aggregates the readings of the 3 key Methanometers in one synthetic measurement representing the highest reading of the lot.
        Applies a MinMax scaler to each feature column and stores the corresponding scaler in a dictionary of scalers.
        The dictionary is needed to be able to inverse_transform each column with its own scaler later on.
        The function returns a tuple containing two scaled dateasets (train_data and test_data) AND a dictionary of scalers."""

    # Create a copy of the initial DataFrame before editing it
    df = df.copy()

    # Create datetime indication for time, drop previous time indications
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    df.set_index('time', inplace=True)
    df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)

    # Replace 5 columns on current intensity in 5 motors by one average indication
    df['AMP_AVG'] = df[['AMP1_IR', 'AMP2_IR', 'DMP3_IR', 'DMP4_IR', 'AMP5_IR']].mean(axis=1)
    df.drop(columns=['AMP1_IR', 'AMP2_IR', 'DMP3_IR', 'DMP4_IR', 'AMP5_IR'], inplace=True)

    # Create a synthetic indicator for methane concentration - the highest reading of the MM256, MM263, MM264 group
    df['MM_RATE'] = df[['MM256', 'MM263', 'MM264']].max(axis=1)
    df.drop(columns=['MM256', 'MM263', 'MM264'], inplace=True)

    # Create a synthetic indicator for airflow speed - the average reading of the AN422, AN423 group
    df['AN_RATE'] = df[['AN422', 'AN423']].mean(axis=1)
    df.drop(columns=['AN422', 'AN423'], inplace=True)

    # Create a synthetic indicator for temperature - the average reading of the TP1711, TP1721 group
    df['TP_RATE'] = df[['TP1711', 'TP1721']].mean(axis=1)
    df.drop(columns=['TP1711', 'TP1721'], inplace=True)

    # Create a synthetic indicator for air humidity - the average reading of the RH1712, RH1722 group
    df['RH_RATE'] = df[['RH1712', 'RH1722']].mean(axis=1)
    df.drop(columns=['RH1712', 'RH1722'], inplace=True)

    # Create a synthetic indicator for athmospheric pressure - the average reading of the BA1713, BA1723 group
    df['BA_RATE'] = df[['BA1713', 'BA1723']].mean(axis=1)
    df.drop(columns=['BA1713', 'BA1723'], inplace=True)

    # Create a binary flag for observations in which methane concentration exceeds the alert_rate: 1 means observation above alert_rate
    df["ALERT"] = df['MM_RATE'].ge(alert_rate).astype(int)

    # Split dataframe into train and test datasets - not separating targets from features for now (model-dependent need)
    train_data, test_data = train_test_split(df, test_size=test_size, shuffle=False)

    # Scale numerical features without data leakage
    features_to_scale = train_data.select_dtypes(include='number')
    scalers = {}
    for feature in features_to_scale :
        mm_scaler = MinMaxScaler()
        train_data[feature] = mm_scaler.fit_transform(train_data[[feature]])
        test_data[feature] =  mm_scaler.transform(test_data[[feature]])
        scalers[feature] = mm_scaler

    return train_data, test_data, scalers


def slice_arrays(df: pd.DataFrame, start_index, stop_index, window_length_in_sec: int =360, forecast_horizon_in_sec: int =180) -> tuple :
    ''' creates a subset of the input dataframe - the observations between start_index and stop_index (excluded),
        prepares dataframe slices containing observations exceeding the alert_rate passed to function prerocess_split,
        the shape of the slices is governed by arguments window_length_in_sec and forecast_horizon_in_sec,
        returns a tuple of tensors '''

    input_length_in_sec = window_length_in_sec - forecast_horizon_in_sec
    if input_length_in_sec <= 0:
        raise ValueError("window_length_in_sec must be > forecast_horizon_in_sec")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("slice_arrays expects df.index to be a DatetimeIndex")

    # Creates subset to limit training time initially
    subset_df = df.iloc[start_index:stop_index]

    # Define columns to be included in X and y - eliminating columns not required for modelling
    excluded_cols = ['datetime', 'slice_id', 'trigger_time', 't_rel_s', 'ALERT']
    feature_cols = [c for c in subset_df.columns if c not in excluded_cols]
    target_cols = ['MM_RATE'] if 'MM_RATE' in subset_df.columns else ['MM256', 'MM263', 'MM264']

    # Use boolean mask to identify index positions in which methane concentration exceeds the alert_rate
    trigger_mask = (subset_df['ALERT'] == 1)
    trigger_times = subset_df.index[trigger_mask]

    if trigger_times.empty:
        X_empty = np.empty((0, input_length_in_sec + 1, len(feature_cols)), dtype=np.float32)
        y_empty = np.empty((0, forecast_horizon_in_sec, len(target_cols)), dtype=np.float32)
        return X_empty, y_empty

    # Pull the sequences directly into NumPy arrays to avoid the large
    # intermediate DataFrames created by repeated copy/concat operations.
    subset_index = subset_df.index
    feature_values = subset_df.loc[:, feature_cols].to_numpy(copy=False)
    target_values = subset_df.loc[:, target_cols].to_numpy(copy=False)
    one_second = pd.Timedelta(seconds=1)

    X, y = [], []
    for t0 in trigger_times:
        X_times = pd.date_range(end=t0, periods=input_length_in_sec + 1, freq='s')
        y_times = pd.date_range(start=t0 + one_second, periods=forecast_horizon_in_sec, freq='s')

        X_idx = subset_index.get_indexer(X_times)
        y_idx = subset_index.get_indexer(y_times)

        if (X_idx < 0).any() or (y_idx < 0).any():
            continue

        X.append(np.asarray(feature_values[X_idx], dtype=np.float32))
        y.append(np.asarray(target_values[y_idx], dtype=np.float32))

    del feature_values
    del target_values
    del trigger_mask
    del trigger_times
    del subset_df

    if not X:
        X_empty = np.empty((0, input_length_in_sec +1, len(feature_cols)), dtype=np.float32)
        y_empty = np.empty((0, forecast_horizon_in_sec, len(target_cols)), dtype=np.float32)
        gc.collect()
        return X_empty, y_empty

    X_array = np.stack(X)
    y_array = np.stack(y)

    del X
    del y
    gc.collect()

    return X_array, y_array


def sample_datasets(df_scaled: pd.DataFrame) -> list :
    """ splits a pre-processed, scaled main dataset into 3 sample datasets focused on MM256, MM263, MM264.
        takes a PRE-PROCESSED, SCALED dataset as input.
        returns a list containing 3 sample datasets corresponding to captors MM256, MM263 and MM264 in this order."""

    # create sample scaled dataset for sensor MM256 (methanometer)
    MM256_features = ['AN422', 'TP1711', 'RH1712', 'BA1713', 'MM256', 'CM861', 'CR863', 'P_864', 'TC862', 'WM868', 'AMP_AVG', 'F_SIDE', 'V', 'ALERT']
    MM256_df = df_scaled[MM256_features]

    # create sample scaled dataset for sensor MM263 (methanometer)
    MM263_features = ['AN422', 'TP1711', 'RH1712', 'BA1713', 'MM263', 'CM861', 'CR863', 'P_864', 'TC862', 'WM868', 'AMP_AVG', 'F_SIDE', 'V', 'ALERT']
    MM263_df = df_scaled[MM263_features]

    # create sample scaled dataset for sensor MM264 (methanometer)
    MM264_features = ['AN422', 'TP1711', 'RH1712', 'BA1713', 'MM264', 'CM861', 'CR863', 'P_864', 'TC862', 'WM868', 'AMP_AVG', 'F_SIDE', 'V', 'ALERT']
    MM264_df = df_scaled[MM264_features]

    return [MM256_df, MM263_df, MM264_df]


def feature_target(df: pd.DataFrame, target: str) -> tuple:
    """ within a DataFrame, isolates target variable from features
        takes as input a dataframe and a string corresponding to the column representing the target"""
    y = df[target]
    X = df.drop(columns=[target])
    return X, y
