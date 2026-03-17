# Data pre-processing

import gc

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# %pip install -U aeon NOTE: catch22 module, replaces pandas, scikit learn etc by lower versions to manage dependencies

def cast_float_columns_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Cast floating-point columns to float32 while leaving ints/datetimes unchanged."""
    float_cols = df.select_dtypes(include=['floating']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)
    return df


def load_data_local() -> pd.DataFrame:
    """loads the dataset from local repertory raw_data into a DataFrame"""
    return cast_float_columns_to_float32(pd.read_csv('raw_data/methane_data.csv'))


def build_sequence_arrays(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    alert_values: np.ndarray,
    input_steps: int,
    forecast_horizon_in_sec: int,
    step_size_in_sec: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Build X/y sequence tensors using positional NumPy indexing only."""
    if input_steps <= 0:
        raise ValueError("input_steps must be > 0")
    if forecast_horizon_in_sec <= 0:
        raise ValueError("forecast_horizon_in_sec must be > 0")
    if step_size_in_sec <= 0:
        raise ValueError("step_size_in_sec must be > 0")

    history_span = (input_steps - 1) * step_size_in_sec
    trigger_positions = np.flatnonzero(alert_values == 1)

    if trigger_positions.size == 0:
        X_empty = np.empty((0, input_steps, feature_values.shape[1]), dtype=np.float32)
        y_empty = np.empty((0, forecast_horizon_in_sec, target_values.shape[1]), dtype=np.float32)
        return X_empty, y_empty

    valid_positions = trigger_positions[
        (trigger_positions >= history_span)
        & (trigger_positions + forecast_horizon_in_sec < len(alert_values))]

    if valid_positions.size == 0:
        X_empty = np.empty((0, input_steps, feature_values.shape[1]), dtype=np.float32)
        y_empty = np.empty((0, forecast_horizon_in_sec, target_values.shape[1]), dtype=np.float32)
        return X_empty, y_empty

    x_offsets = np.arange(-history_span, 1, step_size_in_sec, dtype=np.int32)
    y_offsets = np.arange(1, forecast_horizon_in_sec + 1, dtype=np.int32)

    X_indices = valid_positions[:, None] + x_offsets[None, :]
    y_indices = valid_positions[:, None] + y_offsets[None, :]

    X_array = feature_values[X_indices].astype(np.float32, copy=False)
    y_array = target_values[y_indices].astype(np.float32, copy=False)

    return X_array, y_array


def preprocess_split(input_df: pd.DataFrame, test_size: float =0.3, alert_rate : float =1.0) -> tuple:
    """ Preprocess function --> datetime / a single current intensity feature / train_test split / scaled values. Uses NumPy to speed-up computations.
        Takes as input a DataFrame and a test_size (defaulting to 0.3 when no test_size argument is passed).
        Applies a MinMax scaler to each feature column and stores the corresponding scaler in a dictionary of scalers.
        The dictionary is needed to be able to inverse_transform each column with its own scaler later on.
        The function returns a tuple containing two scaled dateasets (train_data and test_data) AND a dictionary of scalers."""

    # Create a copy of the initial DataFrame before editing it
    df = input_df.copy()
    df['CR863'] = df['CR863'].astype(np.float32)

    # Create datetime indication for time, drop previous time indications
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    df.set_index('time', inplace=True)
    df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)

    # Replace 5 columns on current intensity in 5 motors by one average indication
    df['AMP_AVG'] = df[['AMP1_IR', 'AMP2_IR', 'DMP3_IR', 'DMP4_IR', 'AMP5_IR']].mean(axis=1)
    df.drop(columns=['AMP1_IR', 'AMP2_IR', 'DMP3_IR', 'DMP4_IR', 'AMP5_IR'], inplace=True)

    # Create a binary flag for observations in which methane concentration exceeds the alert_rate: 1 means observation above alert_rate
    df["ALERT"] = df[["MM256", "MM263", "MM264"]].ge(alert_rate).any(axis=1).astype(int)

    # Cast floating values as float32 to save memory and speed-up computation
    df = cast_float_columns_to_float32(df)

    # Split dataframe into train and test datasets - not separating targets from features for now (model-dependent need)
    train_data, test_data = train_test_split(df, test_size=test_size, shuffle=False)

    # Scale numerical features without data leakage
    features_to_scale = train_data.select_dtypes(include='floating')
    scalers = {}
    for feature in features_to_scale :
        mm_scaler = MinMaxScaler()
        train_data[feature] = mm_scaler.fit_transform(train_data[[feature]]).astype(np.float32)
        test_data[feature] =  mm_scaler.transform(test_data[[feature]]).astype(np.float32)
        scalers[feature] = mm_scaler

    return train_data, test_data, scalers


def preprocess_max(df: pd.DataFrame, test_size: float =0.3, alert_rate : float =1.0) -> tuple:
    """ Preprocess function --> datetime / max reading of MM256, MM263 MM264 / train_test split / scaled values.
        Takes as input a DataFrame and a test_size (defaulting to 0.3 when no test_size argument is passed). Uses NumPy to speed-up computations.
        Aggregates the readings of the key Sensors in one synthetic measurement.
        - For Methanometers, the synthetic indicator is the highest reading of the 3 critical methanometers (MM256, MM263, MM264).
        - for other instruments (anemometers, barometers, air humidity captors, thermometers), the synthetic indicator is the average reading of the instruments.
        Applies a MinMax scaler to each feature column and stores the corresponding scaler in a dictionary of scalers. Needed to be able to inverse_transform each column with its own scaler later on.
        The function returns a tuple containing two scaled dateasets (train_data and test_data) AND the dictionary of scalers."""

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

    # Cast floating values as float32 to save memory and speed-up computation
    df = cast_float_columns_to_float32(df)

    # Create a binary flag for observations in which methane concentration exceeds the alert_rate: 1 means observation above alert_rate
    df["ALERT"] = df['MM_RATE'].ge(alert_rate).astype(int)

    # Split dataframe into train and test datasets - not separating targets from features for now (model-dependent need)
    train_data, test_data = train_test_split(df, test_size=test_size, shuffle=False)

    # Scale numerical features without data leakage
    features_to_scale = train_data.select_dtypes(include='floating')
    scalers = {}
    for feature in features_to_scale :
        mm_scaler = MinMaxScaler()
        train_data[feature] = mm_scaler.fit_transform(train_data[[feature]]).astype(np.float32)
        test_data[feature] =  mm_scaler.transform(test_data[[feature]]).astype(np.float32)
        scalers[feature] = mm_scaler

    return train_data, test_data, scalers


def slice_arrays(df: pd.DataFrame, start_index: int = 0, stop_index: int | None = None, window_length_in_sec: int =300, forecast_horizon_in_sec: int =120) -> tuple :
    ''' creates a subset of the input dataframe - the observations between start_index and stop_index (excluded),
        prepares dataframe slices containing observations exceeding the alert_rate passed to function prerocess_split,
        the shape of the slices is governed by arguments window_length_in_sec and forecast_horizon_in_sec,
        returns a tuple of tensors '''

    input_length_in_sec = window_length_in_sec - forecast_horizon_in_sec
    if input_length_in_sec <= 0:
        raise ValueError("window_length_in_sec must be > forecast_horizon_in_sec")

    # Creates subset to limit training time initially
    if stop_index is None:
        stop_index = len(df)
    subset_df = df.iloc[start_index:stop_index]

    # Define columns to be included in X and y - eliminating columns not required for modelling
    excluded_cols = ['datetime', 'slice_id', 'trigger_time', 't_rel_s', 'ALERT']
    feature_cols = [c for c in subset_df.columns if c not in excluded_cols]
    target_cols = ['MM_RATE'] if 'MM_RATE' in subset_df.columns else ['MM256', 'MM263', 'MM264']

    # Pull the sequences into NumPy arrays and build windows from positional indices only.
    feature_values = subset_df.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)
    target_values = subset_df.loc[:, target_cols].to_numpy(dtype=np.float32, copy=False)
    alert_values = subset_df['ALERT'].to_numpy(copy=False)
    del subset_df

    X_array, y_array = build_sequence_arrays(
        feature_values=feature_values,
        target_values=target_values,
        alert_values=alert_values,
        input_steps=input_length_in_sec,
        forecast_horizon_in_sec=forecast_horizon_in_sec,
        step_size_in_sec=1,
    )

    del feature_values
    del target_values
    del alert_values
    gc.collect()

    return X_array, y_array



def preprocess_c22(df: pd.DataFrame, test_size: float =0.3, alert_rate : float =1.0, window_length_in_sec: int =300, forecast_horizon_in_sec: int =120, step_size_in_sec: int =10) -> tuple:
    """ Preprocess function --> synthetic sensor features / train_test split / scaled values.
        Takes as input a DataFrame and a test_size (defaulting to 0.3 when no test_size argument is passed). Uses NumPy to speed-up computations.
        Aggregates the readings of the key Sensors in one synthetic measurement.
        - For Methanometers, the synthetic indicator is the highest reading of the 3 critical methanometers (MM256, MM263, MM264).
        - for other instruments (anemometers, barometers, air humidity captors, thermometers), the synthetic indicator is the average reading of the instruments.
        WORK IN PROGRESS - TO BE UPDATED"""

    input_length_in_sec = window_length_in_sec - forecast_horizon_in_sec
    if input_length_in_sec <= 0:
        raise ValueError("window_length_in_sec must be > forecast_horizon_in_sec")

    if step_size_in_sec <= 0:
        raise ValueError("step_size_in_sec must be > 0")

    # Create a copy of the initial DataFrame before editing it
    df = df.copy()

    # Row order is the implicit time axis, so we can drop the calendar columns
    # instead of creating a DatetimeIndex just for sequence construction.
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

    # Cast floating values as float32 to save memory and speed-up computation
    df = cast_float_columns_to_float32(df)

    # Create a binary flag for observations in which methane concentration exceeds the alert_rate: 1 means observation above alert_rate
    df["ALERT"] = df['MM_RATE'].ge(alert_rate).astype(int)

    # Define columns to be included in X and y - eliminating columns not required for modelling
    excluded_cols = ['datetime', 'slice_id', 'trigger_time', 't_rel_s', 'ALERT']
    feature_cols = [c for c in df.columns if c not in excluded_cols]
    target_cols = ['MM_RATE']

    # Pull the sequences directly into NumPy arrays
    input_steps = int(np.ceil(input_length_in_sec / step_size_in_sec))
    feature_values = df.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)
    target_values = df.loc[:, target_cols].to_numpy(dtype=np.float32, copy=False)
    alert_values = df['ALERT'].to_numpy(copy=False)

    # Delete computation results we no longer need (free up memory)
    X_array, y_array = build_sequence_arrays(
        feature_values=feature_values,
        target_values=target_values,
        alert_values=alert_values,
        input_steps=input_steps,
        forecast_horizon_in_sec=forecast_horizon_in_sec,
        step_size_in_sec=step_size_in_sec,
    )

    del feature_values
    del target_values
    del alert_values
    del df
    gc.collect()

    # Split X_array and y_array into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=test_size, shuffle=False)

    # Imports to add engineered features with aeon.catch22 (may require running previously: %pip install -U aeon)
    from aeon.transformations.collection.feature_based import Catch22

    # Create catch22 features
    c22_mv = Catch22(replace_nans=True)
    X_train_transformed = np.asarray(c22_mv.fit_transform(X_train), dtype=np.float32)
    X_test_transformed = np.asarray(c22_mv.transform(X_test), dtype=np.float32)

    # Scale transformed features without data leakage.
    scalers = {}
    X_train_scaled = np.empty_like(X_train_transformed, dtype=np.float32)
    X_test_scaled = np.empty_like(X_test_transformed, dtype=np.float32)

    for feature_idx in range(X_train_transformed.shape[1]):
        mm_scaler = MinMaxScaler()
        X_train_scaled[:, [feature_idx]] = mm_scaler.fit_transform(
            X_train_transformed[:, [feature_idx]]
        ).astype(np.float32)
        X_test_scaled[:, [feature_idx]] = mm_scaler.transform(
            X_test_transformed[:, [feature_idx]]
        ).astype(np.float32)
        scalers[feature_idx] = mm_scaler

    return (
        X_train_scaled,
        y_train.astype(np.float32, copy=False),
        X_test_scaled,
        y_test.astype(np.float32, copy=False),
        scalers,
    )


# We initially thought this would be useful to train the model - EDA helped us realize this wasn't a good option
# We have kept the code for now - will probably get rid of it eventually
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

# We initially thought this would be useful to train the model - EDA helped us realize this wasn't a good option
# We have kept the code for now - will probably get rid of it eventually
def feature_target(df: pd.DataFrame, target: str) -> tuple:
    """ within a DataFrame, isolates target variable from features
        takes as input a dataframe and a string corresponding to the column representing the target"""
    y = df[target]
    X = df.drop(columns=[target])
    return X, y
