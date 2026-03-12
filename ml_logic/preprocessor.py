# Data pre-processing

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# from sklearn.pipeline import make_pipeline
# from sklearn.compose import ColumnTransformer, make_column_transformer
# from sklearn.preprocessing import FunctionTransformer


def load_data_local() -> pd.DataFrame:
    """loads the dataset from local repertory raw_data into a DataFrame"""
    return pd.read_csv('raw_data/methane_data.csv')


def preprocess_split(df: pd.DataFrame, test_size: float =0.3, alert_rate : float =1.0) -> tuple:
    """ Preprocess function --> datetime / a single current intensity feature / train_test split / scaled values
        takes as input a DataFrame and a test_size (defaulting to 0.3 when no test_size argument is passed)
        returns a tuple containing two scaled dateasets : train_data and test_data """

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
    features_to_scale = ['AN311', 'AN422', 'AN423', 'TP1721', 'RH1722', 'BA1723', 'TP1711', 'RH1712', 'BA1713', 'MM252', 'MM261', 'MM262', 'MM263',\
                     'MM264', 'MM256', 'MM211', 'CM861','CR863', 'P_864', 'TC862', 'WM868', 'AMP_AVG', 'F_SIDE', 'V']
    mm_scaler = MinMaxScaler()
    for feature in features_to_scale :
        train_data[feature] = mm_scaler.fit_transform(train_data[[feature]])
        test_data[feature] =  mm_scaler.transform(test_data[[feature]])

    return train_data, test_data

# TO BE DONE: add new features (lagged variables for models other than SARIMA)

def slice_arrays(df: pd.DataFrame, start_time, stop_time, window_length_in_sec: int =360, forecast_horizon_in_sec: int =180) -> tuple :
    ''' creates a subset of the input dataframe - the observations between start_time and stop_time (excluded),
        prepares dataframe slices containing observations exceeding the alert_rate passed to function prerocess_split,
        the shape of the slices is governed by arguments window_length_in_sec and forecast_horizon_in_sec,
        returns a tuple of tensors '''

    input_length_in_sec = window_length_in_sec - forecast_horizon_in_sec
    if input_length_in_sec <= 0:
        raise ValueError("window_length_in_sec must be > forecast_horizon_in_sec")

    # Creates subset to limit training time initially
    subset_df = df.iloc[start_time:stop_time]

    # Define columns to be included in X and y - eliminating columns not required for modelling
    excluded_cols = ['datetime', 'slice_id', 'trigger_time', 't_rel_s', 'ALERT']
    feature_cols = [c for c in subset_df.columns if c not in excluded_cols]
    target_cols = ["MM256", "MM263", "MM264"]

    # Use boolean mask to identify index positions in which methane concentration exceeds the alert_rate
    trigger_mask = (subset_df['ALERT'] == 1)
    trigger_times = subset_df.index[trigger_mask]

    if trigger_times.empty:
        X_empty = np.empty((0, input_length_in_sec + 1, len(feature_cols)), dtype=np.float32)
        y_empty = np.empty((0, forecast_horizon_in_sec, len(target_cols)), dtype=np.float32)
        return X_empty, y_empty

    # Create a DataFrame of slices with length window_length_in_second and forecast_horizon_in sec after trigger time
    # the DataFrame includes columns indicating slice_id, trigger_time on which the window is 'focused', and time relative of the observation vs the trigger
    slices = []

    for i, t0 in enumerate(trigger_times, start=1):
        start = t0 - pd.Timedelta(seconds=input_length_in_sec)
        end = start + pd.Timedelta(seconds=window_length_in_sec + 1)
        w = subset_df[(subset_df.index >= start) & (subset_df.index < end)].copy()
        w['slice_id'] = i
        w['trigger_time'] = t0
        w['t_rel_s'] = (w.index - t0).total_seconds().astype(int)
        slices.append(w)
    data_slices = pd.concat(slices).reset_index(names='datetime')

    # From data_slices, generate arrays for an LSTM model
    wip_df = data_slices.sort_values(['slice_id', 'datetime']).copy()
    X_df = wip_df[(wip_df['t_rel_s'] >= -input_length_in_sec) & (wip_df['t_rel_s'] <= 0)]
    y_df = wip_df[(wip_df['t_rel_s'] >= 1) & (wip_df['t_rel_s'] <= forecast_horizon_in_sec)]

    # Only select full-length slices
    good_X = X_df.groupby('slice_id').size()
    good_y = y_df.groupby('slice_id').size()
    good_ids = good_X[good_X == input_length_in_sec + 1].index.intersection(
        good_y[good_y == forecast_horizon_in_sec].index)

    if len(good_ids) == 0:
        X_empty = np.empty((0, input_length_in_sec +1, len(feature_cols)), dtype=np.float32)
        y_empty = np.empty((0, forecast_horizon_in_sec, len(target_cols)), dtype=np.float32)
        return X_empty, y_empty

    # Stack arrays to create tensors
    X, y = [], []
    for sid in good_ids:
        X_seq = X_df.loc[(X_df['slice_id'] == sid), feature_cols].to_numpy(dtype=np.float32)
        y_seq = y_df.loc[(y_df['slice_id'] == sid), target_cols].to_numpy(dtype=np.float32)
        X.append(X_seq)
        y.append(y_seq)

    return np.stack(X), np.stack(y)


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

if __name__ == "__main__":
    df_result = preprocess_split(df)
    df_result.to_csv('raw_data/methane_data_processed.csv', index=False)
    print("Fichier sauvegardé !")
