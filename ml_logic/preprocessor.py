# Data pre-processing

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# from sklearn.pipeline import make_pipeline
# from sklearn.compose import ColumnTransformer, make_column_transformer
# from sklearn.preprocessing import FunctionTransformer


def load_data_local() -> pd.DataFrame:
    """loads the dataset from local repertory raw_data into a DataFrame"""
    return pd.read_csv('raw_data/methane_data.csv')


def preprocess_split(df: pd.DataFrame, test_size: float =0.3) -> tuple:
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

def sample_datasets(df_scaled: pd.DataFrame) -> list :
    """ splits a pre-processed, scaled main dataset into 3 sample datasets focused on MM256, MM263, MM264.
        takes a PRE-PROCESSED, SCALED dataset as input.
        returns a list containing 3 sample datasets corresponding to captors MM256, MM263 and MM264 in this order."""

    # create sample scaled dataset for sensor MM256 (methanometer)
    MM256_features = ['AN422', 'TP1711', 'RH1712', 'BA1713', 'MM256', 'CM861', 'CR863', 'P_864', 'TC862', 'WM868', 'AMP_AVG', 'F_SIDE', 'V']
    MM256_df = df_scaled[MM256_features]

    # create sample scaled dataset for sensor MM263 (methanometer)
    MM263_features = ['AN422', 'TP1711', 'RH1712', 'BA1713', 'MM263', 'CM861', 'CR863', 'P_864', 'TC862', 'WM868', 'AMP_AVG', 'F_SIDE', 'V']
    MM263_df = df_scaled[MM263_features]

    # create sample scaled dataset for sensor MM264 (methanometer)
    MM264_features = ['AN422', 'TP1711', 'RH1712', 'BA1713', 'MM264', 'CM861', 'CR863', 'P_864', 'TC862', 'WM868', 'AMP_AVG', 'F_SIDE', 'V']
    MM264_df = df_scaled[MM264_features]

    return [MM256_df, MM263_df, MM264_df]


def feature_target(df: pd.DataFrame, target: str) -> tuple:
    """ within a DataFrame, isolates target variable from features """
    y = df[target]
    X = df.drop(columns=[target])
    return X, y
