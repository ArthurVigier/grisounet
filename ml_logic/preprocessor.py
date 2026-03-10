# Data pre-processing

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.compose import ColumnTransformer, make_column_transformer
# from sklearn.preprocessing import FunctionTransformer

# Loading the dataset from local directory raw_data
df = pd.read_csv('raw_data/methane_data.csv')

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    # Create datetime indication for time, drop previous time indications
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    df.set_index('time')
    df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)

    # Replace 5 columns on current intensity in 5 motors by one average indication
    df['AMP_AVG'] = df[['AMP1_IR', 'AMP2_IR', 'DMP3_IR', 'DMP4_IR', 'AMP5_IR']].mean(axis=1)
    df.drop(columns=['AMP1_IR', 'AMP2_IR', 'DMP3_IR', 'DMP4_IR', 'AMP5_IR'], inplace=True)

    # Scale numerical features
    df_scaled = df[['time']]
    features_to_scale = ['AN311', 'AN422', 'AN423', 'TP1721', 'RH1722', 'BA1723', 'TP1711', 'RH1712', 'BA1713', 'MM252', 'MM261', 'MM262', 'MM263',\
                     'MM264', 'MM256', 'MM211', 'CM861','CR863', 'P_864', 'TC862', 'WM868', 'AMP_AVG', 'F_SIDE', 'V']
    mm_scaler = MinMaxScaler()
    for feature in features_to_scale :
        df_scaled[feature] = mm_scaler.fit_transform(df[[feature]])

    print("✅ df_scaled processed, with shape", df_scaled.shape)

    return df_scaled
