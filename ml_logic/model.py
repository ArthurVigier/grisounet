from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from aeon.transformations.collection.feature_based import Catch22
# Exemple simple
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from ml_logic.preprocessor import load_data_local , preprocess_split, sample_datasets,feature_target
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import tensorflow as tf

catch22_feature_names = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",          # parfois pnn50 dans d'autres implémentations
    "SB_BinaryStats_diff_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th001",      # ou th0_01 selon version
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "DN_OutlierInclude_abs_001_mdrmd",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "DN_OutlierInclude_abs_005_mdrmd",
    "DN_OutlierInclude_p_005_mdrmd",
    "DN_OutlierInclude_n_005_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_alpha1_exp_range",
    "FC_LocalSimple_lfit.taures",
    "Mean",
    "StandardDeviation"
    ]

def linreg(X,y):
    model = LinearRegression(tol=1)
    model = model.fit(X, y)
    print('R2: ', r2_score(y, model.predict(X)))
    return model

def catch22_features(df, target_col):
    ts = df[target_col].values.reshape(df.shape[0], -1)  # ts pour time-serie # shape (n_samples, n_timesteps)
    c22 = Catch22()
    # les noms des features
    c22 = Catch22(catch24=True)  # ou catch24=True pour 24 features
    df_transformed = c22.fit_transform(ts)  # shape (n_samples, 22) ou (n_samples, 22 * n_channels)

    # Associer noms
    feature_names = catch22_feature_names  # ou étends si multivariate
    print(feature_names)
    df_transformed = pd.DataFrame(df_transformed, columns=feature_names)
    return df_transformed

def smap_loss(y_true, y_pred):
    return 200 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def load_and_preprocess():
    df = load_data_local()
    train_df, test_df = preprocess_split(df)
    MM256_df, MM263_df, MM264_df = sample_datasets(train_df)
    return MM256_df, MM263_df, MM264_df , test_df , df

def lstm(df,target_col,lags=300, alpha=1.0, test_ratio=0.3, horizon=180):

        # ----------------------------
    # Paramètres
    # ----------------------------
    lags = lags
    alpha = alpha
    test_ratio = test_ratio
    horizon = horizon  # prédiction à t+180
    # ----------------------------
    # Préparation des données (commune aux deux modèles)
    # ----------------------------
    X_df = df.drop(columns=[target_col]).reset_index(drop=True)
    y_series = df[target_col].reset_index(drop=True).shift(-horizon)

    print("X_df shape:", X_df.shape)
    print("y_series shape:", y_series.shape)

    # ----------------------------
    # Création des lags
    # ----------------------------
    lagged_dfs = []
    for i in range(1, lags + 1):
        lag_i = X_df.shift(i).copy()
        lag_i.columns = [f"{col}_lag{i}" for col in X_df.columns]
        lagged_dfs.append(lag_i)

    X_lagged = pd.concat(lagged_dfs, axis=1)

    print("X_lagged shape:", X_lagged.shape)

    # ----------------------------
    # Assemblage et nettoyage
    # ----------------------------
    data = pd.concat([X_lagged, y_series.rename(target_col)], axis=1)

    print("data shape before cleaning:", data.shape)
    print("NaN in target:", data[target_col].isna().sum())
    print("NaN total before cleaning:", data.isna().sum().sum())

    # 1) On enlève seulement la dernière ligne sans target
    data = data.dropna(subset=[target_col]).reset_index(drop=True)

    # 2) On remplit les NaN des features
    feature_cols = [col for col in data.columns if col != target_col]
    data[feature_cols] = data[feature_cols].fillna(0)

    print("data shape after target cleaning:", data.shape)
    print("NaN total after feature fill:", data.isna().sum().sum())

    # ----------------------------
    # Matrices finales
    # ----------------------------
    X = data[feature_cols].values
    y = data[target_col].values

    print("Final X shape:", X.shape)
    print("Final y shape:", y.shape)

    # ----------------------------
    # Split temporel
    # ----------------------------
    split = int(len(X) * (1 - test_ratio))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("Train shape (Ridge/FFN):", X_train.shape, y_train.shape)
    print("Test shape (Ridge/FFN) :", X_test.shape, y_test.shape)

    # ----------------------------
    # Reshaping pour LSTM
    # ----------------------------
    # LSTM expects input in the form (samples, timesteps, features)
    # Here, timesteps = lags, features = X_df.shape[1] (original features)

    X_train_reshaped = X_train.reshape(X_train.shape[0], lags, X_df.shape[1])
    X_test_reshaped = X_test.reshape(X_test.shape[0], lags, X_df.shape[1])

    print(f"Reshaped X_train shape (LSTM): {X_train_reshaped.shape}")
    print(f"Reshaped X_test shape (LSTM): {X_test_reshaped.shape}")

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Input
    import tensorflow as tf

    def pinball_loss_keras(y_true, y_pred, quantile=0.5):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))

    model = Sequential()
    # Input layer for LSTM expects (timesteps, features)
    model.add(Input(shape=(lags, X_df.shape[1])))
    model.add(LSTM(units=64)) # Using LSTM instead of SimpleRNN
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: pinball_loss_keras(y_true, y_pred, quantile=0.5))
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2)
    y_pred = model.predict(X_test_reshaped)

    return model , y_test , y_pred
