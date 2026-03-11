from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from aeon.transformations.collection.feature_based import Catch22
# Exemple simple
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer


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
    
