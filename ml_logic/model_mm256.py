"""Single-sensor LSTM models for MM256 methane forecasting."""

import numpy as np


def pinball_loss(y_true, y_pred, quantile=0.9):
    """Pinball (quantile) loss."""
    import tensorflow as tf

    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))


def build_simple_lstm_mm256(
    input_length: int,
    n_features: int,
    forecast_horizon: int,
    n_targets: int = 1,
    units: int = 64,
    quantile: float = 0.9,
):
    """Build the simple encoder-decoder LSTM used as the model baseline."""
    from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Input(shape=(input_length, n_features)),
        LSTM(units, return_sequences=False),
        RepeatVector(forecast_horizon),
        LSTM(units, return_sequences=True),
        TimeDistributed(Dense(n_targets)),
    ])
    model.compile(
        optimizer="adam",
        loss=lambda y_true, y_pred: pinball_loss(y_true, y_pred, quantile=quantile),
    )
    return model


def build_advanced_lstm_mm256(
    input_length: int,
    n_features: int,
    forecast_horizon: int,
    n_targets: int = 1,
    units: int = 64,
    quantile: float = 0.9,
):
    """Build the deeper encoder-decoder LSTM used for the MM256 candidate model."""
    from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Input(shape=(input_length, n_features)),
        LSTM(units, return_sequences=True),
        LSTM(units, return_sequences=False),
        RepeatVector(forecast_horizon),
        LSTM(units, return_sequences=True),
        TimeDistributed(Dense(n_targets)),
    ])
    model.compile(
        optimizer="adam",
        loss=lambda y_true, y_pred: pinball_loss(y_true, y_pred, quantile=quantile),
    )
    return model


def build_mm256_model(
    variant: str,
    input_length: int,
    n_features: int,
    forecast_horizon: int,
    n_targets: int = 1,
    units: int = 64,
    quantile: float = 0.9,
):
    """Build one of the supported MM256 architectures."""
    builders = {
        "simple": build_simple_lstm_mm256,
        "advanced": build_advanced_lstm_mm256,
    }
    if variant not in builders:
        raise ValueError(f"Unknown MM256 model variant: {variant}")

    return builders[variant](
        input_length=input_length,
        n_features=n_features,
        forecast_horizon=forecast_horizon,
        n_targets=n_targets,
        units=units,
        quantile=quantile,
    )


def _fit_mm256_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 40,
    batch_size: int = 32,
    patience: int = 5,
):
    import tensorflow as tf

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    fit_kwargs = {
        "epochs": epochs,
        "batch_size": batch_size,
        "callbacks": [early_stop],
    }
    if X_val is not None and X_val.shape[0] > 0:
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        fit_kwargs["validation_split"] = 0.2

    history = model.fit(X_train, y_train, **fit_kwargs)

    y_pred = None
    if X_val is not None and X_val.shape[0] > 0:
        y_pred = model.predict(X_val, batch_size=batch_size)

    return history, y_pred


def simple_lstm_mm256(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    units: int = 64,
    epochs: int = 40,
    batch_size: int = 32,
    patience: int = 5,
    quantile: float = 0.9,
) -> tuple:
    """Train the simple MM256 LSTM model."""
    model = build_simple_lstm_mm256(
        input_length=X_train.shape[1],
        n_features=X_train.shape[2],
        forecast_horizon=y_train.shape[1],
        n_targets=y_train.shape[2] if y_train.ndim == 3 else 1,
        units=units,
        quantile=quantile,
    )
    history, y_pred = _fit_mm256_model(
        model,
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )
    return model, history, y_pred


def advanced_lstm_mm256(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    units: int = 64,
    epochs: int = 40,
    batch_size: int = 32,
    patience: int = 5,
    quantile: float = 0.9,
) -> tuple:
    """Train the advanced MM256 LSTM model."""
    model = build_advanced_lstm_mm256(
        input_length=X_train.shape[1],
        n_features=X_train.shape[2],
        forecast_horizon=y_train.shape[1],
        n_targets=y_train.shape[2] if y_train.ndim == 3 else 1,
        units=units,
        quantile=quantile,
    )
    history, y_pred = _fit_mm256_model(
        model,
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )
    return model, history, y_pred
