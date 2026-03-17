"""
Single-sensor LSTM models for MM256 methane forecasting.

Provides a simple LSTM encoder-decoder and an advanced variant.
Both output shape (n_samples, horizon, 1) — single sensor only.

The workflow trains the simple model first.  The advanced model can be
enabled later once the simple baseline is validated.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Input,
    RepeatVector,
    TimeDistributed,
)


# ---------------------------------------------------------------------------
# Loss function — pinball loss for quantile regression
# ---------------------------------------------------------------------------
def pinball_loss(y_true, y_pred, quantile=0.9):
    """Pinball (quantile) loss.

    At quantile=0.9 the model is penalised more for under-predicting,
    which is the safe choice for methane early-warning.
    """
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))


# ---------------------------------------------------------------------------
# Simple LSTM — single encoder layer + dense decoder
# ---------------------------------------------------------------------------
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
    """Train a simple encoder-decoder LSTM for single-sensor forecasting.

    Architecture
    ------------
    Input  -> LSTM(units) -> RepeatVector(horizon) -> LSTM(units, seq) -> Dense(1)

    This is the **baseline model** — one encoder layer, one decoder layer.
    Start here; upgrade to ``advanced_lstm_mm256`` only if needed.

    Parameters
    ----------
    X_train : (n_samples, input_length, n_features)
    y_train : (n_samples, horizon, 1)
    X_val, y_val : optional validation arrays.  If None, uses
        ``validation_split=0.2`` from training data.
    units : LSTM hidden size.
    epochs, batch_size, patience : training hyperparameters.
    quantile : quantile for pinball loss.

    Returns
    -------
    (model, history, y_pred)
        y_pred is None when X_val is None or empty.
    """
    input_length = X_train.shape[1]
    n_features = X_train.shape[2]
    horizon = y_train.shape[1]

    model = Sequential([
        Input(shape=(input_length, n_features)),
        LSTM(units, return_sequences=False),
        RepeatVector(horizon),
        LSTM(units, return_sequences=True),
        TimeDistributed(Dense(1)),
    ])

    model.compile(
        optimizer="adam",
        loss=lambda yt, yp: pinball_loss(yt, yp, quantile=quantile),
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    fit_kwargs = dict(
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
    )

    if X_val is not None and X_val.shape[0] > 0:
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        fit_kwargs["validation_split"] = 0.2

    history = model.fit(X_train, y_train, **fit_kwargs)

    # Predict on validation set
    y_pred = None
    if X_val is not None and X_val.shape[0] > 0:
        y_pred = model.predict(X_val, batch_size=batch_size)
        score = model.evaluate(X_val, y_val, verbose=0)
        print(f"Simple LSTM — val loss: {score:.6f}")

    return model, history, y_pred


# ---------------------------------------------------------------------------
# Advanced LSTM — two encoder layers (deeper context encoding)
# ---------------------------------------------------------------------------
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
    """Train a deeper encoder-decoder LSTM for single-sensor forecasting.

    Architecture
    ------------
    Input -> LSTM(units, seq) -> LSTM(units) -> RepeatVector -> LSTM(units, seq) -> Dense(1)

    Two stacked encoder LSTMs capture longer-range temporal dependencies.
    Use this once the simple baseline is validated.

    Parameters & Returns: same as ``simple_lstm_mm256``.
    """
    input_length = X_train.shape[1]
    n_features = X_train.shape[2]
    horizon = y_train.shape[1]

    model = Sequential([
        Input(shape=(input_length, n_features)),
        # Encoder — two stacked LSTMs
        LSTM(units, return_sequences=True),
        LSTM(units, return_sequences=False),
        # Decoder
        RepeatVector(horizon),
        LSTM(units, return_sequences=True),
        TimeDistributed(Dense(1)),
    ])

    model.compile(
        optimizer="adam",
        loss=lambda yt, yp: pinball_loss(yt, yp, quantile=quantile),
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    fit_kwargs = dict(
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
    )

    if X_val is not None and X_val.shape[0] > 0:
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        fit_kwargs["validation_split"] = 0.2

    history = model.fit(X_train, y_train, **fit_kwargs)

    y_pred = None
    if X_val is not None and X_val.shape[0] > 0:
        y_pred = model.predict(X_val, batch_size=batch_size)
        score = model.evaluate(X_val, y_val, verbose=0)
        print(f"Advanced LSTM — val loss: {score:.6f}")

    return model, history, y_pred
