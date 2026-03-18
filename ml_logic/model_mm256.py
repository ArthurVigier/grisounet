"""Single-sensor LSTM models for MM256 methane forecasting."""

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="grisounet")
class PinballLoss(tf.keras.losses.Loss):
    """Serializable pinball loss used by the MM256 models."""

    def __init__(
        self,
        quantile: float = 0.8,
        name: str = "pinball_loss",
        reduction: str = "sum_over_batch_size",
    ):
        super().__init__(name=name, reduction=reduction)
        self.quantile = float(quantile)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(
            tf.maximum(self.quantile * error, (self.quantile - 1.0) * error)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"quantile": self.quantile})
        return config


def pinball_loss(y_true, y_pred, quantile=0.8):
    """Backward-compatible functional wrapper around the serializable loss."""
    return PinballLoss(quantile=quantile)(y_true, y_pred)


def build_simple_lstm_mm256(
    input_length: int,
    n_features: int,
    forecast_horizon: int,
    n_targets: int = 1,
    units: int = 64,
    quantile: float = 0.8,
    n_static_features: int = 0,
):
    """Build the simple encoder-decoder LSTM used as the model baseline."""
    from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, RepeatVector, TimeDistributed
    from tensorflow.keras.models import Model

    sequence_input = Input(shape=(input_length, n_features), name="sequence_input")
    encoded = LSTM(units, return_sequences=False)(sequence_input)

    model_inputs = [sequence_input]
    context = encoded
    if n_static_features > 0:
        catch22_input = Input(shape=(n_static_features,), name="catch22_input")
        catch22_context = Dense(max(16, units // 2), activation="relu")(catch22_input)
        context = Concatenate()([encoded, catch22_context])
        model_inputs.append(catch22_input)

    repeated = RepeatVector(forecast_horizon)(context)
    decoded = LSTM(units, return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(n_targets))(decoded)

    model = Model(inputs=model_inputs if len(model_inputs) > 1 else sequence_input, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss=PinballLoss(quantile=quantile),
    )
    return model


def build_advanced_lstm_mm256(
    input_length: int,
    n_features: int,
    forecast_horizon: int,
    n_targets: int = 1,
    units: int = 64,
    quantile: float = 0.8,
    n_static_features: int = 0,
):
    """Build the deeper encoder-decoder LSTM used for the MM256 candidate model."""
    from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, RepeatVector, TimeDistributed
    from tensorflow.keras.models import Model

    sequence_input = Input(shape=(input_length, n_features), name="sequence_input")
    encoded = LSTM(units, return_sequences=True)(sequence_input)
    encoded = LSTM(units, return_sequences=False)(encoded)

    model_inputs = [sequence_input]
    context = encoded
    if n_static_features > 0:
        catch22_input = Input(shape=(n_static_features,), name="catch22_input")
        catch22_context = Dense(max(16, units // 2), activation="relu")(catch22_input)
        context = Concatenate()([encoded, catch22_context])
        model_inputs.append(catch22_input)

    repeated = RepeatVector(forecast_horizon)(context)
    decoded = LSTM(units, return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(n_targets))(decoded)

    model = Model(inputs=model_inputs if len(model_inputs) > 1 else sequence_input, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss=PinballLoss(quantile=quantile),
    )
    return model


def build_mm256_model(
    variant: str,
    input_length: int,
    n_features: int,
    forecast_horizon: int,
    n_targets: int = 1,
    units: int = 64,
    quantile: float = 0.8,
    n_static_features: int = 0,
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
        n_static_features=n_static_features,
    )


def _fit_mm256_model(
    model,
    X_train,
    y_train: np.ndarray,
    X_val=None,
    y_val: np.ndarray | None = None,
    epochs: int = 40,
    batch_size: int = 128,
    patience: int = 5,
):
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
    if X_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        fit_kwargs["validation_split"] = 0.2

    history = model.fit(X_train, y_train, **fit_kwargs)

    y_pred = None
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val, batch_size=batch_size)

    return history, y_pred


def simple_lstm_mm256(
    X_train,
    y_train: np.ndarray,
    X_val=None,
    y_val: np.ndarray | None = None,
    units: int = 64,
    epochs: int = 40,
    batch_size: int = 128,
    patience: int = 5,
    quantile: float = 0.8,
    n_static_features: int = 0,
) -> tuple:
    """Train the simple MM256 LSTM model."""
    model = build_simple_lstm_mm256(
        input_length=X_train.shape[1],
        n_features=X_train.shape[2],
        forecast_horizon=y_train.shape[1],
        n_targets=y_train.shape[2] if y_train.ndim == 3 else 1,
        units=units,
        quantile=quantile,
        n_static_features=n_static_features,
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
    X_train,
    y_train: np.ndarray,
    X_val=None,
    y_val: np.ndarray | None = None,
    units: int = 64,
    epochs: int = 40,
    batch_size: int = 128,
    patience: int = 5,
    quantile: float = 0.8,
    n_static_features: int = 0,
) -> tuple:
    """Train the advanced MM256 LSTM model."""
    model = build_advanced_lstm_mm256(
        input_length=X_train.shape[1],
        n_features=X_train.shape[2],
        forecast_horizon=y_train.shape[1],
        n_targets=y_train.shape[2] if y_train.ndim == 3 else 1,
        units=units,
        quantile=quantile,
        n_static_features=n_static_features,
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
