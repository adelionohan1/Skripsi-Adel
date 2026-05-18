from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow belum terpasang. Instal dependency proyek terlebih dahulu."
    ) from exc

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def create_lstm_datasets(
    train_scaled: np.ndarray,
    test_scaled: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window datasets for LSTM.

    # bagian ini sesuai metodologi penelitian skripsi
    """
    if len(train_scaled) <= sequence_length:
        raise ValueError("Data training terlalu pendek untuk sliding window LSTM.")

    x_train, y_train = [], []
    for index in range(sequence_length, len(train_scaled)):
        x_train.append(train_scaled[index - sequence_length : index, 0])
        y_train.append(train_scaled[index, 0])

    x_test, y_test = [], []
    combined = np.vstack([train_scaled[-sequence_length:], test_scaled])
    for index in range(sequence_length, len(combined)):
        x_test.append(combined[index - sequence_length : index, 0])
        y_test.append(combined[index, 0])

    x_train = np.array(x_train).reshape(-1, sequence_length, 1)
    y_train = np.array(y_train)
    x_test = np.array(x_test).reshape(-1, sequence_length, 1)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


def create_lstm_supervised_datasets(
    train_close_scaled: np.ndarray,
    test_close_scaled: np.ndarray,
    train_target_scaled: np.ndarray,
    test_target_scaled: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build LSTM windows aligned with the supervised modeling rows.

    Each window ends at the feature date of a row, while the label is that row's
    next-day Target. Test windows may use the last training closes as historical
    context, but labels and scalers remain strictly separated.
    """
    if len(train_close_scaled) < sequence_length:
        raise ValueError("Data training terlalu pendek untuk sliding window LSTM.")

    x_train, y_train, train_row_positions = [], [], []
    for index in range(sequence_length - 1, len(train_close_scaled)):
        start_index = index - sequence_length + 1
        x_train.append(train_close_scaled[start_index : index + 1, 0])
        y_train.append(train_target_scaled[index, 0])
        train_row_positions.append(index)

    x_test, y_test = [], []
    combined_close = np.vstack([train_close_scaled[-sequence_length + 1 :], test_close_scaled])
    for test_index in range(len(test_close_scaled)):
        end_index = sequence_length - 1 + test_index
        x_test.append(combined_close[end_index - sequence_length + 1 : end_index + 1, 0])
        y_test.append(test_target_scaled[test_index, 0])

    x_train = np.array(x_train).reshape(-1, sequence_length, 1)
    y_train = np.array(y_train)
    x_test = np.array(x_test).reshape(-1, sequence_length, 1)
    y_test = np.array(y_test)
    train_row_positions = np.array(train_row_positions, dtype=int)
    return x_train, y_train, x_test, y_test, train_row_positions


def build_lstm_model(
    sequence_length: int,
    units: list[int],
    dropout: float,
    learning_rate: float,
):
    """Construct a 3-layer LSTM architecture."""
    model = Sequential(
        [
            LSTM(units[0], return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(dropout),
            LSTM(units[1], return_sequences=True),
            Dropout(dropout),
            LSTM(units[2]),
            Dropout(dropout),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_lstm_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sequence_length: int,
    epochs: int,
    units: list[int],
    dropout: float,
    batch_size: int,
    learning_rate: float,
    verbose: int = 0,
):
    """Train the LSTM model with MSE loss, early stopping, and adaptive learning rate."""
    model = build_lstm_model(sequence_length, units, dropout, learning_rate)

    checkpoint_path = MODELS_DIR / "best_lstm.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=0,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, history


def forecast_lstm(
    model,
    scaler,
    close_values: np.ndarray,
    sequence_length: int,
    future_steps: int,
) -> np.ndarray:
    """Forecast future values recursively using the last observed window."""
    scaled_history = scaler.transform(close_values.reshape(-1, 1)).flatten().tolist()
    predictions = []
    window = scaled_history[-sequence_length:]

    for _ in range(future_steps):
        window_array = np.array(window[-sequence_length:]).reshape(1, sequence_length, 1)
        next_scaled = model.predict(window_array, verbose=0)[0, 0]
        predictions.append(next_scaled)
        window.append(next_scaled)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions
