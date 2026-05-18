from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


def train_xgboost_model(
    x_train: np.ndarray,
    y_train: pd.Series,
    params: dict,
    tune: bool = False,
) -> XGBRegressor:
    """Train an XGBoost regressor with optional lightweight grid search."""
    if XGBRegressor is None:
        raise ValueError(
            "Dependency `xgboost` belum terpasang. Instal dependency dari requirements.txt "
            "sebelum menjalankan model XGBoost dan Hybrid Stacking."
        )

    if tune:
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 4],
            "learning_rate": [0.03, 0.05],
        }
        base_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=params.get("random_state", 42),
            seed=params.get("seed", 42),
            n_jobs=1,
        )
        cv_splits = min(3, max(2, len(x_train) // 30))
        search = GridSearchCV(
            estimator=base_model,
            param_grid=grid,
            scoring="neg_root_mean_squared_error",
            cv=TimeSeriesSplit(n_splits=cv_splits),
            n_jobs=1,
            verbose=0,
        )
        search.fit(x_train, y_train)
        return search.best_estimator_

    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    return model


def forecast_xgboost(
    model: XGBRegressor,
    feature_scaler,
    history_df: pd.DataFrame,
    feature_columns: list[str],
    feature_builder,
    future_steps: int,
) -> np.ndarray:
    """
    Forecast future values recursively for simulation only.

    # bagian ini sesuai metodologi penelitian skripsi
    """
    simulated_history = history_df.copy().reset_index(drop=True)
    predictions = []

    for _ in range(future_steps):
        feature_ready = feature_builder(simulated_history)
        latest_row = feature_ready.iloc[[-1]]
        scaled_feature = feature_scaler.transform(latest_row[feature_columns])
        next_close = model.predict(scaled_feature)[0]
        predictions.append(float(next_close))

        next_date = simulated_history["Date"].iloc[-1] + pd.tseries.offsets.BDay(1)
        next_row = {
            "Date": next_date,
            "Open": next_close,
            "High": next_close,
            "Low": next_close,
            "Close": next_close,
            "Volume": simulated_history["Volume"].iloc[-1],
        }
        simulated_history = pd.concat([simulated_history, pd.DataFrame([next_row])], ignore_index=True)

    return np.array(predictions)


def forecast_hybrid_stacking(
    hybrid_model: XGBRegressor,
    lstm_model,
    close_scaler,
    feature_scaler,
    history_df: pd.DataFrame,
    feature_columns: list[str],
    feature_builder,
    sequence_length: int,
    future_steps: int,
) -> np.ndarray:
    """
    Forecast future values with the same hybrid stacking idea used in testing.

    LSTM first produces a next-step price estimate. That estimate is appended as
    LSTM_Prediction to the scaled technical/original features, then the XGBoost
    meta-learner produces the final hybrid forecast.
    """
    simulated_history = history_df.copy().reset_index(drop=True)
    predictions = []

    for _ in range(future_steps):
        feature_ready = feature_builder(simulated_history)
        latest_row = feature_ready.iloc[[-1]]
        scaled_feature = feature_scaler.transform(latest_row[feature_columns])

        close_window = simulated_history["Close"].to_numpy()[-sequence_length:]
        scaled_window = close_scaler.transform(close_window.reshape(-1, 1))
        lstm_next_scaled = lstm_model.predict(scaled_window.reshape(1, sequence_length, 1), verbose=0)
        lstm_next = close_scaler.inverse_transform(lstm_next_scaled).flatten()[0]

        meta_features = np.column_stack([scaled_feature, [lstm_next]])
        next_close = float(hybrid_model.predict(meta_features)[0])
        predictions.append(next_close)

        next_date = simulated_history["Date"].iloc[-1] + pd.tseries.offsets.BDay(1)
        next_row = {
            "Date": next_date,
            "Open": next_close,
            "High": next_close,
            "Low": next_close,
            "Close": next_close,
            "Volume": simulated_history["Volume"].iloc[-1],
        }
        simulated_history = pd.concat([simulated_history, pd.DataFrame([next_row])], ignore_index=True)

    return np.array(predictions)
