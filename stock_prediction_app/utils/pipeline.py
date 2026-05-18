from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.settings import FEATURE_COLUMNS, LSTM_CONFIG, RANDOM_STATE, TICKER_COMPANY_NAMES, TRAIN_RATIO, XGB_CONFIG
from utils.data_loader import download_stock_data
from utils.evaluation import build_evaluation_table, calculate_metrics
from utils.feature_engineering import add_technical_indicators, build_modeling_frame
from utils.reproducibility import set_global_seed

set_global_seed(RANDOM_STATE)

from utils.model_lstm import create_lstm_supervised_datasets, forecast_lstm, train_lstm_model
from utils.model_xgboost import forecast_hybrid_stacking, forecast_xgboost, train_xgboost_model
from utils.preprocessing import clean_stock_data, scale_feature_sets, split_time_series


def run_research_pipeline(
    ticker: str,
    period: str,
    sequence_length: int,
    lstm_epochs: int,
    future_steps: int,
    tune_xgb: bool = False,
) -> dict:
    """
    Execute the end-to-end research pipeline.

    # bagian ini sesuai metodologi penelitian skripsi
    """
    set_global_seed(RANDOM_STATE)

    raw_df = download_stock_data(ticker=ticker, period=period)
    clean_df = clean_stock_data(raw_df)
    min_required_rows = max(sequence_length + 30, 80)
    if len(clean_df) < min_required_rows:
        raise ValueError(
            f"Data {ticker} hanya memiliki {len(clean_df)} observasi bersih. "
            f"Butuh minimal sekitar {min_required_rows} observasi untuk indikator teknikal, split train-test, dan window LSTM."
        )

    overview_df = add_technical_indicators(clean_df)
    modeling_df = build_modeling_frame(clean_df)
    if len(modeling_df) < max(sequence_length + 20, 50):
        raise ValueError(
            f"Data {ticker} terlalu sedikit setelah indikator teknikal dibuat. "
            "Pilih periode yang lebih panjang atau ticker lain."
        )

    train_df, test_df = split_time_series(modeling_df, train_ratio=TRAIN_RATIO)
    if len(train_df) < sequence_length or len(test_df) < 2:
        raise ValueError(
            "Train-test split tidak cukup untuk melatih dan mengevaluasi model. "
            "Kurangi Sliding Window LSTM atau pilih periode data lebih panjang."
        )

    x_train_scaled, x_test_scaled, feature_scaler = scale_feature_sets(
        train_df,
        test_df,
        FEATURE_COLUMNS,
    )
    scaling_preview_before = train_df[FEATURE_COLUMNS].head(5).reset_index(drop=True)
    scaling_preview_after = pd.DataFrame(
        x_train_scaled[:5],
        columns=FEATURE_COLUMNS,
    )

    xgb_model = train_xgboost_model(x_train_scaled, train_df["Target"], XGB_CONFIG, tune=tune_xgb)
    xgb_predictions = xgb_model.predict(x_test_scaled)
    xgb_metrics = calculate_metrics(test_df["Target"], xgb_predictions)
    feature_importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": xgb_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False).reset_index(drop=True)
    xgb_prediction_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(test_df["Target_Date"]),
            "Actual": test_df["Target"].values,
            "Prediction": xgb_predictions,
            "Model": "XGBoost",
        }
    )

    close_scaler = MinMaxScaler()
    train_close_scaled = close_scaler.fit_transform(train_df[["Close"]].to_numpy())
    test_close_scaled = close_scaler.transform(test_df[["Close"]].to_numpy())
    train_target_scaled = close_scaler.transform(train_df[["Target"]].to_numpy())
    test_target_scaled = close_scaler.transform(test_df[["Target"]].to_numpy())
    x_train_lstm, y_train_lstm, x_test_lstm, y_test_lstm, lstm_train_positions = create_lstm_supervised_datasets(
        train_close_scaled,
        test_close_scaled,
        train_target_scaled,
        test_target_scaled,
        sequence_length,
    )
    lstm_model, lstm_history = train_lstm_model(
        x_train=x_train_lstm,
        y_train=y_train_lstm,
        sequence_length=sequence_length,
        epochs=lstm_epochs,
        units=LSTM_CONFIG["units"],
        dropout=LSTM_CONFIG["dropout"],
        batch_size=LSTM_CONFIG["batch_size"],
        learning_rate=LSTM_CONFIG["learning_rate"],
        verbose=LSTM_CONFIG["verbose"],
    )
    lstm_train_predictions_scaled = lstm_model.predict(x_train_lstm, verbose=0)
    lstm_predictions_scaled = lstm_model.predict(x_test_lstm, verbose=0)
    lstm_train_predictions = close_scaler.inverse_transform(lstm_train_predictions_scaled).flatten()
    lstm_predictions = close_scaler.inverse_transform(lstm_predictions_scaled).flatten()
    y_test_actual = test_df["Target"].to_numpy()

    # Hybrid Stacking LSTM-XGBoost
    # 1. LSTM bertindak sebagai base learner/feature extractor dan menghasilkan
    #    prediksi harga target untuk baris train dan test yang sudah aligned.
    # 2. Prediksi LSTM skala asli digabungkan sebagai fitur tambahan bersama
    #    fitur teknikal/asli yang telah di-scale dari data training saja.
    # 3. XGBoost meta-learner belajar dari gabungan fitur tersebut untuk
    #    menghasilkan prediksi akhir hybrid pada test set.
    x_train_meta_base = x_train_scaled[lstm_train_positions]
    y_train_meta = train_df["Target"].iloc[lstm_train_positions]
    X_train_meta = np.column_stack([x_train_meta_base, lstm_train_predictions])
    X_test_meta = np.column_stack([x_test_scaled, lstm_predictions])

    hybrid_model = train_xgboost_model(X_train_meta, y_train_meta, XGB_CONFIG, tune=tune_xgb)
    hybrid_predictions = hybrid_model.predict(X_test_meta)
    hybrid_metrics = calculate_metrics(test_df["Target"], hybrid_predictions)
    hybrid_feature_importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS + ["LSTM_Prediction"],
            "Importance": hybrid_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    hybrid_prediction_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(test_df["Target_Date"]),
            "Actual": test_df["Target"].values,
            "Prediction": hybrid_predictions,
            "Model": "Hybrid Stacking LSTM-XGBoost",
        }
    )
    hybrid_meta_preview = test_df[
        ["Open", "High", "RSI_14", "BB_Upper", "Target"]
    ].copy()
    hybrid_meta_preview.insert(0, "LSTM_Prediction", lstm_predictions)
    hybrid_meta_preview = hybrid_meta_preview.head(10).reset_index(drop=True)

    lstm_metrics = calculate_metrics(y_test_actual, lstm_predictions)
    lstm_prediction_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(test_df["Target_Date"]).values,
            "Actual": y_test_actual,
            "Prediction": lstm_predictions,
            "Model": "LSTM",
        }
    )

    metrics_table = build_evaluation_table(
        {
            "LSTM": lstm_metrics,
            "XGBoost": xgb_metrics,
            "Hybrid Stacking LSTM-XGBoost": hybrid_metrics,
        }
    )

    future_dates = pd.bdate_range(clean_df["Date"].iloc[-1] + pd.tseries.offsets.BDay(1), periods=future_steps)
    lstm_future = forecast_lstm(
        model=lstm_model,
        scaler=close_scaler,
        close_values=clean_df["Close"].to_numpy(),
        sequence_length=sequence_length,
        future_steps=future_steps,
    )
    xgb_future = forecast_xgboost(
        model=xgb_model,
        feature_scaler=feature_scaler,
        history_df=clean_df,
        feature_columns=FEATURE_COLUMNS,
        feature_builder=add_technical_indicators,
        future_steps=future_steps,
    )
    hybrid_future = forecast_hybrid_stacking(
        hybrid_model=hybrid_model,
        lstm_model=lstm_model,
        close_scaler=close_scaler,
        feature_scaler=feature_scaler,
        history_df=clean_df,
        feature_columns=FEATURE_COLUMNS,
        feature_builder=add_technical_indicators,
        sequence_length=sequence_length,
        future_steps=future_steps,
    )
    simulation_df = pd.DataFrame(
        {
            "Date": future_dates,
            "LSTM": lstm_future,
            "XGBoost": xgb_future,
            "Hybrid Stacking LSTM-XGBoost": hybrid_future,
        }
    )

    best_rmse_model = metrics_table.sort_values("RMSE").iloc[0]["Model"]
    best_mae_model = metrics_table.sort_values("MAE").iloc[0]["Model"]
    split_info = {
        "train_start": pd.to_datetime(train_df["Date"].iloc[0]),
        "train_end": pd.to_datetime(train_df["Date"].iloc[-1]),
        "test_start": pd.to_datetime(test_df["Date"].iloc[0]),
        "test_end": pd.to_datetime(test_df["Date"].iloc[-1]),
        "split_date": pd.to_datetime(test_df["Date"].iloc[0]),
        "train_count": int(len(train_df)),
        "test_count": int(len(test_df)),
        "train_ratio": float(TRAIN_RATIO),
    }
    sequence_info = {
        "sequence_length": int(sequence_length),
        "x_train_shape": tuple(x_train_lstm.shape),
        "y_train_shape": tuple(y_train_lstm.shape),
        "x_test_shape": tuple(x_test_lstm.shape),
        "y_test_shape": tuple(y_test_lstm.shape),
    }

    summary = {
        "observation_count": int(len(clean_df)),
        "latest_close": float(clean_df["Close"].iloc[-1]),
        "daily_change_pct": float(clean_df["Close"].pct_change().iloc[-1] * 100),
        "company_name": TICKER_COMPANY_NAMES.get(ticker, ticker),
        "best_rmse_model": best_rmse_model,
        "best_mae_model": best_mae_model,
    }
    correlation_matrix = modeling_df[FEATURE_COLUMNS + ["Target"]].corr(numeric_only=True)

    return {
        "params": {
            "ticker": ticker,
            "period": period,
            "sequence_length": sequence_length,
            "lstm_epochs": lstm_epochs,
            "future_steps": future_steps,
        },
        "raw_data": clean_df,
        "feature_data": overview_df,
        "processed_data": modeling_df,
        "lstm_prediction": lstm_prediction_frame,
        "xgb_prediction": xgb_prediction_frame,
        "hybrid_prediction": hybrid_prediction_frame,
        "evaluation_table": metrics_table,
        "simulation": simulation_df,
        "feature_importance": feature_importance_df,
        "hybrid_feature_importance": hybrid_feature_importance_df,
        "hybrid_meta_preview": hybrid_meta_preview,
        "correlation_matrix": correlation_matrix,
        "split_info": split_info,
        "scaling_preview_before": scaling_preview_before,
        "scaling_preview_after": scaling_preview_after,
        "sequence_info": sequence_info,
        "summary": summary,
        "lstm_metrics": lstm_metrics,
        "xgb_metrics": xgb_metrics,
        "hybrid_metrics": hybrid_metrics,
        "lstm_history": pd.DataFrame(lstm_history.history),
    }
