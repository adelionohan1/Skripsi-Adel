from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    """Calculate MAPE and return percentage."""
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    epsilon = 1e-8
    return float(np.mean(np.abs((y_true_arr - y_pred_arr) / np.maximum(np.abs(y_true_arr), epsilon))) * 100)


def directional_accuracy(y_true, y_pred) -> float:
    """Calculate directional accuracy using day-over-day direction."""
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    if len(y_true_arr) < 2 or len(y_pred_arr) < 2:
        return 0.0
    actual_direction = np.sign(np.diff(y_true_arr))
    predicted_direction = np.sign(np.diff(y_pred_arr))
    accuracy = np.mean(actual_direction == predicted_direction)
    return float(accuracy * 100)


def calculate_metrics(y_true, y_pred) -> dict:
    """Calculate MAE, RMSE, MAPE and directional accuracy on the test set."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    directional = directional_accuracy(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "MAPE": mape,
        "Directional Accuracy (%)": directional,
    }


def build_evaluation_table(metrics_dict: dict[str, dict]) -> pd.DataFrame:
    """Transform metric dictionary into a comparison table."""
    rows = []
    for model_name, metric_values in metrics_dict.items():
        row = {
            "Model": model_name,
            "MAE": metric_values["MAE"],
            "RMSE": metric_values["RMSE"],
            "MAPE": metric_values.get("MAPE", float("nan")),
            "Directional Accuracy (%)": metric_values.get("Directional Accuracy (%)", float("nan")),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
