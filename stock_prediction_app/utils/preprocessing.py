from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clean_stock_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean duplicated rows, sort by date, and handle missing values."""
    df = dataframe.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date").reset_index(drop=True)
    numeric_columns = df.select_dtypes(include=["number"]).columns
    df[numeric_columns] = df[numeric_columns].ffill().bfill()
    return df


def split_time_series(dataframe: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time series data without shuffling."""
    split_index = int(len(dataframe) * train_ratio)
    train_df = dataframe.iloc[:split_index].copy()
    test_df = dataframe.iloc[split_index:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Data terlalu sedikit untuk melakukan split train-test 80:20.")
    return train_df, test_df


def scale_series(
    train_series: pd.Series,
    test_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale a single series using MinMaxScaler fitted only on training data."""
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.to_frame())
    test_scaled = scaler.transform(test_series.to_frame())
    return train_scaled, test_scaled, scaler


def scale_feature_sets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale feature matrices using training data only."""
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_df[list(feature_columns)])
    test_features = scaler.transform(test_df[list(feature_columns)])
    return train_features, test_features, scaler
