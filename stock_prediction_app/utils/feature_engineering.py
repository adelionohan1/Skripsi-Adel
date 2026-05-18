from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create rolling technical indicators without forward-looking leakage."""
    df = dataframe.copy()

    close = df["Close"]
    rolling_mean_10 = close.rolling(window=10, min_periods=10).mean()
    rolling_mean_20 = close.rolling(window=20, min_periods=20).mean()
    rolling_std_20 = close.rolling(window=20, min_periods=20).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)

    df["SMA_10"] = rolling_mean_10
    df["SMA_20"] = rolling_mean_20
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["BB_Middle"] = rolling_mean_20
    df["BB_Upper"] = rolling_mean_20 + (2 * rolling_std_20)
    df["BB_Lower"] = rolling_mean_20 - (2 * rolling_std_20)

    return df


def build_modeling_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a supervised learning frame for next-step prediction.

    # bagian ini sesuai metodologi penelitian skripsi
    """
    df = add_technical_indicators(dataframe)
    df["Target"] = df["Close"].shift(-1)
    df["Target_Date"] = df["Date"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df
