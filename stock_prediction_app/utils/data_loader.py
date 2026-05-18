from __future__ import annotations

import time

import pandas as pd
import streamlit as st


def _normalize_downloaded_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data.empty:
        raise ValueError(
            f"Tidak ada data yang berhasil diunduh untuk {ticker}. "
            "Ticker mungkin tidak aktif di Yahoo Finance atau periode yang dipilih terlalu pendek."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Data {ticker} tidak lengkap. Kolom yang hilang: {sorted(missing_columns)}. "
            "Silakan pilih ticker lain atau periode data yang lebih panjang."
        )
    return data


@st.cache_data(ttl=1800, show_spinner=False)
def _download_stock_data_cached(ticker: str, period: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ValueError(
            "Dependency `yfinance` belum terpasang. Instal dependency dari requirements.txt "
            "sebelum menjalankan analisis data saham."
        ) from exc

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            data = yf.download(ticker, period=period, auto_adjust=True, progress=False, threads=False)
            return _normalize_downloaded_data(data, ticker)
        except Exception as exc:
            last_error = exc
            error_message = str(exc)
            is_rate_limited = "YFRateLimitError" in error_message or "Too Many Requests" in error_message
            if is_rate_limited and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            if is_rate_limited:
                raise ValueError(
                    f"Yahoo Finance sedang membatasi permintaan data untuk {ticker}. "
                    "Tunggu beberapa menit lalu coba lagi, atau gunakan hasil analisis sebelumnya yang sudah tersimpan."
                ) from exc
            raise ValueError(
                f"Yahoo Finance gagal mengembalikan data untuk {ticker}. "
                "Periksa koneksi internet atau pilih ticker/periode lain."
            ) from exc

    raise ValueError(
        f"Gagal mengunduh data untuk {ticker} setelah beberapa percobaan. "
        "Silakan coba lagi beberapa menit lagi."
    ) from last_error


def download_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Download historical stock data from Yahoo Finance with caching and retry."""
    return _download_stock_data_cached(ticker, period).copy()
