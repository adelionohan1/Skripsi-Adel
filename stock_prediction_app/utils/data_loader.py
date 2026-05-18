import pandas as pd


def download_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Download historical stock data from Yahoo Finance."""
    try:
        import yfinance as yf

        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except ImportError as exc:
        raise ValueError(
            "Dependency `yfinance` belum terpasang. Instal dependency dari requirements.txt "
            "sebelum menjalankan analisis data saham."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Yahoo Finance gagal mengembalikan data untuk {ticker}. "
            "Periksa koneksi internet atau pilih ticker/periode lain."
        ) from exc

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
