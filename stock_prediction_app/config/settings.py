from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "assets"
CSS_FILE = ASSETS_DIR / "style.css"

APP_TITLE = "Stock Prediction Dashboard"
APP_SUBTITLE = "Analisis Hybrid Stacking LSTM-XGBoost untuk saham dengan volatilitas ekstrem Grup Bakrie."
APP_ICON = "📈"

DEFAULT_TICKER = "BUMI.JK"
AVAILABLE_TICKERS = ["BUMI.JK", "DEWA.JK", "BNBR.JK", "ENRG.JK", "UNSP.JK", "ELTY.JK", "VIVA.JK"]
TICKER_COMPANY_NAMES = {
    "BUMI.JK": "PT Bumi Resources Tbk",
    "DEWA.JK": "PT Darma Henwa Tbk",
    "BNBR.JK": "PT Bakrie & Brothers Tbk",
    "ENRG.JK": "PT Energi Mega Persada Tbk",
    "UNSP.JK": "PT Bakrie Sumatera Plantations Tbk",
    "ELTY.JK": "PT Bakrieland Development Tbk",
    "VIVA.JK": "PT Visi Media Asia Tbk",
}
DEFAULT_PERIOD = "5y"
AVAILABLE_PERIODS = ["1y", "2y", "5y", "10y", "max"]

TRAIN_RATIO = 0.8
TARGET_COLUMN = "Close"
RANDOM_STATE = 42

DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_LSTM_EPOCHS = 20
DEFAULT_FUTURE_STEPS = 10

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_10",
    "SMA_20",
    "RSI_14",
    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
]

LSTM_CONFIG = {
    "units": [64, 32, 16],
    "dropout": 0.2,
    "batch_size": 16,
    "learning_rate": 0.001,
    "verbose": 0,
}

XGB_CONFIG = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "seed": 42,
    "n_jobs": 1,
}
