from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from config.settings import (
    DEFAULT_FUTURE_STEPS,
    DEFAULT_LSTM_EPOCHS,
    DEFAULT_PERIOD,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TICKER,
    TICKER_COMPANY_NAMES,
)
from utils.pipeline import run_research_pipeline


PLOTLY_DARK_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "#111827",
    "font": {"color": "#E5E7EB"},
    "legend": {"font": {"color": "#E5E7EB"}},
    "margin": {"l": 28, "r": 28, "t": 56, "b": 32},
}


def build_empty_results(error_message: str | None = None, ticker: str = DEFAULT_TICKER) -> dict:
    return {
        "params": {
            "ticker": ticker,
            "period": DEFAULT_PERIOD,
            "sequence_length": DEFAULT_SEQUENCE_LENGTH,
            "lstm_epochs": DEFAULT_LSTM_EPOCHS,
            "future_steps": DEFAULT_FUTURE_STEPS,
        },
        "raw_data": pd.DataFrame(),
        "feature_data": pd.DataFrame(),
        "processed_data": pd.DataFrame(),
        "lstm_prediction": pd.DataFrame(),
        "xgb_prediction": pd.DataFrame(),
        "hybrid_prediction": pd.DataFrame(),
        "evaluation_table": pd.DataFrame(),
        "simulation": pd.DataFrame(),
        "feature_importance": pd.DataFrame(),
        "hybrid_feature_importance": pd.DataFrame(),
        "hybrid_meta_preview": pd.DataFrame(),
        "correlation_matrix": pd.DataFrame(),
        "split_info": {},
        "scaling_preview_before": pd.DataFrame(),
        "scaling_preview_after": pd.DataFrame(),
        "sequence_info": {},
        "summary": {
            "observation_count": 0,
            "latest_close": 0.0,
            "daily_change_pct": 0.0,
            "company_name": TICKER_COMPANY_NAMES.get(ticker, ticker),
            "best_rmse_model": "-",
            "best_mae_model": "-",
        },
        "lstm_metrics": {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "Directional Accuracy (%)": 0.0},
        "xgb_metrics": {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "Directional Accuracy (%)": 0.0},
        "hybrid_metrics": {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "Directional Accuracy (%)": 0.0},
        "lstm_history": pd.DataFrame(),
        "error": error_message,
    }


def get_results() -> dict:
    if "research_results" not in st.session_state:
        with st.spinner("Memproses data..."):
            try:
                st.session_state["research_results"] = run_research_pipeline(
                    ticker=DEFAULT_TICKER,
                    period=DEFAULT_PERIOD,
                    sequence_length=DEFAULT_SEQUENCE_LENGTH,
                    lstm_epochs=DEFAULT_LSTM_EPOCHS,
                    future_steps=DEFAULT_FUTURE_STEPS,
                )
            except Exception as exc:
                st.session_state["research_results"] = build_empty_results(str(exc), DEFAULT_TICKER)
    return st.session_state["research_results"]


def has_data(data) -> bool:
    if data is None:
        return False
    if isinstance(data, pd.DataFrame):
        return not data.empty
    return True


def dataframe_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def make_download_filename(prefix: str, ticker: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = ticker.replace(".", "_")
    return f"{prefix}_{safe_ticker}_{timestamp}.csv"


def render_metric_card(column, label: str, value: str, delta: str | None = None, tone: str = "neutral") -> None:
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    with column:
        st.markdown(
            f"""
            <div class="card metric-shell metric-{tone}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {delta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_empty_state(message: str = "Data tidak tersedia") -> None:
    st.warning(message)
