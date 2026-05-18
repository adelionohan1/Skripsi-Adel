from pathlib import Path

import streamlit as st

from config.settings import (
    APP_ICON,
    APP_TITLE,
    AVAILABLE_PERIODS,
    AVAILABLE_TICKERS,
    CSS_FILE,
    DEFAULT_FUTURE_STEPS,
    DEFAULT_LSTM_EPOCHS,
    DEFAULT_PERIOD,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TICKER,
    TICKER_COMPANY_NAMES,
)
from utils.pipeline import run_research_pipeline
from utils.ui import build_empty_results


st.set_page_config(
    page_title="Home",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_local_css(css_path: Path) -> None:
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def initialize_session_state() -> None:
    st.session_state.setdefault("ticker_input", DEFAULT_TICKER)
    if st.session_state["ticker_input"] not in AVAILABLE_TICKERS:
        st.session_state["ticker_input"] = DEFAULT_TICKER
    st.session_state.setdefault("period_input", DEFAULT_PERIOD)
    st.session_state.setdefault("sequence_length_input", DEFAULT_SEQUENCE_LENGTH)
    st.session_state.setdefault("lstm_epochs_input", DEFAULT_LSTM_EPOCHS)
    st.session_state.setdefault("future_steps_input", DEFAULT_FUTURE_STEPS)
    st.session_state.setdefault("tune_xgb_input", False)
    st.session_state.setdefault("force_retrain_input", False)


def build_config_key(params: dict) -> tuple:
    return (
        params["ticker"],
        params["period"],
        params["sequence_length"],
        params["lstm_epochs"],
        params["future_steps"],
        params["tune_xgb"],
    )


def render_sidebar_controls() -> tuple[bool, bool]:
    with st.sidebar:
        st.markdown("## Konfigurasi Penelitian")
        st.selectbox(
            "Ticker Yahoo Finance",
            AVAILABLE_TICKERS,
            key="ticker_input",
            format_func=lambda ticker: f"{ticker} - {TICKER_COMPANY_NAMES.get(ticker, ticker)}",
        )
        st.selectbox("Periode Data", AVAILABLE_PERIODS, key="period_input")
        st.slider("Sliding Window LSTM", min_value=10, max_value=90, key="sequence_length_input")
        st.slider("Epoch LSTM", min_value=10, max_value=80, step=5, key="lstm_epochs_input")
        st.slider("Horizon Simulasi", min_value=5, max_value=30, key="future_steps_input")
        st.checkbox(
            "Gunakan GridSearch ringan untuk XGBoost",
            key="tune_xgb_input",
            help="Opsional: tuning XGBoost dengan grid kecil untuk hasil perbandingan yang lebih baik.",
        )
        st.checkbox(
            "Paksa training ulang",
            key="force_retrain_input",
            help="Jika aktif, model dilatih ulang walaupun konfigurasi sama.",
        )
        run_button = st.button("Run Analysis", use_container_width=True, type="primary")
        reset_button = st.button("Reset Results", use_container_width=True)
        st.caption("Model pembanding: LSTM standalone, XGBoost standalone, dan Hybrid Stacking LSTM-XGBoost.")
    return run_button, reset_button


def update_pipeline(force_refresh: bool = False) -> None:
    params = {
        "ticker": st.session_state["ticker_input"].upper().strip(),
        "period": st.session_state["period_input"],
        "sequence_length": st.session_state["sequence_length_input"],
        "lstm_epochs": st.session_state["lstm_epochs_input"],
        "future_steps": st.session_state["future_steps_input"],
        "tune_xgb": st.session_state.get("tune_xgb_input", False),
    }
    config_key = build_config_key(params)
    has_cached_result = "research_results" in st.session_state and st.session_state.get("last_config_key") == config_key
    force_retrain = st.session_state.get("force_retrain_input", False)
    should_train = "research_results" not in st.session_state or st.session_state.get("last_config_key") != config_key

    if force_refresh and has_cached_result and not force_retrain:
        st.session_state["reuse_info"] = "Konfigurasi belum berubah, menggunakan hasil sebelumnya."
        return

    if force_refresh and force_retrain:
        should_train = True

    if should_train:
        st.session_state.pop("reuse_info", None)
        with st.spinner("Menjalankan pipeline penelitian, melatih model, dan menyiapkan visual..."):
            try:
                st.session_state["research_results"] = run_research_pipeline(**params)
                st.session_state["pipeline_params"] = params
                st.session_state["last_config_key"] = config_key
                st.session_state.pop("pipeline_error", None)
            except Exception as exc:
                st.session_state["research_results"] = build_empty_results(str(exc), params["ticker"])
                st.session_state["pipeline_params"] = params
                st.session_state["last_config_key"] = config_key
                st.session_state["pipeline_error"] = str(exc)


load_local_css(CSS_FILE)
initialize_session_state()
run_button, reset_button = render_sidebar_controls()

if reset_button:
    for key in ["research_results", "pipeline_params", "last_config_key", "pipeline_error", "reuse_info"]:
        st.session_state.pop(key, None)
    st.rerun()

update_pipeline(force_refresh=run_button or "research_results" not in st.session_state)

pages = {
    "Overview": [
        st.Page("pages/Home.py", title="Home", icon=":material/home:"),
        st.Page("pages/Data_Overview.py", title="Data Overview", icon=":material/dataset:"),
    ],
    "Models": [
        st.Page("pages/LSTM_Model.py", title="LSTM Model", icon=":material/neurology:"),
        st.Page("pages/XGBoost_Model.py", title="XGBoost Model", icon=":material/account_tree:"),
        st.Page("pages/Hybrid_Stacking.py", title="Hybrid Stacking", icon=":material/hub:"),
    ],
    "Analysis": [
        st.Page("pages/Evaluation.py", title="Evaluation", icon=":material/analytics:"),
        st.Page("pages/Simulation.py", title="Simulation", icon=":material/show_chart:"),
    ],
}

pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()
