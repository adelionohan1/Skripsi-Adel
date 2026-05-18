import streamlit as st

from config.settings import APP_SUBTITLE, APP_TITLE
from utils.ui import dataframe_to_csv, get_results, has_data, make_download_filename, render_empty_state, render_metric_card


results = get_results()
summary = results["summary"]
evaluation_df = results["evaluation_table"]
pipeline_error = results.get("error") or st.session_state.get("pipeline_error")

st.title("Home")
st.markdown(
    f"""
    <div class="hero-card">
        <p class="hero-tag">Dashboard Penelitian</p>
        <h1>{APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if pipeline_error:
    st.warning(pipeline_error)
if st.session_state.get("reuse_info"):
    st.info(st.session_state["reuse_info"])

metric_cols = st.columns(4)
render_metric_card(metric_cols[0], "Ticker", results["params"]["ticker"], tone="neutral")
render_metric_card(metric_cols[1], "Nama Emiten", summary.get("company_name", "-"), tone="neutral")
render_metric_card(metric_cols[2], "Observasi", f"{summary['observation_count']:,}", tone="neutral")
render_metric_card(metric_cols[3], "Harga Terakhir", f"{summary['latest_close']:.2f}", tone="neutral")

model_cols = st.columns(3)
daily_tone = "positive" if summary["daily_change_pct"] >= 0 else "negative"
render_metric_card(
    model_cols[0],
    "Perubahan Harian",
    f"{summary['daily_change_pct']:.2f}%",
    delta="Naik" if summary["daily_change_pct"] >= 0 else "Turun",
    tone=daily_tone,
)
render_metric_card(model_cols[1], "Model Terbaik RMSE", summary.get("best_rmse_model", "-"), tone="positive")
render_metric_card(model_cols[2], "Model Terbaik MAE", summary.get("best_mae_model", "-"), tone="positive")

if has_data(evaluation_df):
    best_rmse = evaluation_df.sort_values("RMSE").iloc[0]
    best_mae = evaluation_df.sort_values("MAE").iloc[0]
    best_direction = evaluation_df.sort_values("Directional Accuracy (%)", ascending=False).iloc[0]
    worst_rmse = evaluation_df.sort_values("RMSE", ascending=False).iloc[0]
    st.markdown(
        f"""
        <div class="card summary-card">
            <strong>Ringkasan otomatis:</strong>
            Model terbaik berdasarkan RMSE adalah <strong>{best_rmse['Model']}</strong>
            dengan RMSE sebesar <strong>{best_rmse['RMSE']:.4f}</strong>.
            MAE terbaik dicapai oleh <strong>{best_mae['Model']}</strong>
            dengan MAE sebesar <strong>{best_mae['MAE']:.4f}</strong>.
            Directional Accuracy tertinggi dimiliki <strong>{best_direction['Model']}</strong>
            sebesar <strong>{best_direction['Directional Accuracy (%)']:.2f}%</strong>.
            Model dengan RMSE terbesar adalah <strong>{worst_rmse['Model']}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

summary_tab, detail_tab, visual_tab = st.tabs(["Ringkasan", "Detail Penelitian", "Dokumentasi"])

with summary_tab:
    left_col, right_col = st.columns([1.25, 1.0])
    with left_col:
        st.markdown("### Preview Hasil Evaluasi")
        if has_data(evaluation_df):
            st.dataframe(evaluation_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Evaluation CSV",
                data=dataframe_to_csv(evaluation_df),
                file_name=make_download_filename("evaluation", results["params"]["ticker"]),
                mime="text/csv",
                use_container_width=True,
            )
        else:
            render_empty_state()
    with right_col:
        st.markdown("### Fokus Sistem")
        st.markdown(
            """
            - Sistem memprediksi harga saham volatilitas tinggi dengan pendekatan Hybrid Stacking LSTM-XGBoost.
            - Evaluasi utama dilakukan pada test set dengan MAE, RMSE, MAPE, dan Directional Accuracy.
            - Simulasi future horizon ditampilkan sebagai eksplorasi perilaku model, bukan bukti utama evaluasi.
            - Konfigurasi yang sama memakai hasil sebelumnya dari `session_state` untuk menjaga stabilitas hasil.
            """
        )

with detail_tab:
    with st.expander("Metodologi Penelitian", expanded=True):
        st.markdown(
            """
            - Data historis diunduh dari Yahoo Finance dan dibersihkan sebelum pemodelan.
            - Fitur teknikal meliputi SMA 10, SMA 20, RSI 14, dan Bollinger Bands.
            - Split data dilakukan secara sequential tanpa shuffle.
            - MinMaxScaler hanya di-fit pada data training untuk mencegah data leakage.
            - Model yang dibandingkan adalah LSTM standalone, XGBoost standalone, dan Hybrid Stacking LSTM-XGBoost.
            """
        )
    with st.expander("Navigasi Dashboard", expanded=False):
        st.markdown(
            """
            - `Home` merangkum hasil penelitian.
            - `Data Overview` mendokumentasikan preprocessing dan feature engineering.
            - `LSTM Model`, `XGBoost Model`, dan `Hybrid Stacking` menampilkan implementasi tiap model.
            - `Evaluation` dan `Simulation` menampilkan analisis performa dan eksplorasi prediksi.
            """
        )

with visual_tab:
    best_model = evaluation_df.iloc[0]["Model"] if has_data(evaluation_df) else "-"
    st.success(f"Model terbaik berdasarkan RMSE saat ini: {best_model}")
    if has_data(evaluation_df):
        st.dataframe(
            evaluation_df.style.highlight_min(subset=["RMSE"], color="#14532d"),
            use_container_width=True,
            hide_index=True,
        )
    else:
        render_empty_state()

st.caption("Sumber: Hasil Implementasi Sistem, 2026")
