import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.ui import PLOTLY_DARK_LAYOUT, get_results, has_data, render_empty_state


def render_hybrid_flow() -> None:
    sankey = go.Figure(
        data=[
            go.Sankey(
                node={
                    "label": [
                        "Historical Data",
                        "LSTM Feature Extraction",
                        "LSTM Prediction",
                        "Technical Indicators",
                        "XGBoost Meta-Learner",
                        "Final Prediction",
                    ],
                    "pad": 18,
                    "thickness": 18,
                    "color": ["#38bdf8", "#22c55e", "#f59e0b", "#38bdf8", "#f97316", "#f8fafc"],
                },
                link={
                    "source": [0, 1, 2, 0, 3, 4],
                    "target": [1, 2, 4, 3, 4, 5],
                    "value": [1, 1, 1, 1, 1, 2],
                    "color": [
                        "rgba(56,189,248,0.35)",
                        "rgba(34,197,94,0.35)",
                        "rgba(245,158,11,0.35)",
                        "rgba(56,189,248,0.22)",
                        "rgba(249,115,22,0.35)",
                        "rgba(248,250,252,0.35)",
                    ],
                },
            )
        ]
    )
    sankey.update_layout(title="Diagram Hybrid Stacking LSTM-XGBoost", **PLOTLY_DARK_LAYOUT)
    st.plotly_chart(sankey, use_container_width=True)


results = get_results()
lstm_df = results.get("lstm_prediction", pd.DataFrame())
xgb_df = results.get("xgb_prediction", pd.DataFrame())
hybrid_df = results.get("hybrid_prediction", pd.DataFrame())
meta_feature_df = results.get("hybrid_meta_preview", pd.DataFrame())

st.title("Hybrid Stacking")
st.markdown("Halaman inti penelitian yang mendokumentasikan arsitektur dan hasil implementasi Hybrid Stacking LSTM-XGBoost.")

diagram_tab, meta_tab, prediction_tab, comparison_tab = st.tabs(
    ["Diagram", "Meta-Feature", "Hybrid Prediction", "Comparison"]
)

with diagram_tab:
    render_hybrid_flow()

with meta_tab:
    st.markdown("### Meta-Feature Dataset")
    if has_data(meta_feature_df):
        st.dataframe(meta_feature_df.round(4), use_container_width=True, hide_index=True)
    else:
        render_empty_state("Preview meta-feature hybrid belum tersedia.")

with prediction_tab:
    if has_data(hybrid_df):
        hybrid_chart = go.Figure()
        hybrid_chart.add_trace(
            go.Scatter(x=hybrid_df["Date"], y=hybrid_df["Actual"], mode="lines", name="Aktual", line={"color": "#f8fafc", "width": 3})
        )
        hybrid_chart.add_trace(
            go.Scatter(
                x=hybrid_df["Date"],
                y=hybrid_df["Prediction"],
                mode="lines",
                name="Prediksi Hybrid",
                line={"color": "#f59e0b", "width": 2.2},
            )
        )
        hybrid_chart.update_layout(title="Actual vs Prediction (Hybrid Stacking)", hovermode="x unified", **PLOTLY_DARK_LAYOUT)
        st.plotly_chart(hybrid_chart, use_container_width=True)
    else:
        render_empty_state("Prediksi hybrid belum tersedia.")

with comparison_tab:
    if has_data(lstm_df) and has_data(xgb_df) and has_data(hybrid_df):
        comparison_chart = go.Figure()
        comparison_chart.add_trace(
            go.Scatter(x=hybrid_df["Date"], y=hybrid_df["Actual"], mode="lines", name="Aktual", line={"color": "#f8fafc", "width": 3})
        )
        comparison_chart.add_trace(
            go.Scatter(x=lstm_df["Date"], y=lstm_df["Prediction"], mode="lines", name="LSTM", line={"color": "#38bdf8", "width": 2})
        )
        comparison_chart.add_trace(
            go.Scatter(x=xgb_df["Date"], y=xgb_df["Prediction"], mode="lines", name="XGBoost", line={"color": "#22c55e", "width": 2})
        )
        comparison_chart.add_trace(
            go.Scatter(x=hybrid_df["Date"], y=hybrid_df["Prediction"], mode="lines", name="Hybrid", line={"color": "#f59e0b", "width": 2.2})
        )
        comparison_chart.update_layout(title="Perbandingan LSTM, XGBoost, dan Hybrid", hovermode="x unified", **PLOTLY_DARK_LAYOUT)
        st.plotly_chart(comparison_chart, use_container_width=True)
    else:
        render_empty_state("Perbandingan model belum tersedia.")

with st.expander("Penjelasan Hybrid Stacking", expanded=True):
    st.markdown(
        """
        - LSTM terlebih dahulu dilatih untuk menangkap pola temporal harga saham dan menghasilkan prediksi next-step.
        - Output prediksi LSTM digunakan sebagai `meta-feature` tambahan bersama fitur harga dan indikator teknikal.
        - XGBoost bertindak sebagai `meta-learner` yang mempelajari gabungan fitur tersebut untuk menghasilkan prediksi akhir.
        - Pendekatan hybrid dipilih agar pola sekuensial dari LSTM dan kemampuan non-linear boosting dari XGBoost dapat saling melengkapi.
        """
    )

st.caption("Sumber: Hasil Implementasi Sistem, 2026")
