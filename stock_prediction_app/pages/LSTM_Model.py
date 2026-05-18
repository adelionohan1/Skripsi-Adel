from io import StringIO

import plotly.graph_objects as go
import streamlit as st

from config.settings import LSTM_CONFIG
from utils.model_lstm import build_lstm_model
from utils.ui import PLOTLY_DARK_LAYOUT, get_results, has_data, render_empty_state


def render_lstm_architecture() -> None:
    sankey = go.Figure(
        data=[
            go.Sankey(
                node={
                    "label": [
                        "Input",
                        "LSTM 64",
                        "Dropout",
                        "LSTM 32",
                        "Dropout",
                        "LSTM 16",
                        "Dropout",
                        "Dense 16",
                        "Output",
                    ],
                    "pad": 18,
                    "thickness": 18,
                    "color": ["#38bdf8", "#38bdf8", "#1f2937", "#22c55e", "#1f2937", "#14b8a6", "#1f2937", "#f59e0b", "#f97316"],
                },
                link={
                    "source": [0, 1, 2, 3, 4, 5, 6, 7],
                    "target": [1, 2, 3, 4, 5, 6, 7, 8],
                    "value": [1, 1, 1, 1, 1, 1, 1, 1],
                    "color": [
                        "rgba(56,189,248,0.35)",
                        "rgba(56,189,248,0.35)",
                        "rgba(34,197,94,0.35)",
                        "rgba(34,197,94,0.35)",
                        "rgba(20,184,166,0.35)",
                        "rgba(20,184,166,0.35)",
                        "rgba(245,158,11,0.35)",
                        "rgba(249,115,22,0.35)",
                    ],
                },
            )
        ]
    )
    sankey.update_layout(title="Arsitektur Model LSTM", **PLOTLY_DARK_LAYOUT)
    st.plotly_chart(sankey, use_container_width=True)


def get_model_summary(sequence_length: int) -> str:
    buffer = StringIO()
    model = build_lstm_model(
        sequence_length=sequence_length,
        units=LSTM_CONFIG["units"],
        dropout=LSTM_CONFIG["dropout"],
        learning_rate=LSTM_CONFIG["learning_rate"],
    )
    model.summary(print_fn=lambda line: buffer.write(f"{line}\n"))
    return buffer.getvalue()


results = get_results()
lstm_df = results.get("lstm_prediction")
history_df = results.get("lstm_history")

st.title("LSTM Model")
st.markdown("Implementasi model LSTM standalone untuk menangkap pola temporal harga saham.")

metric_cols = st.columns(5)
metric_cols[0].metric("Sequence Length", results["params"]["sequence_length"])
metric_cols[1].metric("Epoch", results["params"]["lstm_epochs"])
metric_cols[2].metric("Batch Size", LSTM_CONFIG["batch_size"])
metric_cols[3].metric("Learning Rate", LSTM_CONFIG["learning_rate"])
metric_cols[4].metric("Units", "64-32-16")

arch_tab, summary_tab, loss_tab, prediction_tab = st.tabs(
    ["Arsitektur", "Model Summary", "Training Loss", "Actual vs Prediction"]
)

with arch_tab:
    render_lstm_architecture()

with summary_tab:
    st.code(get_model_summary(results["params"]["sequence_length"]), language="text")

with loss_tab:
    if has_data(history_df) and "loss" in history_df.columns:
        loss_chart = go.Figure()
        loss_chart.add_trace(
            go.Scatter(y=history_df["loss"], mode="lines", name="Training Loss", line={"color": "#38bdf8", "width": 2.4})
        )
        if "val_loss" in history_df.columns:
            loss_chart.add_trace(
                go.Scatter(y=history_df["val_loss"], mode="lines", name="Validation Loss", line={"color": "#f59e0b", "width": 2.2})
            )
        loss_chart.update_layout(title="Training Loss LSTM", xaxis_title="Epoch", yaxis_title="Loss", **PLOTLY_DARK_LAYOUT)
        st.plotly_chart(loss_chart, use_container_width=True)
    else:
        render_empty_state("Kurva training loss LSTM belum tersedia.")

with prediction_tab:
    if has_data(lstm_df):
        prediction_chart = go.Figure()
        prediction_chart.add_trace(
            go.Scatter(x=lstm_df["Date"], y=lstm_df["Actual"], mode="lines", name="Aktual", line={"color": "#f8fafc", "width": 3})
        )
        prediction_chart.add_trace(
            go.Scatter(x=lstm_df["Date"], y=lstm_df["Prediction"], mode="lines", name="Prediksi LSTM", line={"color": "#38bdf8", "width": 2.2})
        )
        prediction_chart.update_layout(title="Actual vs Prediction (LSTM)", hovermode="x unified", **PLOTLY_DARK_LAYOUT)
        st.plotly_chart(prediction_chart, use_container_width=True)
    else:
        render_empty_state("Prediksi LSTM belum tersedia.")

st.caption("Sumber: Hasil Implementasi Sistem, 2026")
