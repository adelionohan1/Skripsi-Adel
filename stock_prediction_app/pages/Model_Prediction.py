import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from utils.ui import PLOTLY_DARK_LAYOUT, dataframe_to_csv, get_results, has_data, make_download_filename, render_empty_state, render_metric_card


results = get_results()
lstm_df = results["lstm_prediction"]
xgb_df = results["xgb_prediction"]
hybrid_df = results["hybrid_prediction"]
feature_importance_df = results["feature_importance"]
hybrid_feature_importance_df = results.get("hybrid_feature_importance")
best_model = results["evaluation_table"].iloc[0]["Model"] if has_data(results["evaluation_table"]) else "-"
prediction_export_df = pd.concat([lstm_df, xgb_df, hybrid_df], ignore_index=True)

st.title("Model Prediction")

model_choice = st.radio(
    "Pilih model untuk ditampilkan",
    options=["Semua", "LSTM", "XGBoost", "Hybrid Stacking"],
    horizontal=True,
    help="Gunakan toggle ini untuk fokus pada salah satu model atau bandingkan semua model secara bersamaan.",
)
show_charts = st.checkbox("Tampilkan grafik prediksi", value=True)

metric_cols = st.columns(4)
render_metric_card(metric_cols[0], "LSTM RMSE", f"{results['lstm_metrics']['RMSE']:.4f}")
render_metric_card(metric_cols[1], "XGBoost RMSE", f"{results['xgb_metrics']['RMSE']:.4f}")
render_metric_card(metric_cols[2], "Hybrid MAE", f"{results['hybrid_metrics']['MAE']:.4f}")
render_metric_card(metric_cols[3], "Hybrid RMSE", f"{results['hybrid_metrics']['RMSE']:.4f}")

summary_tab, detail_tab, visual_tab = st.tabs(["Ringkasan", "Detail Model", "Visualisasi"])

with summary_tab:
    st.success(f"Highlight model terbaik saat ini: {best_model}")
    st.markdown("---")
    if not show_charts:
        st.info("Grafik prediksi disembunyikan. Aktifkan kembali toggle untuk melihat performa model.")
    elif has_data(lstm_df):
        comparison_chart = go.Figure()
        comparison_chart.add_trace(go.Scatter(x=lstm_df["Date"], y=lstm_df["Actual"], name="Aktual", mode="lines", line={"color": "#f8fafc", "width": 3}))
        if model_choice in ["Semua", "LSTM"] and has_data(lstm_df):
            comparison_chart.add_trace(go.Scatter(x=lstm_df["Date"], y=lstm_df["Prediction"], name="LSTM", mode="lines", line={"color": "#38bdf8"}))
        if model_choice in ["Semua", "XGBoost"] and has_data(xgb_df):
            comparison_chart.add_trace(go.Scatter(x=xgb_df["Date"], y=xgb_df["Prediction"], name="XGBoost", mode="lines", line={"color": "#22c55e"}))
        if model_choice in ["Semua", "Hybrid Stacking"] and has_data(hybrid_df):
            comparison_chart.add_trace(go.Scatter(x=hybrid_df["Date"], y=hybrid_df["Prediction"], name="Hybrid Stacking", mode="lines", line={"color": "#f59e0b"}))
        comparison_chart.update_layout(title="Aktual vs Prediksi pada Test Set", hovermode="x unified", **PLOTLY_DARK_LAYOUT)
        st.plotly_chart(comparison_chart, use_container_width=True)

    if not show_charts and (model_choice == "Semua" or not has_data(lstm_df) and not has_data(xgb_df) and not has_data(hybrid_df)):
        render_empty_state()

    if has_data(prediction_export_df):
        st.download_button(
            "Download Prediction CSV",
            data=dataframe_to_csv(prediction_export_df),
            file_name=make_download_filename("prediction", results["params"]["ticker"]),
            mime="text/csv",
            use_container_width=True,
        )

with detail_tab:
    history_df = results["lstm_history"].copy()
    if has_data(history_df) and "val_loss" in history_df.columns:
        loss_chart = go.Figure()
        loss_chart.add_trace(go.Scatter(y=history_df["loss"], mode="lines", name="Train Loss", line={"color": "#38bdf8"}))
        loss_chart.add_trace(go.Scatter(y=history_df["val_loss"], mode="lines", name="Validation Loss", line={"color": "#f59e0b"}))
        loss_chart.update_layout(title="LSTM Training Curve", xaxis_title="Epoch", yaxis_title="Loss", **PLOTLY_DARK_LAYOUT)
        st.plotly_chart(loss_chart, use_container_width=True)
    else:
        render_empty_state()

    with st.expander("Informasi model", expanded=True):
        st.markdown(
            """
            `# bagian ini sesuai metodologi penelitian skripsi`

            - LSTM menggunakan 3 layer, dropout, optimizer Adam, dan loss MSE.
            - XGBoost digunakan sebagai model pembanding non-deep-learning.
            - Hybrid Stacking memakai prediksi LSTM sebagai fitur tambahan untuk XGBoost meta-learner.
            - Seluruh evaluasi yang ditampilkan di halaman ini berasal dari data test set.
            """
        )

with visual_tab:
    if has_data(feature_importance_df) or has_data(hybrid_feature_importance_df):
        if has_data(feature_importance_df):
            importance_chart = go.Figure()
            importance_chart.add_trace(
                go.Bar(
                    x=feature_importance_df["Importance"],
                    y=feature_importance_df["Feature"],
                    orientation="h",
                    marker={"color": "#22c55e"},
                    name="Importance",
                )
            )
            importance_chart.update_layout(title="Feature Importance XGBoost", yaxis={"autorange": "reversed"}, **PLOTLY_DARK_LAYOUT)
            st.plotly_chart(importance_chart, use_container_width=True)

        if has_data(hybrid_feature_importance_df):
            hybrid_importance_chart = go.Figure()
            hybrid_importance_chart.add_trace(
                go.Bar(
                    x=hybrid_feature_importance_df["Importance"],
                    y=hybrid_feature_importance_df["Feature"],
                    orientation="h",
                    marker={"color": "#f59e0b"},
                    name="Importance",
                )
            )
            hybrid_importance_chart.update_layout(title="Feature Importance Hybrid Stacking", yaxis={"autorange": "reversed"}, **PLOTLY_DARK_LAYOUT)
            st.plotly_chart(hybrid_importance_chart, use_container_width=True)
    else:
        render_empty_state()
