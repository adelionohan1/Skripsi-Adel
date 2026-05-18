import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import PLOTLY_DARK_LAYOUT, dataframe_to_csv, get_results, has_data, make_download_filename, render_empty_state, render_metric_card


results = get_results()
evaluation_df = results["evaluation_table"]
xgb_importance_df = results.get("feature_importance", pd.DataFrame())
hybrid_importance_df = results.get("hybrid_feature_importance", pd.DataFrame())

st.title("Evaluation")

if not has_data(evaluation_df):
    render_empty_state()
else:
    # Gabungkan semua prediction frames untuk visualisasi
    all_predictions = pd.concat([
        results.get("lstm_prediction", pd.DataFrame()),
        results.get("xgb_prediction", pd.DataFrame()),
        results.get("hybrid_prediction", pd.DataFrame())
    ], ignore_index=True)

    metric_cols = st.columns(3)
    best_mae = evaluation_df.sort_values("MAE").iloc[0]
    best_rmse = evaluation_df.sort_values("RMSE").iloc[0]
    best_direction = evaluation_df.sort_values("Directional Accuracy (%)", ascending=False).iloc[0]
    render_metric_card(metric_cols[0], "Model Terbaik (MAE)", best_mae["Model"], f"{best_mae['MAE']:.4f}", tone="positive")
    render_metric_card(metric_cols[1], "Model Terbaik (RMSE)", best_rmse["Model"], f"{best_rmse['RMSE']:.4f}", tone="positive")
    render_metric_card(metric_cols[2], "Akurasi Arah", best_direction["Model"], f"{best_direction['Directional Accuracy (%)']:.2f}%", tone="positive")

    summary_tab, detail_tab, visual_tab = st.tabs(["Ringkasan", "Detail Model", "Visualisasi"])

    with summary_tab:
        st.success(f"Highlight model terbaik berdasarkan RMSE: {best_rmse['Model']}")
        st.markdown(
            """
            <div class='card' style='padding:1rem; margin-bottom:1rem; border:1px solid rgba(56,189,248,0.35);'>
            <strong>Catatan evaluasi:</strong> Semua metrik dihitung pada test set yang sama untuk LSTM standalone, XGBoost standalone, dan Hybrid Stacking LSTM-XGBoost.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(evaluation_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Evaluation CSV",
            data=dataframe_to_csv(evaluation_df),
            file_name=make_download_filename("evaluation", results["params"]["ticker"]),
            mime="text/csv",
            use_container_width=True,
        )

    with detail_tab:
        with st.expander("Penjelasan metodologi", expanded=True):
            st.markdown(
                """
                `# bagian ini sesuai metodologi penelitian skripsi`

                Evaluasi menggunakan metrik MAE dan RMSE, dan hanya dihitung pada test set.
                Pendekatan ini menjaga validitas eksperimen karena model tidak dievaluasi pada data training.
                """
            )

    with visual_tab:
        # Plot perbandingan error
        melted_df = evaluation_df.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Value")
        bar_chart = px.bar(
            melted_df,
            x="Model",
            y="Value",
            color="Metric",
            barmode="group",
            title="Perbandingan Error pada Test Set",
            text_auto=".4f",
            color_discrete_sequence=["#38bdf8", "#f59e0b"],
        )
        bar_chart.update_layout(**PLOTLY_DARK_LAYOUT)
        st.plotly_chart(bar_chart, use_container_width=True)

        # Plot aktual vs prediksi untuk semua model
        if not all_predictions.empty:
            line_chart = go.Figure()
            actual_df = all_predictions.drop_duplicates(subset=["Date"])[["Date", "Actual"]]
            line_chart.add_trace(
                go.Scatter(
                    x=actual_df["Date"],
                    y=actual_df["Actual"],
                    mode="lines",
                    name="Aktual",
                    line={"color": "#f8fafc", "width": 3},
                )
            )
            model_colors = {
                "LSTM": "#38bdf8",
                "XGBoost": "#22c55e",
                "Hybrid Stacking LSTM-XGBoost": "#f59e0b",
            }
            for model_name, model_df in all_predictions.groupby("Model", sort=False):
                line_chart.add_trace(
                    go.Scatter(
                        x=model_df["Date"],
                        y=model_df["Prediction"],
                        mode="lines",
                        name=model_name,
                        line={"color": model_colors.get(model_name, "#a78bfa"), "width": 2},
                    )
                )
            line_chart.update_layout(title="Aktual vs Prediksi Model pada Test Set", hovermode="x unified", **PLOTLY_DARK_LAYOUT)
            st.plotly_chart(line_chart, use_container_width=True)
        else:
            st.warning("Data prediksi tidak tersedia untuk visualisasi.")

        technical_features = {"SMA_10", "SMA_20", "RSI_14", "BB_Middle", "BB_Upper", "BB_Lower"}

        if not xgb_importance_df.empty:
            top_xgb = xgb_importance_df.iloc[0]
            xgb_importance_chart = px.bar(
                xgb_importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance XGBoost Standalone",
                color_discrete_sequence=["#22c55e"],
            )
            xgb_importance_chart.update_layout(yaxis={"autorange": "reversed"}, **PLOTLY_DARK_LAYOUT)
            st.plotly_chart(xgb_importance_chart, use_container_width=True)
            xgb_technical_share = xgb_importance_df.loc[xgb_importance_df["Feature"].isin(technical_features), "Importance"].sum()
            st.info(
                f"Interpretasi XGBoost: fitur paling berpengaruh adalah {top_xgb['Feature']} "
                f"dengan importance {top_xgb['Importance']:.4f}. Total kontribusi indikator teknikal "
                f"sekitar {xgb_technical_share:.4f} dari total importance model."
            )

        if not hybrid_importance_df.empty:
            top_hybrid = hybrid_importance_df.iloc[0]
            hybrid_importance_chart = px.bar(
                hybrid_importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance Hybrid Stacking",
                color_discrete_sequence=["#f59e0b"],
            )
            hybrid_importance_chart.update_layout(yaxis={"autorange": "reversed"}, **PLOTLY_DARK_LAYOUT)
            st.plotly_chart(hybrid_importance_chart, use_container_width=True)
            lstm_importance = hybrid_importance_df.loc[
                hybrid_importance_df["Feature"] == "LSTM_Prediction",
                "Importance",
            ]
            lstm_importance_value = float(lstm_importance.iloc[0]) if not lstm_importance.empty else 0.0
            hybrid_technical_share = hybrid_importance_df.loc[hybrid_importance_df["Feature"].isin(technical_features), "Importance"].sum()
            st.info(
                f"Interpretasi Hybrid: fitur paling berpengaruh adalah {top_hybrid['Feature']} "
                f"dengan importance {top_hybrid['Importance']:.4f}. Fitur LSTM_Prediction memiliki "
                f"importance {lstm_importance_value:.4f}, sedangkan gabungan indikator teknikal "
                f"berkontribusi sekitar {hybrid_technical_share:.4f}."
            )

st.caption("Sumber: Hasil Implementasi Sistem, 2026")
