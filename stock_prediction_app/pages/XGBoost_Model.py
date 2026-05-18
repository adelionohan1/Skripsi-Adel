import plotly.graph_objects as go
import streamlit as st

from config.settings import FEATURE_COLUMNS, XGB_CONFIG
from utils.ui import PLOTLY_DARK_LAYOUT, get_results, has_data, render_empty_state


results = get_results()
xgb_df = results.get("xgb_prediction")
feature_importance_df = results.get("feature_importance")

st.title("XGBoost Model")
st.markdown("Implementasi model XGBoost standalone dengan input fitur harga dan indikator teknikal.")

metric_cols = st.columns(5)
metric_cols[0].metric("n_estimators", XGB_CONFIG["n_estimators"])
metric_cols[1].metric("max_depth", XGB_CONFIG["max_depth"])
metric_cols[2].metric("learning_rate", XGB_CONFIG["learning_rate"])
metric_cols[3].metric("subsample", XGB_CONFIG["subsample"])
metric_cols[4].metric("colsample_bytree", XGB_CONFIG["colsample_bytree"])

feature_tab, importance_tab, prediction_tab = st.tabs(
    ["Feature Input", "Feature Importance", "Actual vs Prediction"]
)

with feature_tab:
    st.markdown("### Fitur Input XGBoost")
    st.dataframe(
        {"Feature": FEATURE_COLUMNS},
        use_container_width=True,
        hide_index=True,
    )

with importance_tab:
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
        importance_chart.update_layout(
            title="Feature Importance XGBoost Standalone",
            yaxis={"autorange": "reversed"},
            **PLOTLY_DARK_LAYOUT,
        )
        st.plotly_chart(importance_chart, use_container_width=True)
    else:
        render_empty_state("Feature importance XGBoost belum tersedia.")

with prediction_tab:
    if has_data(xgb_df):
        prediction_chart = go.Figure()
        prediction_chart.add_trace(
            go.Scatter(x=xgb_df["Date"], y=xgb_df["Actual"], mode="lines", name="Aktual", line={"color": "#f8fafc", "width": 3})
        )
        prediction_chart.add_trace(
            go.Scatter(x=xgb_df["Date"], y=xgb_df["Prediction"], mode="lines", name="Prediksi XGBoost", line={"color": "#22c55e", "width": 2.2})
        )
        prediction_chart.update_layout(title="Actual vs Prediction (XGBoost)", hovermode="x unified", **PLOTLY_DARK_LAYOUT)
        st.plotly_chart(prediction_chart, use_container_width=True)
    else:
        render_empty_state("Prediksi XGBoost belum tersedia.")

st.caption("Sumber: Hasil Implementasi Sistem, 2026")
