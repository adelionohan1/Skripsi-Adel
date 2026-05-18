import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import PLOTLY_DARK_LAYOUT, get_results, has_data, render_empty_state, render_metric_card


RESULTS_CAPTION = "Sumber: Hasil Implementasi Sistem, 2026"
RAW_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
PROCESSED_COLUMNS = [
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
    "Target",
]

results = get_results()
raw_df = results.get("raw_data", pd.DataFrame())
feature_df = results.get("feature_data", pd.DataFrame())
processed_df = results.get("processed_data", pd.DataFrame())
correlation_matrix = results.get("correlation_matrix", pd.DataFrame())
split_info = results.get("split_info", {})
scaling_before_df = results.get("scaling_preview_before", pd.DataFrame())
scaling_after_df = results.get("scaling_preview_after", pd.DataFrame())
sequence_info = results.get("sequence_info", {})

st.title("Data Overview")
st.markdown(
    """
    <div class="card section-card">
        <strong>Dokumentasi preprocessing penelitian</strong><br>
        Halaman ini merangkum tahapan preprocessing, feature engineering, normalisasi, transformasi sequence,
        dan analisis korelasi untuk model Hybrid Stacking LSTM-XGBoost.
    </div>
    """,
    unsafe_allow_html=True,
)

if not has_data(raw_df):
    render_empty_state("Data mentah tidak tersedia untuk divisualisasikan.")
else:
    top_cols = st.columns(4)
    render_metric_card(top_cols[0], "Rentang Data", f"{raw_df['Date'].min().date()} s.d. {raw_df['Date'].max().date()}")
    render_metric_card(top_cols[1], "Harga Minimum", f"{raw_df['Close'].min():.2f}")
    render_metric_card(top_cols[2], "Harga Maksimum", f"{raw_df['Close'].max():.2f}")
    split_label = f"{split_info.get('train_count', 0)} / {split_info.get('test_count', 0)}"
    render_metric_card(top_cols[3], "Train / Test", split_label, delta="Sequential Split", tone="neutral")

    preprocess_tab, technical_tab, correlation_tab = st.tabs(
        ["Preprocessing", "Indikator Teknikal", "Korelasi & Metodologi"]
    )

    with preprocess_tab:
        st.subheader("Raw Dataset Preview")
        st.write("Dataset historis saham sebelum preprocessing.")
        st.dataframe(raw_df[RAW_COLUMNS].head(10), use_container_width=True, hide_index=True)
        st.caption(RESULTS_CAPTION)

        st.subheader("Missing Value Analysis")
        missing_summary_df = raw_df[RAW_COLUMNS].isna().sum().reset_index()
        missing_summary_df.columns = ["Kolom", "Jumlah Missing Value"]
        metric_cols = st.columns(2)
        render_metric_card(metric_cols[0], "Jumlah Total Data", f"{len(raw_df):,}")
        render_metric_card(metric_cols[1], "Jumlah Fitur", f"{len(RAW_COLUMNS)}")
        if int(missing_summary_df["Jumlah Missing Value"].sum()) == 0:
            st.success("No missing values detected.")
        else:
            st.warning("Masih terdapat missing value pada data mentah.")
        st.dataframe(missing_summary_df, use_container_width=True, hide_index=True)
        st.caption(RESULTS_CAPTION)

        st.subheader("Processed Dataset Preview")
        st.write("Dataset setelah preprocessing dan feature engineering.")
        if has_data(processed_df):
            available_processed_columns = [column for column in PROCESSED_COLUMNS if column in processed_df.columns]
            st.dataframe(
                processed_df[available_processed_columns].head(10),
                use_container_width=True,
                hide_index=True,
            )
        else:
            render_empty_state("Dataset hasil preprocessing belum tersedia.")
        st.caption(RESULTS_CAPTION)

        st.subheader("Normalization Visualization")
        st.write("Normalisasi dilakukan menggunakan MinMaxScaler pada data training untuk menghindari data leakage.")
        norm_cols = st.columns(2)
        with norm_cols[0]:
            st.markdown("**Sebelum Scaling**")
            if has_data(scaling_before_df):
                st.dataframe(scaling_before_df.head(5), use_container_width=True, hide_index=True)
            else:
                render_empty_state("Preview data sebelum scaling belum tersedia.")
        with norm_cols[1]:
            st.markdown("**Sesudah Scaling**")
            if has_data(scaling_after_df):
                st.dataframe(scaling_after_df.head(5).round(4), use_container_width=True, hide_index=True)
            else:
                render_empty_state("Preview data sesudah scaling belum tersedia.")
        st.caption(RESULTS_CAPTION)

        st.subheader("Sequence Transformation Information")
        st.write("Data dibentuk menggunakan metode sliding window untuk mempertahankan dependensi temporal pada time series.")
        if sequence_info:
            sequence_cols = st.columns(5)
            render_metric_card(sequence_cols[0], "Sequence Length", str(sequence_info.get("sequence_length", "-")))
            render_metric_card(sequence_cols[1], "X_train shape", str(sequence_info.get("x_train_shape", "-")))
            render_metric_card(sequence_cols[2], "y_train shape", str(sequence_info.get("y_train_shape", "-")))
            render_metric_card(sequence_cols[3], "X_test shape", str(sequence_info.get("x_test_shape", "-")))
            render_metric_card(sequence_cols[4], "y_test shape", str(sequence_info.get("y_test_shape", "-")))
        else:
            render_empty_state("Informasi sequence LSTM belum tersedia.")
        st.caption(RESULTS_CAPTION)

        st.subheader("Sliding Window Sequence Transformation")
        if len(raw_df) > 31:
            window_length = int(sequence_info.get("sequence_length", 30))
            window_length = min(window_length, max(5, len(raw_df) - 1))
            illustration_df = raw_df[["Date", "Close"]].head(window_length + 1).copy()
            history_df = illustration_df.iloc[:-1]
            target_row = illustration_df.iloc[-1]

            sliding_window_chart = go.Figure()
            sliding_window_chart.add_trace(
                go.Scatter(
                    x=history_df["Date"],
                    y=history_df["Close"],
                    mode="lines+markers",
                    name=f"Window {window_length} Hari",
                    line={"color": "#38bdf8", "width": 2.5},
                )
            )
            sliding_window_chart.add_trace(
                go.Scatter(
                    x=[target_row["Date"]],
                    y=[target_row["Close"]],
                    mode="markers+text",
                    name="Target Berikutnya",
                    marker={"color": "#f59e0b", "size": 12},
                    text=["Target"],
                    textposition="top center",
                )
            )
            sliding_window_chart.update_layout(
                title="Sliding Window Sequence Transformation",
                xaxis_title="Tanggal",
                yaxis_title="Harga Close",
                hovermode="x unified",
                **PLOTLY_DARK_LAYOUT,
            )
            st.plotly_chart(sliding_window_chart, use_container_width=True)
        else:
            render_empty_state("Data belum cukup untuk membuat ilustrasi sliding window.")
        st.caption(RESULTS_CAPTION)

    with technical_tab:
        st.subheader("Visualisasi Train-Test Split")
        if has_data(raw_df) and split_info:
            split_chart = go.Figure()
            split_chart.add_trace(
                go.Scatter(
                    x=raw_df["Date"],
                    y=raw_df["Close"],
                    mode="lines",
                    name="Close",
                    line={"color": "#f8fafc", "width": 2.5},
                )
            )
            split_chart.add_vrect(
                x0=split_info["train_start"],
                x1=split_info["train_end"],
                fillcolor="#22c55e",
                opacity=0.12,
                line_width=0,
                annotation_text="Training",
                annotation_position="top left",
            )
            split_chart.add_vrect(
                x0=split_info["test_start"],
                x1=split_info["test_end"],
                fillcolor="#f59e0b",
                opacity=0.14,
                line_width=0,
                annotation_text="Testing",
                annotation_position="top left",
            )
            split_chart.add_vline(x=split_info["split_date"], line_dash="dash", line_color="#f59e0b")
            split_chart.update_layout(
                title="Pembagian Sequential Train-Test",
                xaxis_title="Tanggal",
                yaxis_title="Harga Close",
                hovermode="x unified",
                **PLOTLY_DARK_LAYOUT,
            )
            st.plotly_chart(split_chart, use_container_width=True)
        else:
            render_empty_state("Informasi split train-test belum tersedia.")
        st.caption(RESULTS_CAPTION)

        st.subheader("Visualisasi Technical Indicators")
        if has_data(feature_df):
            price_indicator_chart = go.Figure()
            price_indicator_chart.add_trace(
                go.Scatter(
                    x=feature_df["Date"],
                    y=feature_df["Close"],
                    mode="lines",
                    name="Close",
                    line={"color": "#f8fafc", "width": 2.8},
                )
            )
            price_indicator_chart.add_trace(
                go.Scatter(x=feature_df["Date"], y=feature_df["SMA_10"], mode="lines", name="SMA 10", line={"color": "#38bdf8", "width": 2})
            )
            price_indicator_chart.add_trace(
                go.Scatter(x=feature_df["Date"], y=feature_df["SMA_20"], mode="lines", name="SMA 20", line={"color": "#22c55e", "width": 2})
            )
            price_indicator_chart.add_trace(
                go.Scatter(x=feature_df["Date"], y=feature_df["BB_Upper"], mode="lines", name="BB Upper", line={"color": "#f59e0b", "width": 1.7})
            )
            price_indicator_chart.add_trace(
                go.Scatter(x=feature_df["Date"], y=feature_df["BB_Lower"], mode="lines", name="BB Lower", line={"color": "#f97316", "width": 1.7})
            )
            price_indicator_chart.update_layout(
                title="Pergerakan Harga, SMA, dan Bollinger Bands",
                xaxis_title="Tanggal",
                yaxis_title="Harga",
                hovermode="x unified",
                **PLOTLY_DARK_LAYOUT,
            )
            st.plotly_chart(price_indicator_chart, use_container_width=True)
            st.caption(RESULTS_CAPTION)

            tech_cols = st.columns(2)
            with tech_cols[0]:
                rsi_chart = go.Figure()
                rsi_chart.add_trace(
                    go.Scatter(
                        x=feature_df["Date"],
                        y=feature_df["RSI_14"],
                        mode="lines",
                        name="RSI 14",
                        line={"color": "#f59e0b", "width": 2.3},
                    )
                )
                rsi_chart.add_hline(y=70, line_dash="dash", line_color="#ef4444")
                rsi_chart.add_hline(y=30, line_dash="dash", line_color="#22c55e")
                rsi_chart.update_layout(title="Relative Strength Index (RSI 14)", hovermode="x unified", **PLOTLY_DARK_LAYOUT)
                st.plotly_chart(rsi_chart, use_container_width=True)
                st.caption(RESULTS_CAPTION)
            with tech_cols[1]:
                sma_chart = go.Figure()
                sma_chart.add_trace(
                    go.Scatter(
                        x=feature_df["Date"],
                        y=feature_df["Close"],
                        mode="lines",
                        name="Close",
                        line={"color": "#f8fafc", "width": 2.4},
                    )
                )
                sma_chart.add_trace(
                    go.Scatter(x=feature_df["Date"], y=feature_df["SMA_10"], mode="lines", name="SMA 10", line={"color": "#38bdf8", "width": 2})
                )
                sma_chart.add_trace(
                    go.Scatter(x=feature_df["Date"], y=feature_df["SMA_20"], mode="lines", name="SMA 20", line={"color": "#22c55e", "width": 2})
                )
                sma_chart.update_layout(title="Simple Moving Average", hovermode="x unified", **PLOTLY_DARK_LAYOUT)
                st.plotly_chart(sma_chart, use_container_width=True)
                st.caption(RESULTS_CAPTION)
        else:
            render_empty_state("Data indikator teknikal belum tersedia.")

    with correlation_tab:
        st.subheader("Feature Correlation Heatmap")
        if has_data(correlation_matrix):
            heatmap = px.imshow(
                correlation_matrix.round(2),
                text_auto=".2f",
                color_continuous_scale="Tealgrn",
                aspect="auto",
                title="Heatmap Korelasi Fitur dan Target",
            )
            heatmap.update_layout(coloraxis_colorbar={"title": "Korelasi"}, **PLOTLY_DARK_LAYOUT)
            st.plotly_chart(heatmap, use_container_width=True)
        else:
            render_empty_state("Heatmap korelasi belum tersedia.")
        st.caption(RESULTS_CAPTION)

        with st.expander("Keterangan metodologi penelitian", expanded=True):
            st.markdown(
                """
                - Evaluasi utama dilakukan hanya pada data test set.
                - Simulasi future horizon hanya bersifat eksploratif.
                - Model yang dibandingkan adalah LSTM standalone, XGBoost standalone, dan Hybrid Stacking LSTM-XGBoost.
                - Train-test split dilakukan secara sequential tanpa shuffle.
                - MinMaxScaler hanya di-fit pada data training untuk mencegah data leakage.
                - Sequence LSTM dibentuk dengan metode sliding window untuk menjaga dependensi temporal time series.
                """
            )
