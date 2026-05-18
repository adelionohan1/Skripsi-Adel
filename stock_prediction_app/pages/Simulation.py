import plotly.graph_objects as go
import streamlit as st

from utils.ui import PLOTLY_DARK_LAYOUT, dataframe_to_csv, get_results, has_data, make_download_filename, render_empty_state, render_metric_card


results = get_results()
simulation_df = results["simulation"]
HYBRID_COLUMN = "Hybrid Stacking LSTM-XGBoost"

st.title("Simulation")
st.markdown("#### Simulasi (Bukan Evaluasi Utama)")
st.warning("Simulasi menggunakan horizon prediksi dan hanya untuk tujuan eksplorasi. Hasil prediksi 30 hari ke depan tidak digunakan sebagai bukti utama evaluasi model.")

if not has_data(simulation_df):
    render_empty_state()
else:
    metric_cols = st.columns(3)
    render_metric_card(metric_cols[0], "Prediksi Akhir LSTM", f"{simulation_df['LSTM'].iloc[-1]:.2f}")
    render_metric_card(metric_cols[1], "Prediksi Akhir XGBoost", f"{simulation_df['XGBoost'].iloc[-1]:.2f}")
    if HYBRID_COLUMN in simulation_df.columns:
        render_metric_card(metric_cols[2], "Prediksi Akhir Hybrid", f"{simulation_df[HYBRID_COLUMN].iloc[-1]:.2f}", tone="positive")
    else:
        render_metric_card(metric_cols[2], "Prediksi Akhir Hybrid", "-", tone="neutral")

    summary_tab, detail_tab, visual_tab = st.tabs(["Ringkasan", "Detail Model", "Visualisasi"])

    with summary_tab:
        simulation_chart = go.Figure()
        simulation_chart.add_trace(
            go.Scatter(x=simulation_df["Date"], y=simulation_df["LSTM"], mode="lines+markers", name="LSTM", line={"color": "#38bdf8"})
        )
        simulation_chart.add_trace(
            go.Scatter(x=simulation_df["Date"], y=simulation_df["XGBoost"], mode="lines+markers", name="XGBoost", line={"color": "#22c55e"})
        )
        if HYBRID_COLUMN in simulation_df.columns:
            simulation_chart.add_trace(
                go.Scatter(
                    x=simulation_df["Date"],
                    y=simulation_df[HYBRID_COLUMN],
                    mode="lines+markers",
                    name=HYBRID_COLUMN,
                    line={"color": "#f59e0b"},
                )
            )
        simulation_chart.update_layout(
            title="Simulasi Prediksi Future Horizon",
            xaxis_title="Tanggal",
            yaxis_title="Harga Prediksi",
            hovermode="x unified",
            **PLOTLY_DARK_LAYOUT,
        )
        st.plotly_chart(simulation_chart, use_container_width=True)
        st.download_button(
            "Download Simulation CSV",
            data=dataframe_to_csv(simulation_df),
            file_name=make_download_filename("simulation", results["params"]["ticker"]),
            mime="text/csv",
            use_container_width=True,
        )

    with detail_tab:
        with st.expander("Informasi model", expanded=True):
            st.markdown(
                """
                `# bagian ini sesuai metodologi penelitian skripsi`

                Simulasi ini tidak digunakan sebagai dasar evaluasi model.
                Tujuannya hanya untuk eksplorasi perilaku prediksi beberapa langkah ke depan.
                Hybrid simulation tetap mengikuti konsep stacking: prediksi LSTM digunakan sebagai fitur tambahan untuk XGBoost meta-learner.
                """
            )

    with visual_tab:
        st.dataframe(simulation_df, use_container_width=True, hide_index=True)

st.caption("Sumber: Hasil Implementasi Sistem, 2026")
