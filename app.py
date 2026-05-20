import streamlit as st
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import spectra_utils as su

st.set_page_config(page_title="Raman Spectra Matching", layout="wide", page_icon="🔬")

st.title("Raman Spectra Analysis & Matching")
st.write("Upload here the file raw.")

st.sidebar.header("Analysis settings")
top_n_slider = st.sidebar.slider("Number of Top Match (TOP_N)", min_value=1, max_value=10, value=3)
apply_smooth = st.sidebar.checkbox("Apply Smoothing Whittaker", value=True)

if apply_smooth:
    whittaker_lambda = st.sidebar.select_slider(
        "Lambda value for smoothing",
        options=[1, 10, 50, 100, 500, 1000, 5000, 10000],
        value=100,
    )
else:
    whittaker_lambda = None

st.subheader("1. Upload Query files")
uploaded_files = st.file_uploader(
    "Put here your .txt (also 'si' calibration files)", 
    type=["txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    with tempfile.TemporaryDirectory() as temp_dir:
        query_dir_path = Path(temp_dir)
        
        for uploaded_file in uploaded_files:
            with open(query_dir_path / uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        with st.spinner("Database charging via cache..."):
            try:
                db_items = su.get_db_spectra_cache(False)
            except Exception as e:
                st.error(f"Error in Database Charging: {e}.")
                st.stop()

        rows = []
        TOP_N = top_n_slider

        st.info("Processing and matching spectra...")
        progress_bar = st.progress(0.0)
        
        all_files = list(query_dir_path.rglob("*.txt"))
        grouped_queries = su.group_files_by_base(all_files)
        valid_queries = [k for k in grouped_queries.keys() if not su.is_si_file(Path(k))]
        total_queries = len(valid_queries)

        query_generator = su.load_query_spectra(query_dir_path, apply_smooth)

        for index, (key, qx, qy) in enumerate(query_generator):
            if total_queries > 0:
                progress_bar.progress(min((index + 1) / total_queries, 1.0))
            
            MIN_X = 250
            mask = (qx >= MIN_X)
            qx, qy = qx[mask], qy[mask]

            scores_cosine = []

            for path, dx, dy in db_items:
                resampled = su.resample_overlap(qx, qy, dx, dy)
                if resampled is None:
                    continue

                yq, yd, xreal = resampled

                if np.max(xreal) <= 2500:
                    has_strong_peak = False
                else:
                    peak_mask = (xreal >= 2400) & (xreal <= 3200)
                    if np.any(peak_mask):
                        x_sub = xreal[peak_mask]
                        y_sub = yq[peak_mask]

                        if np.max(x_sub) >= 2400:
                            global_max = np.max(yq)
                            peaks, _ = su.find_peaks(
                                y_sub,
                                height=0.5 * np.max(y_sub),
                                prominence=0.1 * np.max(y_sub)
                            )
                            peak_strength = np.max(y_sub) / global_max
                            has_strong_peak = (len(peaks) > 0 and peak_strength > 0.25)
                        else:
                            has_strong_peak = False
                    else:
                        has_strong_peak = False

                if has_strong_peak:
                    yq, yd = su.square_root_transform(yq, yd)
                else:
                    yq, yd = su.power_transformation(yq, yd)

                if yq is None:
                    continue

                norm_q = np.linalg.norm(yq)
                norm_d = np.linalg.norm(yd)
                yq_norm = yq / norm_q
                yd_norm = yd / norm_d 

                score_cos = su.cosine_similarity(yq, yd)
                score_pea = su.pearson_correlation(yq, yd)

                scores_cosine.append((score_cos, score_pea, path, yq_norm, yd_norm, xreal))

            scores_cosine.sort(key=lambda item: item[0], reverse=True)

            unique_matches = []
            seen_parents = set()

            for s_cos, s_pea, path, yq_vec, yd_vec, grid in scores_cosine:
                parent_name = path.parent.name
                if parent_name not in seen_parents:
                    seen_parents.add(parent_name)
                    unique_matches.append((s_cos, s_pea, path, yq_vec, yd_vec, grid))
                if len(unique_matches) == TOP_N:
                    break

            if not unique_matches:
                continue

            best_cos, best_pea, best_path, yq_plot, yd_plot, x_plot = unique_matches[0]

            with st.expander(f"View Matching Results for: {key}"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_plot,
                    y=yq_plot,
                    mode='lines',
                    name=f"Query: {key}",
                    line=dict(color='blue', width=2)
                ))

                legend_text = f"DB Top Match: {best_path.name}<br>(Cos: {best_cos:.4f}, Pear: {best_pea:.4f})"
                fig.add_trace(go.Scatter(
                    x=x_plot,
                    y=yd_plot,
                    mode='lines',
                    name=legend_text,
                    line=dict(color='red', width=1.5, dash='dash'),
                    opacity=0.8
                ))

                fig.update_layout(
                    title=f"Spectral Comparison: {key} vs {best_path.parent.name}",
                    xaxis_title="Relative Spectral Range (Interpolated) [cm⁻¹]",
                    yaxis_title="Intensity (Normalized)",
                    template="plotly_white",
                    hovermode="x unified",
                    height=400,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.7)")
                )
                
                st.plotly_chart(fig, use_container_width=True)

            for rank, (s_cos, s_pea, path, _, _, _) in enumerate(unique_matches, start=1):
                rows.append([
                    key, f"{s_cos:.6f}", f"{s_pea:.6f}", path.parent.name, path.name, f"Rank {rank}"
                ])

        progress_bar.empty()
        st.success("Processing completed successfully!")

        if rows:
            st.subheader("Ranking table")
            df_results = pd.DataFrame(rows, columns=[
                "Query", "Cosine Similarity", "Pearson Correlation", "DB Folder", "DB File Name", "Rank"
            ])
            st.dataframe(df_results, use_container_width=True)

            csv_buffer = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv_buffer,
                file_name="matching_raman.csv",
                mime="text/csv"
            )
