import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spectra_utils as su
import streamlit as st

st.set_page_config(page_title="Raman Spectra Matching", layout="wide")
st.title("Raman Spectra Analysis & Matching")
st.write(
    "The aim of this site is to provide the possibility to match Raman spectra"
    " of unkown samples, polymers and other materials with a database called"
    " 'SLoPP and SLoPP-E', for further details about the database and the"
    " material available, please refer to the [Rochman Lab official"
    " website](https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/)."
    " Lately we had some trouble because of the high density presence of pigments inside the samples, so we added a new database of modern pigments"
    " from [CHSOS](https://chsopensource.org/)"
)

st.sidebar.header("Analysis settings")

st.sidebar.markdown(
    "Use the options below to customize the analysis. You can adjust the"
    " number of top matches displayed, set the minimum wavenumber cutoff,"
    " and choose whether to apply smoothing to the spectra before matching."
    " If you choose to apply smoothing, you can also select the lambda value"
    " for the Whittaker smoothing algorithm, here is used [Rampy"
    " libraries](https://rampy.readthedocs.io/en/stable/) and [Whittaker"
    " smoothing publication](https://pubs.acs.org/doi/10.1021/ac034173t). It"
    " is also applied a baseline correction based on [arPLS"
    " algorithm](https://doi.org/10.1039/C4AN01061B)."
)

top_n_slider = st.sidebar.slider(
    "Number of Top Match (TOP_N)", min_value=1, max_value=10, value=3
)

# === NUOVA SEZIONE: Maschera per il Wavenumber minimo ===
min_wavenumber = st.sidebar.slider(
    "Minimum Wavenumber Cutoff [cm⁻¹]",
    min_value=100,
    max_value=1000,
    value=250,
    step=10,
)

apply_smooth = st.sidebar.checkbox("Apply Smoothing Whittaker", value=True)

if apply_smooth:
    whittaker_lambda = st.sidebar.select_slider(
        "Lambda value for smoothing",
        options=[1, 10, 50, 100, 500, 1000, 5000, 10000],
        value=1000,
    )
else:
    whittaker_lambda = 1000

transform_option = st.sidebar.radio(
    "Mathematical Transformation",
    options=[
        "Adaptive (Square root only if peak > 2400 cm⁻¹)",
        "Force Square Root on all",
    ],
    index=0,
)

st.subheader("Upload Query files")

uploaded_files = st.file_uploader(
    "Here to upload your query spectra files (txt format, with two columns:"
    " wavenumber and intensity). You can upload multiple files at once. In"
    " order to better calibrate the spectra is used to use the peak of the"
    " silicon at 520 cm-1, so the spectra should contain this peak also for"
    " different laser wavelength. If you have different spectra for the same"
    " sample, please make sure to name them with the same base name (e.g.,"
    " sample1_01.txt, sample1_02.txt) so they will be grouped together in the"
    " analysis, please refer to the documentation in the github repository for"
    " more information about the concatenation and preprocessing.",
    type=["txt"],
    accept_multiple_files=True,
)

if uploaded_files:

    def patched_preprocess_xy_custom(x, y, apply_smooth, lam=whittaker_lambda):
        if x.size < 3:
            raise ValueError("Not enough points in dataset")

        # Applicazione della maschera minima configurata
        mask = x >= min_wavenumber
        x = x[mask]
        y = y[mask]

        baseline, y_corr = su.baseline_correct(x, y)

        if apply_smooth:
            y_final = su.smooth_whittaker(x, y_corr, lam=lam)
        else:
            y_final = y_corr

        return x, y_final

    su.preprocess_xy_custom = patched_preprocess_xy_custom

    with tempfile.TemporaryDirectory() as temp_dir:
        query_dir_path = Path(temp_dir)

        for uploaded_file in uploaded_files:
            with open(query_dir_path / uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

        with st.spinner("Loading database from cache..."):
            try:
                db_items = su.get_db_spectra_cache(False)
            except Exception as e:
                st.error(f"Error loading database: {e}")
                st.stop()

        rows = []
        TOP_N = top_n_slider

        st.info("Processing and matching spectra...")
        progress_bar = st.progress(0.0)

        all_files = list(query_dir_path.rglob("*.txt"))
        grouped_queries = su.group_files_by_base(all_files)
        valid_queries = [
            k for k in grouped_queries.keys() if not su.is_si_file(Path(k))
        ]
        total_queries = len(valid_queries)

        query_generator = su.load_query_spectra(query_dir_path, apply_smooth)

        for index, (key, qx, qy) in enumerate(query_generator):
            if total_queries > 0:
                progress_bar.progress(min((index + 1) / total_queries, 1.0))

            # Utilizzo del valore selezionato dallo slider
            MIN_X = min_wavenumber
            mask = qx >= MIN_X
            qx, qy = qx[mask], qy[mask]

            scores_cosine = []

            for path, dx, dy in db_items:
                resampled = su.resample_overlap(qx, qy, dx, dy)
                if resampled is None:
                    continue

                yq, yd, xreal = resampled

                if transform_option == "Force Square Root on all":
                    use_square_root = True
                else:
                    if np.max(xreal) <= 2500:
                        has_strong_peak = False
                    else:
                        peak_mask = (xreal >= 2400) & (xreal <= 3200)
                        if np.any(peak_mask):
                            y_sub = yq[peak_mask]
                            global_max = np.max(yq)
                            peaks, _ = su.find_peaks(
                                y_sub,
                                height=0.5 * np.max(y_sub),
                                prominence=0.1 * np.max(y_sub),
                            )
                            peak_strength = (
                                (np.max(y_sub) / global_max)
                                if global_max > 0
                                else 0
                            )
                            has_strong_peak = (
                                len(peaks) > 0 and peak_strength > 0.25
                            )
                        else:
                            has_strong_peak = False

                    use_square_root = has_strong_peak

                if use_square_root:
                    yq, yd = su.square_root_transform(yq, yd)
                else:
                    yq, yd = su.power_transformation(yq, yd)

                if yq is None:
                    continue

                norm_q = np.linalg.norm(yq)
                norm_d = np.linalg.norm(yd)
                yq_norm = yq / norm_q if norm_q > 0 else yq
                yd_norm = yd / norm_d if norm_d > 0 else yd

                score_cos = su.cosine_similarity(yq, yd)
                score_pea = su.pearson_correlation(yq, yd)

                scores_cosine.append(
                    (score_cos, score_pea, path, yq_norm, yd_norm, xreal)
                )

            scores_cosine.sort(key=lambda item: item[0], reverse=True)

            unique_matches = []
            seen_parents = set()

            for s_cos, s_pea, path, yq_vec, yd_vec, grid in scores_cosine:
                parent_name = path.parent.name
                if parent_name not in seen_parents:
                    seen_parents.add(parent_name)
                    unique_matches.append(
                        (s_cos, s_pea, path, yq_vec, yd_vec, grid)
                    )
                if len(unique_matches) == TOP_N:
                    break

            if not unique_matches:
                continue

            # Render UI for query matches
            with st.expander(
                f"View Matching Results for: **{key}**", expanded=False
            ):
                tabs = st.tabs(
                    [
                        f"Rank {rank}"
                        for rank in range(1, len(unique_matches) + 1)
                    ]
                )

                for rank, (
                    best_cos,
                    best_pea,
                    best_path,
                    yq_plot,
                    yd_plot,
                    x_plot,
                ) in enumerate(unique_matches, start=1):
                    # Append result row for final dataframe export
                    rows.append(
                        [
                            key,
                            f"{best_cos:.6f}",
                            f"{best_pea:.6f}",
                            best_path.parent.name,
                            best_path.name,
                            f"Rank {rank}",
                        ]
                    )

                    # Plot tab contents
                    with tabs[rank - 1]:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=x_plot,
                                y=yq_plot,
                                mode="lines",
                                name=f"Query: {key}",
                                line=dict(color="blue", width=2),
                            )
                        )

                        legend_text = (
                            f"DB Match: {best_path.name}<br>(Cos:"
                            f" {best_cos:.4f}, Pear: {best_pea:.4f})"
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x_plot,
                                y=yd_plot,
                                mode="lines",
                                name=legend_text,
                                line=dict(
                                    color="red", width=1.5, dash="dash"
                                ),
                                opacity=0.8,
                            )
                        )

                        fig.update_layout(
                            title=(
                                f"Rank {rank}: {key} vs {best_path.parent.name}"
                            ),
                            xaxis_title="Wavenumber [cm⁻¹]",
                            yaxis_title="Intensity (Normalized)",
                            template="plotly_white",
                            hovermode="x unified",
                            height=400,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,
                                bgcolor="rgba(255, 255, 255, 0.7)",
                            ),
                        )

                        st.plotly_chart(fig, use_container_width=True)

        progress_bar.empty()
        st.success("Processing completed successfully!")

        if rows:
            st.subheader("Ranking Summary")
            df_results = pd.DataFrame(
                rows,
                columns=[
                    "Query",
                    "Cosine Similarity",
                    "Pearson Correlation",
                    "DB Folder",
                    "DB File Name",
                    "Rank",
                ],
            )
            st.dataframe(df_results, use_container_width=True)

            csv_buffer = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results CSV",
                data=csv_buffer,
                file_name="raman_matching_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
