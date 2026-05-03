from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.clustering import (
	compute_elbow_curve,
	compute_silhouette,
	prepare_features_for_clustering,
	reduce_with_pca,
	run_clustering,
	compare_algorithms,
	suggest_dbscan_params,
)
from utils.preprocessing import detect_column_types, load_csv_dataset
from utils.visualization import (
	create_cluster_2d_figure,
	create_cluster_distribution_figure,
	create_elbow_figure,
	create_comparison_charts,
	create_dendrogram_figure,
)
import plotly.express as px

ALGORITHMS = ["K-Means", "K-Medoids", "AGNES (Agglomerative)", "DIANA (Divisive)", "DBSCAN"]

def _load_dataset_from_source(source_name: str):
	sample_path = Path(__file__).resolve().parents[1] / "data" / "sample.csv"

	if source_name == "Use processed dataset from preprocessing":
		if st.session_state.get("processed_df") is None:
			st.warning("No processed dataset found. Go to the Preprocessing page first.")
			return None
		return st.session_state.processed_df.copy()

	if source_name == "Upload new CSV":
		uploaded_file = st.file_uploader("Upload CSV for clustering", type=["csv"], key="cluster_uploader")
		if uploaded_file is None:
			return None
		try:
			return load_csv_dataset(uploaded_file)
		except ValueError as exc:
			st.error(str(exc))
			return None

	try:
		return load_csv_dataset(sample_path)
	except ValueError as exc:
		st.error(str(exc))
		return None


st.title("🧠 Unsupervised Learning - Clustering")
st.caption("Run K-Means, K-Medoids, AGNES, DIANA, or DBSCAN and compare results.")

if "best_k" not in st.session_state:
	st.session_state.best_k = 3

if "elbow_df" not in st.session_state:
	st.session_state.elbow_df = None

if "comparison_history" not in st.session_state:
	st.session_state.comparison_history = []

with st.expander("📥 Dataset Selection", expanded=True):
	source = st.radio(
		"Choose data source",
		options=[
			"Use processed dataset from preprocessing",
			"Upload new CSV",
			"Use sample dataset",
		],
		horizontal=True,
	)

	df = _load_dataset_from_source(source)

if df is None:
	st.info("Select a dataset source to start clustering.")
	st.stop()

numeric_cols, categorical_cols = detect_column_types(df)

if len(df) < 3:
	st.error("Clustering requires at least 3 rows.")
	st.stop()

if len(numeric_cols) < 2:
	st.error("Clustering requires at least 2 numeric columns.")
	st.stop()

# --- STEP 1: ANALYSIS ---
st.subheader("Step 1: 🔍 Analyze Optimal Number of Clusters (Elbow Method)")
with st.expander("📊 Elbow Analysis Configuration", expanded=True):
	analysis_cols = st.columns([2, 1])
	feature_columns = analysis_cols[0].multiselect(
		"Features for analysis",
		options=numeric_cols,
		default=numeric_cols[: min(5, len(numeric_cols))],
		key="analysis_features"
	)
	
	analyze_btn = analysis_cols[1].button("📉 Run Elbow Analysis", width='stretch')

if analyze_btn:
	try:
		scaled_df_analysis, _ = prepare_features_for_clustering(df, feature_columns=feature_columns)
		elbow_df, best_k = compute_elbow_curve(
			scaled_df_analysis,
			algorithm="K-Means",
			min_k=2,
			max_k=min(10, len(scaled_df_analysis) - 1),
			random_state=42,
		)
		st.session_state.elbow_df = elbow_df
		if best_k:
			st.session_state.best_k = best_k
	except Exception as exc:
		st.error(f"Analysis failed: {exc}")

if st.session_state.elbow_df is not None:
	st.info(f"💡 The Elbow method suggests an optimal **k = {st.session_state.best_k}**")
	elbow_fig = create_elbow_figure(st.session_state.elbow_df, algorithm_name="K-Means (Analysis)")
	st.plotly_chart(elbow_fig, width='stretch')

# --- STEP 2: CLUSTERING ---
st.subheader("Step 2: 🚀 Run Clustering Algorithm")
with st.expander("⚙️ Execution Configuration", expanded=True):
	config_cols = st.columns(3)
	algorithm = config_cols[0].selectbox("Algorithm", options=ALGORITHMS)
	
	random_state = 42
	dbscan_params = {}
	if algorithm == "DBSCAN":
		try:
			scaled_df_temp, _ = prepare_features_for_clustering(df, feature_columns=feature_columns)
			s_eps, s_min, k_dist_df = suggest_dbscan_params(scaled_df_temp)
			st.info(f"💡 DBSCAN Suggestion: eps={s_eps:.3f}, min_samples={s_min}")
			
			with st.expander("📈 k-distance graph", expanded=False):
				k_fig = px.line(k_dist_df, x="index", y="distance", title="k-distance Graph")
				k_fig.add_hline(y=s_eps, line_dash="dash", line_color="red")
				st.plotly_chart(k_fig, width='stretch')
			eps_val, min_samples_val = s_eps, s_min
		except Exception:
			eps_val, min_samples_val = 0.5, 5

		dbscan_params["eps"] = config_cols[1].number_input("Epsilon (eps)", min_value=0.01, value=float(eps_val), step=0.05, format="%.3f")
		dbscan_params["min_samples"] = config_cols[2].number_input("Min Samples", min_value=1, value=int(min_samples_val), step=1)
		n_clusters = 0
	else:
		max_clusters = min(10, len(df) - 1)
		n_clusters = config_cols[1].slider("Number of clusters (k)", min_value=2, max_value=max_clusters, value=st.session_state.best_k)
		random_state = config_cols[2].number_input("Random state", min_value=0, value=42, step=1)

	run_clustering_btn = st.button("✨ Apply Clustering", width='stretch')

if run_clustering_btn:
	try:
		scaled_df, _ = prepare_features_for_clustering(df, feature_columns=feature_columns)
		labels, model, exec_time, inertia = run_clustering(
			scaled_df,
			algorithm=algorithm,
			n_clusters=n_clusters,
			random_state=int(random_state),
			**dbscan_params
		)
		
		silhouette = compute_silhouette(scaled_df, labels)
		pca_df = reduce_with_pca(scaled_df, n_components=2)
		pca_df["cluster"] = labels.astype(str)

		clustered_df = df.copy()
		clustered_df["cluster"] = labels
		st.session_state.clustered_df = clustered_df
		st.session_state.cluster_model = model
		
		params_str = f"k={n_clusters}" if algorithm != "DBSCAN" else f"eps={dbscan_params['eps']}"
		st.session_state.comparison_history.append({
			"Algorithm": algorithm,
			"Parameters": params_str,
			"Execution Time (s)": round(exec_time, 6),
			"Silhouette Score": round(silhouette, 4),
			"Inertia": round(inertia, 2) if inertia is not None else "N/A",
			"Clusters Found": len([c for c in labels.unique() if c != -1])
		})

		st.subheader(f"📊 {algorithm} Final Results")
		m_cols = st.columns(4)
		m_cols[0].metric("Silhouette", f"{silhouette:.4f}")
		m_cols[1].metric("Clusters", len([c for c in labels.unique() if c != -1]))
		m_cols[2].metric("Inertia", f"{inertia:.2f}" if inertia is not None else "N/A")
		m_cols[3].metric("Time", f"{exec_time:.4f}s")

		c_fig = create_cluster_2d_figure(pca_df, cluster_col="cluster")
		st.plotly_chart(c_fig, width='stretch')

		if algorithm in ["AGNES (Agglomerative)", "DIANA (Divisive)"]:
			st.subheader("🌲 Dendrogram")
			st.pyplot(create_dendrogram_figure(scaled_df))

		st.download_button("Download Result CSV", clustered_df.to_csv(index=False).encode("utf-8"), "clustered.csv", "text/csv", width='stretch')
	except Exception as exc:
		st.error(str(exc))

# Comparison Section - Only shows executed algorithms
if st.session_state.comparison_history:
	st.divider()
	st.subheader("🏁 Performance Comparison (Executed Algorithms)")
	
	comp_df = pd.DataFrame(st.session_state.comparison_history)
	st.dataframe(comp_df, width='stretch')
	
	time_fig, sil_fig = create_comparison_charts(comp_df)
	chart_cols = st.columns(2)
	chart_cols[0].plotly_chart(time_fig, width='stretch')
	chart_cols[1].plotly_chart(sil_fig, width='stretch')
	
	if st.button("🗑️ Clear Comparison History", width='stretch'):
		st.session_state.comparison_history = []
		st.rerun()
