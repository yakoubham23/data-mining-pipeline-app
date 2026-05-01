from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.clustering import (
	compute_elbow_curve,
	compute_silhouette,
	prepare_features_for_clustering,
	reduce_with_pca,
	run_clustering,
	compare_algorithms,
)
from utils.preprocessing import detect_column_types, load_csv_dataset
from utils.visualization import (
	create_cluster_2d_figure,
	create_cluster_distribution_figure,
	create_elbow_figure,
	create_comparison_charts,
)

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
st.caption("Run K-Means, K-Medoids, AGNES, DIANA, or DBSCAN with performance comparison.")

if "best_k" not in st.session_state:
	st.session_state.best_k = 3

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
overview_cols = st.columns(4)
overview_cols[0].metric("Rows", len(df))
overview_cols[1].metric("Columns", len(df.columns))
overview_cols[2].metric("Numeric features", len(numeric_cols))
overview_cols[3].metric("Categorical features", len(categorical_cols))

if len(df) < 3:
	st.error("Clustering requires at least 3 rows.")
	st.stop()

if len(numeric_cols) < 2:
	st.error("Clustering requires at least 2 numeric columns.")
	st.stop()

with st.expander("⚙️ Clustering Configuration", expanded=True):
	feature_columns = st.multiselect(
		"Numeric features for clustering",
		options=numeric_cols,
		default=numeric_cols[: min(5, len(numeric_cols))],
	)

	config_cols = st.columns(3)
	algorithm = config_cols[0].selectbox("Algorithm", options=ALGORITHMS)
	
	dbscan_params = {}
	if algorithm == "DBSCAN":
		dbscan_params["eps"] = config_cols[1].number_input("Epsilon (eps)", min_value=0.1, value=0.5, step=0.1)
		dbscan_params["min_samples"] = config_cols[2].number_input("Min Samples", min_value=1, value=5, step=1)
		n_clusters = 0 # Not used for DBSCAN
	else:
		max_clusters = min(10, len(df) - 1)
		n_clusters = config_cols[1].slider("Number of clusters (k)", min_value=2, max_value=max_clusters, value=st.session_state.best_k)
		random_state = config_cols[2].number_input("Random state", min_value=0, value=42, step=1)

	btn_cols = st.columns(2)
	run_clustering_btn = btn_cols[0].button("🚀 Run clustering", use_container_width=True)
	compare_btn = btn_cols[1].button("📊 Compare All Algorithms", use_container_width=True)

if run_clustering_btn:
	try:
		scaled_df, _ = prepare_features_for_clustering(df, feature_columns=feature_columns)
		
		# Auto Elbow Suggestion
		elbow_df, best_k = compute_elbow_curve(
			scaled_df,
			algorithm="K-Means", # Usually K-Means is used for elbow
			min_k=2,
			max_k=min(10, len(scaled_df) - 1),
			random_state=42,
		)
		if best_k:
			st.session_state.best_k = best_k
			st.info(f"💡 The Elbow method suggests an optimal k = {best_k}")

		labels, model, exec_time = run_clustering(
			scaled_df,
			algorithm=algorithm,
			n_clusters=n_clusters,
			random_state=int(random_state) if algorithm != "DBSCAN" else 42,
			**dbscan_params
		)
		
		silhouette = compute_silhouette(scaled_df, labels)
		pca_df = reduce_with_pca(scaled_df, n_components=2)
		pca_df["cluster"] = labels.astype(str)

		clustered_df = df.copy()
		clustered_df["cluster"] = labels
		st.session_state.clustered_df = clustered_df
		st.session_state.cluster_model = model

		st.subheader(f"📊 {algorithm} Results")
		metric_cols = st.columns(4)
		metric_cols[0].metric("Silhouette Score", f"{silhouette:.4f}")
		metric_cols[1].metric("Clusters Found", len([c for c in labels.unique() if c != -1]))
		metric_cols[2].metric("Execution Time", f"{exec_time:.4f}s")
		metric_cols[3].metric("Algorithm", algorithm)

		if algorithm != "DBSCAN":
			elbow_fig = create_elbow_figure(elbow_df, algorithm_name="K-Means (Reference)")
			st.plotly_chart(elbow_fig, use_container_width=True)

		cluster_fig = create_cluster_2d_figure(pca_df, cluster_col="cluster")
		st.plotly_chart(cluster_fig, use_container_width=True)

		distribution_fig = create_cluster_distribution_figure(labels)
		st.plotly_chart(distribution_fig, use_container_width=True)

		st.subheader("🧾 Clustered Dataset Preview")
		st.dataframe(clustered_df.head(25), use_container_width=True)

		st.download_button(
			"Download clustered dataset",
			data=clustered_df.to_csv(index=False).encode("utf-8"),
			file_name=f"clustered_{algorithm.lower().replace(' ', '_')}.csv",
			mime="text/csv",
			use_container_width=True,
		)
	except (ValueError, ImportError) as exc:
		st.error(str(exc))

if compare_btn:
	try:
		scaled_df, _ = prepare_features_for_clustering(df, feature_columns=feature_columns)
		comparison_df = compare_algorithms(scaled_df, ALGORITHMS, n_clusters=n_clusters)
		
		st.subheader("🏁 Performance Comparison")
		st.dataframe(comparison_df, use_container_width=True)
		
		time_fig, sil_fig = create_comparison_charts(comparison_df)
		chart_cols = st.columns(2)
		chart_cols[0].plotly_chart(time_fig, use_container_width=True)
		chart_cols[1].plotly_chart(sil_fig, use_container_width=True)
		
	except Exception as exc:
		st.error(f"Comparison failed: {exc}")
