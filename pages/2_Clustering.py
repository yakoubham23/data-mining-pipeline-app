from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.clustering import (
	compute_elbow_curve,
	compute_silhouette,
	prepare_features_for_clustering,
	reduce_with_pca,
	run_clustering,
)
from utils.preprocessing import detect_column_types, load_csv_dataset
from utils.visualization import (
	create_cluster_2d_figure,
	create_cluster_distribution_figure,
	create_elbow_figure,
)


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
st.caption("Run K-Means or K-Medoids with elbow analysis and PCA visualization.")

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
	algorithm = config_cols[0].selectbox("Algorithm", options=["K-Means", "K-Medoids"])
	max_clusters = min(10, len(df) - 1)
	n_clusters = config_cols[1].slider("Number of clusters (k)", min_value=2, max_value=max_clusters, value=3)
	random_state = config_cols[2].number_input("Random state", min_value=0, value=42, step=1)

	run_clustering_btn = st.button("🚀 Run clustering", use_container_width=True)

if run_clustering_btn:
	try:
		scaled_df, _ = prepare_features_for_clustering(df, feature_columns=feature_columns)
		elbow_df = compute_elbow_curve(
			scaled_df,
			algorithm=algorithm,
			min_k=2,
			max_k=min(10, len(scaled_df) - 1),
			random_state=int(random_state),
		)
		labels, model = run_clustering(
			scaled_df,
			algorithm=algorithm,
			n_clusters=n_clusters,
			random_state=int(random_state),
		)
		silhouette = compute_silhouette(scaled_df, labels)
		pca_df = reduce_with_pca(scaled_df, n_components=2)
		pca_df["cluster"] = labels.astype(str)

		clustered_df = df.copy()
		clustered_df["cluster"] = labels
		st.session_state.clustered_df = clustered_df
		st.session_state.cluster_model = model

		st.subheader("📊 Clustering Results")
		metric_cols = st.columns(3)
		metric_cols[0].metric("Silhouette Score", f"{silhouette:.4f}")
		metric_cols[1].metric("Clusters", int(labels.nunique()))
		metric_cols[2].metric("Algorithm", algorithm)

		elbow_fig = create_elbow_figure(elbow_df, algorithm_name=algorithm)
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
			file_name="clustered_dataset.csv",
			mime="text/csv",
			use_container_width=True,
		)
	except (ValueError, ImportError) as exc:
		st.error(str(exc))
