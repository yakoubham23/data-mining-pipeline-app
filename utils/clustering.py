"""Clustering utility functions for unsupervised learning workflows."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
	from sklearn_extra.cluster import KMedoids
except Exception:  # noqa: BLE001
	KMedoids = None


def prepare_features_for_clustering(
	df: pd.DataFrame,
	feature_columns: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
	"""Select, clean, and scale numeric features for clustering."""
	if len(feature_columns) < 2:
		raise ValueError("Select at least two numeric features for clustering.")

	invalid_cols = [col for col in feature_columns if col not in df.columns]
	if invalid_cols:
		raise ValueError(f"Invalid feature columns: {invalid_cols}")

	selected_df = df[feature_columns].copy()
	non_numeric = selected_df.select_dtypes(exclude=["number"]).columns.tolist()
	if non_numeric:
		raise ValueError(f"Clustering requires numeric columns only: {non_numeric}")

	filled_df = selected_df.fillna(selected_df.median(numeric_only=True)).fillna(0)
	scaler = StandardScaler()
	scaled_array = scaler.fit_transform(filled_df)
	scaled_df = pd.DataFrame(scaled_array, columns=feature_columns, index=df.index)
	return scaled_df, scaler


def _create_model(algorithm: str, n_clusters: int, random_state: int):
	if algorithm == "K-Means":
		return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

	if algorithm == "K-Medoids":
		if KMedoids is None:
			raise ImportError(
				"K-Medoids requires scikit-learn-extra. Install it with: pip install scikit-learn-extra"
			)
		return KMedoids(n_clusters=n_clusters, random_state=random_state, init="k-medoids++")

	raise ValueError("Unsupported clustering algorithm.")


def compute_elbow_curve(
	scaled_df: pd.DataFrame,
	algorithm: str,
	min_k: int = 2,
	max_k: int = 10,
	random_state: int = 42,
) -> pd.DataFrame:
	"""Compute inertia values for a range of k values (elbow method)."""
	if len(scaled_df) < 3:
		raise ValueError("At least 3 rows are required to compute the elbow curve.")

	max_valid_k = min(max_k, len(scaled_df) - 1)
	if max_valid_k < min_k:
		raise ValueError("Dataset is too small for the selected elbow k-range.")

	records: list[dict[str, float | int]] = []
	for k in range(min_k, max_valid_k + 1):
		model = _create_model(algorithm=algorithm, n_clusters=k, random_state=random_state)
		model.fit(scaled_df)
		records.append({"k": k, "inertia": float(model.inertia_)})

	return pd.DataFrame(records)


def run_clustering(
	scaled_df: pd.DataFrame,
	algorithm: str,
	n_clusters: int,
	random_state: int = 42,
) -> tuple[pd.Series, object]:
	"""Fit the selected clustering model and return cluster labels."""
	if n_clusters < 2:
		raise ValueError("Number of clusters must be at least 2.")
	if n_clusters >= len(scaled_df):
		raise ValueError("Number of clusters must be smaller than the number of samples.")

	model = _create_model(algorithm=algorithm, n_clusters=n_clusters, random_state=random_state)
	labels = model.fit_predict(scaled_df)
	labels_series = pd.Series(labels, index=scaled_df.index, name="cluster")
	return labels_series, model


def compute_silhouette(scaled_df: pd.DataFrame, labels: pd.Series) -> float:
	"""Compute silhouette score for the current clustering labels."""
	unique_clusters = labels.nunique()
	if unique_clusters < 2:
		raise ValueError("Silhouette score requires at least 2 clusters.")
	return float(silhouette_score(scaled_df, labels))


def reduce_with_pca(scaled_df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
	"""Project scaled data to 2D for plotting."""
	pca = PCA(n_components=n_components, random_state=42)
	components = pca.fit_transform(scaled_df)
	return pd.DataFrame(components, columns=["PC1", "PC2"], index=scaled_df.index)
