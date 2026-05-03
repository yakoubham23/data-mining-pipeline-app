"""Clustering utility functions for unsupervised learning workflows."""

from __future__ import annotations

import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

try:
	from sklearn_extra.cluster import KMedoids
except Exception:  # noqa: BLE001
	KMedoids = None


from sklearn.neighbors import NearestNeighbors

def suggest_dbscan_params(scaled_df: pd.DataFrame) -> tuple[float, int, pd.DataFrame]:
	"""Suggest optimal epsilon and min_samples for DBSCAN."""
	n_features = scaled_df.shape[1]
	
	# Heuristic for min_samples: 2 * dimensions
	suggested_min_samples = max(4, 2 * n_features)
	
	# Estimate epsilon using k-distance graph (k = min_samples)
	neigh = NearestNeighbors(n_neighbors=suggested_min_samples)
	nbrs = neigh.fit(scaled_df)
	distances, _ = nbrs.kneighbors(scaled_df)
	
	# Sort distances to the k-th nearest neighbor
	k_distances = np.sort(distances[:, suggested_min_samples - 1])
	
	# Find elbow in k-distance graph
	suggested_eps = 0.5 # Default fallback
	try:
		# Use kneed to find the elbow point
		x = np.arange(len(k_distances))
		kn = KneeLocator(x, k_distances, curve="convex", direction="increasing")
		if kn.elbow:
			suggested_eps = float(k_distances[kn.elbow])
	except Exception:
		pass
		
	# Prepare data for plotting the k-distance graph
	k_dist_df = pd.DataFrame({
		"index": np.arange(len(k_distances)),
		"distance": k_distances
	})
	
	return suggested_eps, suggested_min_samples, k_dist_df


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


def _create_model(algorithm: str, n_clusters: int, random_state: int, **kwargs):
	if algorithm == "K-Means":
		return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

	if algorithm == "K-Medoids":
		if KMedoids is None:
			raise ImportError(
				"K-Medoids requires scikit-learn-extra. Install it with: pip install scikit-learn-extra"
			)
		return KMedoids(n_clusters=n_clusters, random_state=random_state, init="k-medoids++")

	if algorithm == "AGNES (Agglomerative)":
		return AgglomerativeClustering(n_clusters=n_clusters)

	if algorithm == "DIANA (Divisive)":
		return BisectingKMeans(n_clusters=n_clusters, random_state=random_state, n_init=1)

	if algorithm == "DBSCAN":
		eps = kwargs.get("eps", 0.5)
		min_samples = kwargs.get("min_samples", 5)
		return DBSCAN(eps=eps, min_samples=min_samples)

	raise ValueError("Unsupported clustering algorithm.")


def compute_elbow_curve(
	scaled_df: pd.DataFrame,
	algorithm: str,
	min_k: int = 2,
	max_k: int = 10,
	random_state: int = 42,
) -> tuple[pd.DataFrame, int | None]:
	"""Compute inertia values for a range of k values and suggest best k."""
	if algorithm in ["DBSCAN"]:
		return pd.DataFrame(), None

	if len(scaled_df) < 3:
		raise ValueError("At least 3 rows are required to compute the elbow curve.")

	max_valid_k = min(max_k, len(scaled_df) - 1)
	if max_valid_k < min_k:
		raise ValueError("Dataset is too small for the selected elbow k-range.")

	records: list[dict[str, float | int]] = []
	ks = range(min_k, max_valid_k + 1)
	inertias = []

	for k in ks:
		model = _create_model(algorithm=algorithm, n_clusters=k, random_state=random_state)
		model.fit(scaled_df)
		
		# AgglomerativeClustering doesn't have inertia_, we might use another metric or just skip
		if hasattr(model, "inertia_"):
			inertia = float(model.inertia_)
		else:
			# Fallback or alternative for models without inertia
			inertia = 0.0
			
		records.append({"k": k, "inertia": inertia})
		inertias.append(inertia)

	best_k = None
	if any(v > 0 for v in inertias):
		try:
			kn = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
			best_k = kn.elbow
		except Exception:
			best_k = None

	return pd.DataFrame(records), best_k


def run_clustering(
	scaled_df: pd.DataFrame,
	algorithm: str,
	n_clusters: int,
	random_state: int = 42,
	**kwargs
) -> tuple[pd.Series, object, float, float | None]:
	"""Fit the selected clustering model and return labels, model, execution time, and inertia."""
	start_time = time.time()
	
	model = _create_model(algorithm=algorithm, n_clusters=n_clusters, random_state=random_state, **kwargs)
	
	if algorithm == "DBSCAN":
		labels = model.fit_predict(scaled_df)
	else:
		if n_clusters < 2:
			raise ValueError("Number of clusters must be at least 2.")
		if n_clusters >= len(scaled_df):
			raise ValueError("Number of clusters must be smaller than the number of samples.")
		labels = model.fit_predict(scaled_df)
		
	execution_time = time.time() - start_time
	labels_series = pd.Series(labels, index=scaled_df.index, name="cluster")
	
	# Extract inertia if available
	inertia = None
	if hasattr(model, "inertia_"):
		inertia = float(model.inertia_)
		
	return labels_series, model, execution_time, inertia


def compute_silhouette(scaled_df: pd.DataFrame, labels: pd.Series) -> float:
	"""Compute silhouette score for the current clustering labels."""
	unique_clusters = [c for c in labels.unique() if c != -1] # Exclude DBSCAN noise
	if len(unique_clusters) < 2:
		return 0.0
	return float(silhouette_score(scaled_df, labels))


def compare_algorithms(
	scaled_df: pd.DataFrame,
	algorithms: list[str],
	n_clusters: int,
	random_state: int = 42
) -> pd.DataFrame:
	"""Run multiple algorithms and compare their performance."""
	results = []
	for algo in algorithms:
		try:
			labels, _, exec_time = run_clustering(scaled_df, algo, n_clusters, random_state)
			sil_score = compute_silhouette(scaled_df, labels)
			
			results.append({
				"Algorithm": algo,
				"Execution Time (s)": round(exec_time, 6),
				"Silhouette Score": round(sil_score, 4),
				"Clusters Found": len([c for c in labels.unique() if c != -1])
			})
		except Exception as e:
			print(f"Error comparing {algo}: {e}")
			
	return pd.DataFrame(results)


def reduce_with_pca(scaled_df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
	"""Project scaled data to 2D for plotting."""
	pca = PCA(n_components=n_components, random_state=42)
	components = pca.fit_transform(scaled_df)
	return pd.DataFrame(components, columns=["PC1", "PC2"], index=scaled_df.index)
