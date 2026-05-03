"""Reusable visualization helpers for the Streamlit app."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import confusion_matrix


def create_boxplot_figure(df: pd.DataFrame, columns: list[str]) -> plt.Figure:
	"""Create a Matplotlib boxplot figure for selected columns."""
	width = max(8, len(columns) * 1.2)
	fig, ax = plt.subplots(figsize=(width, 4.5))
	sns.boxplot(data=df[columns], ax=ax)
	ax.set_title("Boxplot - Outlier Detection")
	ax.set_xlabel("Features")
	ax.set_ylabel("Values")
	ax.tick_params(axis="x", rotation=45)
	fig.tight_layout()
	return fig


def create_scatter_figure(
	df: pd.DataFrame,
	x_col: str,
	y_cols: list[str],
	color_col: str | None = None,
) -> go.Figure:
	"""Create a robust scatter figure handling single or multiple Y-axes and color dimensions."""
	
	if len(y_cols) == 1:
		# Case 1: Standard Scatter (1 X, 1 Y)
		# We can use the user's color_col directly for color
		fig = px.scatter(
			df,
			x=x_col,
			y=y_cols[0],
			color=color_col if color_col != "None" else None,
			title=f"Scatter Plot: {x_col} vs {y_cols[0]}",
			template="plotly_white",
			labels={y_cols[0]: "Value"}
		)
	else:
		# Case 2: Multi-Variable Comparison (1 X, Multiple Ys)
		# To differentiate variables AND categories, we melt the dataframe
		id_vars = [x_col]
		if color_col and color_col != "None" and color_col not in y_cols and color_col != x_col:
			id_vars.append(color_col)
			
		melted_df = df.melt(id_vars=id_vars, value_vars=y_cols, var_name="Variable", value_name="Value")
		
		# If user picked a category for color, we use it for symbols to avoid color conflict
		symbol_col = None
		if color_col and color_col != "None":
			# Only use symbol if it's categorical-ish (low cardinality)
			if melted_df[color_col].nunique() < 10:
				symbol_col = color_col
		
		fig = px.scatter(
			melted_df,
			x=x_col,
			y="Value",
			color="Variable", # Color differentiates between the multiple Y variables
			symbol=symbol_col, # Symbol differentiates between the user's chosen category
			title=f"Multi-Variable Comparison: {x_col} vs {', '.join(y_cols)}",
			template="plotly_white",
			hover_data=[color_col] if color_col and color_col != "None" else None
		)

	fig.update_layout(
		margin=dict(l=20, r=20, t=55, b=20),
		legend_title_text="Legend"
	)
	return fig


def create_elbow_figure(elbow_df: pd.DataFrame, algorithm_name: str) -> go.Figure:
	"""Create an elbow-method line plot."""
	fig = px.line(
		elbow_df,
		x="k",
		y="inertia",
		markers=True,
		title=f"Elbow Method - {algorithm_name}",
		template="plotly_white",
	)
	fig.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Inertia")
	return fig


def create_cluster_2d_figure(pca_df: pd.DataFrame, cluster_col: str = "cluster") -> go.Figure:
	"""Create a 2D PCA cluster visualization."""
	fig = px.scatter(
		pca_df,
		x="PC1",
		y="PC2",
		color=cluster_col,
		title="2D Cluster Visualization (PCA)",
		template="plotly_white",
	)
	fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
	fig.update_layout(legend_title_text="Cluster", margin=dict(l=20, r=20, t=55, b=20))
	return fig


def create_cluster_distribution_figure(cluster_series: pd.Series) -> go.Figure:
	"""Create a bar chart for cluster membership counts."""
	counts = cluster_series.value_counts().sort_index().reset_index()
	counts.columns = ["cluster", "count"]
	fig = px.bar(
		counts,
		x="cluster",
		y="count",
		text="count",
		title="Cluster Distribution",
		template="plotly_white",
	)
	fig.update_layout(xaxis_title="Cluster", yaxis_title="Number of Samples")
	return fig


from scipy.cluster.hierarchy import dendrogram, linkage

def create_dendrogram_figure(scaled_df: pd.DataFrame, method: str = "ward") -> plt.Figure:
	"""Create a dendrogram for hierarchical clustering using SciPy."""
	# Limit to 50 samples for readability if the dataset is too large
	if len(scaled_df) > 50:
		data_to_plot = scaled_df.sample(50, random_state=42)
		title_suffix = "(Sample of 50 points)"
	else:
		data_to_plot = scaled_df
		title_suffix = ""

	linked = linkage(data_to_plot, method=method)
	
	fig, ax = plt.subplots(figsize=(10, 5))
	dendrogram(
		linked,
		orientation='top',
		distance_sort='descending',
		show_leaf_counts=True,
		ax=ax
	)
	ax.set_title(f"Hierarchical Clustering Dendrogram {title_suffix}")
	ax.set_xlabel("Sample Index")
	ax.set_ylabel("Distance")
	fig.tight_layout()
	return fig


def create_confusion_matrix_figure(
	y_true: pd.Series,
	y_pred: pd.Series,
	class_labels: list[str],
	model_name: str,
) -> go.Figure:
	"""Create an interactive confusion matrix heatmap."""
	observed_values = sorted(pd.concat([pd.Series(y_true), pd.Series(y_pred)]).unique())
	cm = confusion_matrix(y_true, y_pred, labels=observed_values)
	axis_labels = class_labels if len(class_labels) == len(observed_values) else [str(v) for v in observed_values]

	fig = px.imshow(
		cm,
		text_auto=True,
		x=axis_labels,
		y=axis_labels,
		color_continuous_scale="Blues",
		title=f"Confusion Matrix - {model_name}",
		template="plotly_white",
	)
	fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
	return fig


def create_comparison_charts(comparison_df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
	"""Create bar charts for algorithm comparison."""
	time_fig = px.bar(
		comparison_df,
		x="Algorithm",
		y="Execution Time (s)",
		title="Execution Time Comparison",
		template="plotly_white",
		color="Algorithm"
	)
	
	sil_fig = px.bar(
		comparison_df,
		x="Algorithm",
		y="Silhouette Score",
		title="Silhouette Score Comparison",
		template="plotly_white",
		color="Algorithm"
	)
	
	return time_fig, sil_fig
