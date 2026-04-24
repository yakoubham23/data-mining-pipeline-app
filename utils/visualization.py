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
	y_col: str,
	color_col: str | None = None,
) -> go.Figure:
	"""Create an interactive Plotly scatter figure."""
	hover_data = [col for col in df.columns if col not in {x_col, y_col}][:6]
	fig = px.scatter(
		df,
		x=x_col,
		y=y_col,
		color=color_col,
		hover_data=hover_data,
		title=f"Scatter Plot: {x_col} vs {y_col}",
		template="plotly_white",
	)
	fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
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
