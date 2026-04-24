"""Preprocessing utility functions for tabular datasets."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_csv_dataset(file_source: Any) -> pd.DataFrame:
	"""Load a CSV dataset from an uploaded file or filesystem path."""
	if file_source is None:
		raise ValueError("No CSV source was provided.")

	try:
		if isinstance(file_source, (str, Path)):
			dataframe = pd.read_csv(file_source)
		else:
			dataframe = pd.read_csv(file_source)
	except Exception as exc:  # noqa: BLE001
		raise ValueError(f"Unable to read the CSV file: {exc}") from exc

	if dataframe.empty:
		raise ValueError("The uploaded dataset is empty.")

	return dataframe


def get_dataframe_profile(df: pd.DataFrame) -> dict[str, Any]:
	"""Return key metadata used in the UI summary panels."""
	buffer = StringIO()
	df.info(buf=buffer)

	return {
		"shape": df.shape,
		"head": df.head(),
		"info": buffer.getvalue(),
		"describe": df.describe(include="all").transpose(),
	}


def detect_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
	"""Split columns into numeric and categorical groups."""
	numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
	categorical_cols = [col for col in df.columns if col not in numeric_cols]
	return numeric_cols, categorical_cols


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
	"""Build a tabular report for missing values per column."""
	missing_counts = df.isna().sum()
	report = pd.DataFrame(
		{
			"column": missing_counts.index,
			"missing_count": missing_counts.values,
		}
	)
	report["missing_ratio"] = report["missing_count"] / len(df)
	return report.sort_values(by="missing_count", ascending=False).reset_index(drop=True)


def handle_missing_values(
	df: pd.DataFrame,
	strategy: str,
	columns: list[str] | None = None,
) -> pd.DataFrame:
	"""Apply a missing-value strategy and return a new dataframe."""
	updated = df.copy()
	selected_columns = columns if columns else updated.columns.tolist()
	selected_columns = [col for col in selected_columns if col in updated.columns]

	if not selected_columns and strategy != "dropna":
		raise ValueError("Please select at least one valid column.")

	if strategy == "dropna":
		return updated.dropna().reset_index(drop=True)

	if strategy == "mean":
		numeric_cols = [
			col for col in selected_columns if pd.api.types.is_numeric_dtype(updated[col])
		]
		if not numeric_cols:
			raise ValueError("Mean imputation requires at least one numeric column.")
		updated[numeric_cols] = updated[numeric_cols].fillna(
			updated[numeric_cols].mean(numeric_only=True)
		)
		return updated

	if strategy == "median":
		numeric_cols = [
			col for col in selected_columns if pd.api.types.is_numeric_dtype(updated[col])
		]
		if not numeric_cols:
			raise ValueError("Median imputation requires at least one numeric column.")
		updated[numeric_cols] = updated[numeric_cols].fillna(updated[numeric_cols].median())
		return updated

	if strategy == "mode":
		for col in selected_columns:
			mode_series = updated[col].mode(dropna=True)
			if not mode_series.empty:
				updated[col] = updated[col].fillna(mode_series.iloc[0])
		return updated

	raise ValueError("Unknown missing-value strategy.")


def encode_categorical(
	df: pd.DataFrame,
	columns: list[str] | None = None,
	drop_first: bool = False,
) -> pd.DataFrame:
	"""One-hot encode selected categorical columns."""
	updated = df.copy()
	_, categorical_cols = detect_column_types(updated)

	selected_columns = columns if columns else categorical_cols
	selected_columns = [col for col in selected_columns if col in categorical_cols]

	if not selected_columns:
		return updated

	return pd.get_dummies(
		updated,
		columns=selected_columns,
		drop_first=drop_first,
		dtype=int,
	)


def scale_features(
	df: pd.DataFrame,
	method: str,
	columns: list[str],
) -> tuple[pd.DataFrame, MinMaxScaler | StandardScaler]:
	"""Scale selected numeric columns using MinMax or Standard scaling."""
	if not columns:
		raise ValueError("Please select at least one column for scaling.")

	invalid_cols = [col for col in columns if col not in df.columns]
	if invalid_cols:
		raise ValueError(f"Invalid columns selected: {invalid_cols}")

	non_numeric_cols = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
	if non_numeric_cols:
		raise ValueError(f"Scaling requires numeric columns only: {non_numeric_cols}")

	updated = df.copy()

	if method == "minmax":
		scaler: MinMaxScaler | StandardScaler = MinMaxScaler()
	elif method == "standard":
		scaler = StandardScaler()
	else:
		raise ValueError("Unsupported scaling method.")

	updated[columns] = scaler.fit_transform(updated[columns])
	return updated, scaler
