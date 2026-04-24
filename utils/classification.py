"""Classification utility functions for supervised learning workflows."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def prepare_classification_data(
	df: pd.DataFrame,
	target_column: str,
	feature_columns: list[str] | None = None,
	test_size: float = 0.2,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LabelEncoder | None, list[str]]:
	"""Prepare train/test data with optional target encoding."""
	if target_column not in df.columns:
		raise ValueError("Selected target column does not exist in the dataset.")

	if feature_columns is None or len(feature_columns) == 0:
		feature_columns = [col for col in df.columns if col != target_column]

	if not feature_columns:
		raise ValueError("No feature columns available for training.")

	missing_feature_cols = [col for col in feature_columns if col not in df.columns]
	if missing_feature_cols:
		raise ValueError(f"Invalid feature columns: {missing_feature_cols}")

	model_df = df[feature_columns + [target_column]].copy()
	model_df = model_df.dropna(subset=[target_column])

	if model_df.empty:
		raise ValueError("No rows available after dropping missing target values.")

	X = model_df[feature_columns]
	y_raw = model_df[target_column]

	label_encoder: LabelEncoder | None = None
	if not pd.api.types.is_numeric_dtype(y_raw):
		label_encoder = LabelEncoder()
		y = pd.Series(
			label_encoder.fit_transform(y_raw.astype(str)),
			index=y_raw.index,
			name=target_column,
		)
	else:
		y_numeric = pd.to_numeric(y_raw, errors="coerce")
		if y_numeric.isna().any():
			raise ValueError("Target column contains non-numeric values after conversion.")
		y = y_numeric.astype(int) if ((y_numeric % 1) == 0).all() else y_numeric

	if y.nunique() < 2:
		raise ValueError("Classification requires at least two target classes.")

	stratify = None
	class_counts = y.value_counts()
	if class_counts.min() >= 2:
		stratify = y

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=stratify,
	)

	return X_train, X_test, y_train, y_test, label_encoder, feature_columns


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
	numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
	categorical_features = [col for col in X.columns if col not in numeric_features]

	transformers = []
	if numeric_features:
		numeric_pipeline = Pipeline(
			steps=[
				("imputer", SimpleImputer(strategy="median")),
				("scaler", StandardScaler()),
			]
		)
		transformers.append(("num", numeric_pipeline, numeric_features))

	if categorical_features:
		categorical_pipeline = Pipeline(
			steps=[
				("imputer", SimpleImputer(strategy="most_frequent")),
				("encoder", OneHotEncoder(handle_unknown="ignore")),
			]
		)
		transformers.append(("cat", categorical_pipeline, categorical_features))

	if not transformers:
		raise ValueError("No valid features available for model training.")

	return ColumnTransformer(transformers=transformers)


def build_model_pipelines(X: pd.DataFrame, random_state: int = 42) -> dict[str, Pipeline]:
	"""Create reusable sklearn pipelines for all required classifiers."""
	preprocessor = _build_preprocessor(X)

	estimators = {
		"Logistic Regression": LogisticRegression(max_iter=2000),
		"Decision Tree": DecisionTreeClassifier(random_state=random_state),
		"Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
	}

	return {
		model_name: Pipeline(
			steps=[
				("preprocessor", preprocessor),
				("classifier", estimator),
			]
		)
		for model_name, estimator in estimators.items()
	}


def train_and_evaluate_models(
	X_train: pd.DataFrame,
	X_test: pd.DataFrame,
	y_train: pd.Series,
	y_test: pd.Series,
	selected_models: list[str],
	random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, pd.Series]]:
	"""Train selected models and compute standard classification metrics."""
	if not selected_models:
		raise ValueError("Select at least one model before training.")

	all_pipelines = build_model_pipelines(X_train, random_state=random_state)

	unsupported = [name for name in selected_models if name not in all_pipelines]
	if unsupported:
		raise ValueError(f"Unsupported model(s): {unsupported}")

	metrics_records: list[dict[str, float | str]] = []
	trained_models: dict[str, Pipeline] = {}
	predictions: dict[str, pd.Series] = {}

	for model_name in selected_models:
		pipeline = all_pipelines[model_name]
		pipeline.fit(X_train, y_train)
		y_pred = pd.Series(pipeline.predict(X_test), index=y_test.index, name="prediction")

		accuracy = accuracy_score(y_test, y_pred)
		precision, recall, f1, _ = precision_recall_fscore_support(
			y_test,
			y_pred,
			average="weighted",
			zero_division=0,
		)

		metrics_records.append(
			{
				"Model": model_name,
				"Accuracy": round(float(accuracy), 4),
				"Precision": round(float(precision), 4),
				"Recall": round(float(recall), 4),
				"F1-score": round(float(f1), 4),
			}
		)

		trained_models[model_name] = pipeline
		predictions[model_name] = y_pred

	comparison_df = (
		pd.DataFrame(metrics_records)
		.sort_values(by="F1-score", ascending=False)
		.reset_index(drop=True)
	)

	return comparison_df, trained_models, predictions


def get_class_labels(y: pd.Series, label_encoder: LabelEncoder | None = None) -> list[str]:
	"""Return class labels as strings for confusion-matrix axes."""
	if label_encoder is not None:
		return [str(label) for label in label_encoder.classes_]
	unique_values = sorted(y.unique())
	return [str(label) for label in unique_values]


def save_model_bundle(save_path: str | Path, bundle: dict) -> None:
	"""Persist a trained model bundle with joblib."""
	path = Path(save_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(bundle, path)
