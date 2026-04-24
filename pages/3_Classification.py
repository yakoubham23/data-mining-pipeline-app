from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from utils.classification import (
	get_class_labels,
	prepare_classification_data,
	save_model_bundle,
	train_and_evaluate_models,
)
from utils.preprocessing import load_csv_dataset
from utils.visualization import create_confusion_matrix_figure


MODEL_OPTIONS = ["Logistic Regression", "Decision Tree", "Random Forest"]


def _load_dataset_from_source(source_name: str):
	sample_path = Path(__file__).resolve().parents[1] / "data" / "sample.csv"

	if source_name == "Use processed dataset from preprocessing":
		if st.session_state.get("processed_df") is None:
			st.warning("No processed dataset found. Go to the Preprocessing page first.")
			return None
		return st.session_state.processed_df.copy()

	if source_name == "Upload new CSV":
		uploaded_file = st.file_uploader(
			"Upload CSV for classification", type=["csv"], key="classification_uploader"
		)
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


st.title("🎯 Supervised Learning - Classification")
st.caption("Train and compare multiple classifiers with robust preprocessing pipelines.")

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
	st.info("Select a dataset source to start classification.")
	st.stop()

if len(df.columns) < 2:
	st.error("Classification requires at least one feature column and one target column.")
	st.stop()

summary_cols = st.columns(3)
summary_cols[0].metric("Rows", len(df))
summary_cols[1].metric("Columns", len(df.columns))
summary_cols[2].metric("Missing values", int(df.isna().sum().sum()))

with st.expander("⚙️ Classification Configuration", expanded=True):
	target_column = st.selectbox("Select target column", options=df.columns.tolist())
	feature_candidates = [col for col in df.columns if col != target_column]
	selected_features = st.multiselect(
		"Select feature columns",
		options=feature_candidates,
		default=feature_candidates,
	)

	config_cols = st.columns(3)
	test_size = config_cols[0].slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
	random_state = config_cols[1].number_input("Random state", min_value=0, value=42, step=1)
	selected_models = config_cols[2].multiselect(
		"Models",
		options=MODEL_OPTIONS,
		default=MODEL_OPTIONS,
	)

	run_btn = st.button("🚀 Train and evaluate models", use_container_width=True)

if run_btn:
	try:
		X_train, X_test, y_train, y_test, label_encoder, used_features = prepare_classification_data(
			df=df,
			target_column=target_column,
			feature_columns=selected_features,
			test_size=float(test_size),
			random_state=int(random_state),
		)

		comparison_df, trained_models, predictions = train_and_evaluate_models(
			X_train=X_train,
			X_test=X_test,
			y_train=y_train,
			y_test=y_test,
			selected_models=selected_models,
			random_state=int(random_state),
		)

		st.session_state.classification_result = {
			"comparison_df": comparison_df,
			"trained_models": trained_models,
			"predictions": predictions,
			"y_test": y_test,
			"label_encoder": label_encoder,
			"used_features": used_features,
			"target_column": target_column,
		}
	except ValueError as exc:
		st.error(str(exc))

result = st.session_state.get("classification_result")
if result:
	st.subheader("📋 Model Comparison")
	st.dataframe(result["comparison_df"], use_container_width=True)

	class_labels = get_class_labels(result["y_test"], result["label_encoder"])

	st.subheader("🧩 Confusion Matrices")
	for model_name, y_pred in result["predictions"].items():
		fig = create_confusion_matrix_figure(
			y_true=result["y_test"],
			y_pred=y_pred,
			class_labels=class_labels,
			model_name=model_name,
		)
		st.plotly_chart(fig, use_container_width=True)

	best_model_name = result["comparison_df"].iloc[0]["Model"]
	st.info(f"Best model by F1-score: {best_model_name}")

	if st.button("💾 Save best model", use_container_width=True):
		model_path = Path(__file__).resolve().parents[1] / "models" / "saved_models.pkl"
		bundle = {
			"saved_at": datetime.utcnow().isoformat() + "Z",
			"best_model_name": best_model_name,
			"model": result["trained_models"][best_model_name],
			"target_column": result["target_column"],
			"feature_columns": result["used_features"],
			"label_encoder": result["label_encoder"],
			"metrics": result["comparison_df"].to_dict(orient="records"),
		}
		save_model_bundle(model_path, bundle)
		st.success(f"Model saved successfully to: {model_path}")
