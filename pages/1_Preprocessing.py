from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.preprocessing import (
	detect_column_types,
	encode_categorical,
	get_dataframe_profile,
	handle_missing_values,
	load_csv_dataset,
	missing_values_report,
	scale_features,
)
from utils.visualization import create_boxplot_figure, create_scatter_figure


def _initialize_session_state() -> None:
	if "original_df" not in st.session_state:
		st.session_state.original_df = None
	if "processed_df" not in st.session_state:
		st.session_state.processed_df = None
	if "data_source" not in st.session_state:
		st.session_state.data_source = None


def _set_dataset(df, source_name: str) -> None:
	st.session_state.original_df = df.copy()
	st.session_state.processed_df = df.copy()
	st.session_state.data_source = source_name


_initialize_session_state()

st.title("🧹 Data Preprocessing")
st.caption("Upload any CSV and build a clean dataset for clustering and classification.")

sample_path = Path(__file__).resolve().parents[1] / "data" / "sample.csv"

with st.expander("📥 Load Dataset", expanded=True):
	uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="preprocess_uploader")
	load_cols = st.columns(3)

	if load_cols[0].button(
		"📂 Load uploaded CSV",
		use_container_width=True,
		disabled=uploaded_file is None,
	):
		try:
			loaded_df = load_csv_dataset(uploaded_file)
			_set_dataset(loaded_df, uploaded_file.name)
			st.success(f"Dataset loaded successfully: {uploaded_file.name}")
			st.rerun()
		except ValueError as exc:
			st.error(str(exc))

	if load_cols[1].button("🧪 Load sample dataset", use_container_width=True):
		try:
			loaded_df = load_csv_dataset(sample_path)
			_set_dataset(loaded_df, "sample.csv")
			st.success("Sample dataset loaded successfully.")
			st.rerun()
		except ValueError as exc:
			st.error(str(exc))

	if load_cols[2].button(
		"♻️ Reset to original",
		use_container_width=True,
		disabled=st.session_state.original_df is None,
	):
		st.session_state.processed_df = st.session_state.original_df.copy()
		st.success("Working dataset reset to the original loaded file.")
		st.rerun()

if st.session_state.processed_df is None:
	st.info("Upload a CSV or load the sample dataset to start preprocessing.")
	st.stop()

df = st.session_state.processed_df.copy()
numeric_cols, categorical_cols = detect_column_types(df)

st.success(f"Active data source: {st.session_state.data_source}")

with st.expander("👀 Dataset Preview & Summary", expanded=True):
	profile = get_dataframe_profile(df)
	metric_cols = st.columns(3)
	metric_cols[0].metric("Rows", profile["shape"][0])
	metric_cols[1].metric("Columns", profile["shape"][1])
	metric_cols[2].metric("Missing values", int(df.isna().sum().sum()))

	st.subheader("Head")
	st.dataframe(profile["head"], use_container_width=True)

	st.subheader("DataFrame Info")
	st.code(profile["info"])

	st.subheader("Statistical Summary")
	st.dataframe(profile["describe"], use_container_width=True)

with st.expander("🩹 Missing Values Handling", expanded=True):
	report_df = missing_values_report(df)
	st.dataframe(report_df, use_container_width=True)

	strategy = st.selectbox(
		"Choose strategy",
		options=["dropna", "mean", "median", "mode"],
		format_func=lambda value: {
			"dropna": "Drop rows with missing values",
			"mean": "Fill numeric with mean",
			"median": "Fill numeric with median",
			"mode": "Fill with mode",
		}[value],
		key="missing_strategy",
	)

	default_columns = df.columns.tolist()
	if strategy in {"mean", "median"}:
		default_columns = numeric_cols

	selected_columns = st.multiselect(
		"Columns to process",
		options=df.columns.tolist(),
		default=default_columns,
		key="missing_columns",
	)

	if st.button("✅ Apply missing values strategy", use_container_width=True):
		try:
			columns = None if strategy == "dropna" else selected_columns
			updated_df = handle_missing_values(df, strategy=strategy, columns=columns)
			st.session_state.processed_df = updated_df
			st.success("Missing values strategy applied.")
			st.rerun()
		except ValueError as exc:
			st.error(str(exc))

with st.expander("🔠 Encode Categorical Variables", expanded=True):
	if not categorical_cols:
		st.info("No categorical columns detected.")
	else:
		selected_cat_cols = st.multiselect(
			"Categorical columns to encode",
			options=categorical_cols,
			default=categorical_cols,
			key="encoding_columns",
		)
		drop_first = st.toggle("Drop first level (avoid dummy trap)", value=False)

		if st.button("✅ Apply one-hot encoding", use_container_width=True):
			updated_df = encode_categorical(df, columns=selected_cat_cols, drop_first=drop_first)
			st.session_state.processed_df = updated_df
			st.success("Categorical encoding applied.")
			st.rerun()

with st.expander("📏 Feature Scaling", expanded=True):
	if not numeric_cols:
		st.warning("No numeric columns available for scaling.")
	else:
		selected_num_cols = st.multiselect(
			"Numeric columns to scale",
			options=numeric_cols,
			default=numeric_cols,
			key="scale_columns",
		)
		scaler_name = st.radio(
			"Scaler",
			options=["MinMaxScaler", "StandardScaler"],
			horizontal=True,
			key="scaler_choice",
		)

		if st.button("✅ Apply scaling", use_container_width=True):
			try:
				method = "minmax" if scaler_name == "MinMaxScaler" else "standard"
				updated_df, _ = scale_features(df, method=method, columns=selected_num_cols)
				st.session_state.processed_df = updated_df
				st.success(f"Scaling applied with {scaler_name}.")
				st.rerun()
			except ValueError as exc:
				st.error(str(exc))

with st.expander("📈 Visualizations", expanded=True):
	current_numeric_cols, current_categorical_cols = detect_column_types(df)

	if current_numeric_cols:
		boxplot_cols = st.multiselect(
			"Columns for boxplot",
			options=current_numeric_cols,
			default=current_numeric_cols[: min(5, len(current_numeric_cols))],
			key="boxplot_cols",
		)
		if boxplot_cols:
			st.pyplot(create_boxplot_figure(df, boxplot_cols))

		scatter_cols = st.columns(2)
		x_col = scatter_cols[0].selectbox("Scatter X-axis", options=current_numeric_cols, key="scatter_x")
		y_options = [col for col in current_numeric_cols if col != x_col] or current_numeric_cols
		y_col = scatter_cols[1].selectbox("Scatter Y-axis", options=y_options, key="scatter_y")

		color_options = ["None"] + current_categorical_cols
		color_choice = st.selectbox("Color by", options=color_options, key="scatter_color")
		color_col = None if color_choice == "None" else color_choice

		scatter_fig = create_scatter_figure(df, x_col=x_col, y_col=y_col, color_col=color_col)
		st.plotly_chart(scatter_fig, use_container_width=True)
	else:
		st.warning("Visualizations need at least one numeric column.")

st.subheader("💾 Export Processed Dataset")
st.download_button(
	label="Download processed CSV",
	data=df.to_csv(index=False).encode("utf-8"),
	file_name="processed_dataset.csv",
	mime="text/csv",
	use_container_width=True,
)
