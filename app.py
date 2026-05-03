from __future__ import annotations

import streamlit as st

st.set_page_config(
	page_title="Data Mining Pipeline",
	page_icon="🧠",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

	html, body, [class*="css"]  {
		font-family: 'Manrope', sans-serif;
	}

	.stApp {
		background:
			radial-gradient(circle at 10% 10%, rgba(59, 130, 246, 0.15), transparent 35%),
			radial-gradient(circle at 85% 20%, rgba(16, 185, 129, 0.12), transparent 35%),
			linear-gradient(180deg, #f8fbff 0%, #f7fff9 100%);
	}

	.main .block-container {
		padding-top: 2rem;
		padding-bottom: 2rem;
	}

	.hero {
		background: linear-gradient(120deg, #0f766e 0%, #2563eb 100%);
		border-radius: 22px;
		padding: 1.35rem 1.45rem;
		color: #ffffff;
		box-shadow: 0 14px 30px rgba(2, 6, 23, 0.15);
		animation: riseIn 700ms ease-out;
	}

	.hero h1 {
		margin: 0;
		font-size: 2rem;
		line-height: 1.2;
	}

	.hero p {
		margin-top: 0.55rem;
		opacity: 0.95;
		font-size: 1rem;
	}

	.dashboard-card {
		background: rgba(255, 255, 255, 0.88);
		border: 1px solid #dbeafe;
		border-radius: 16px;
		padding: 1rem 1.1rem;
		box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
		min-height: 128px;
		animation: riseIn 800ms ease-out;
	}

	.dashboard-card h4 {
		margin-top: 0;
		margin-bottom: 0.45rem;
		color: #0f172a;
	}

	.dashboard-card p {
		margin: 0;
		color: #334155;
		font-size: 0.95rem;
	}

	@keyframes riseIn {
		from {
			opacity: 0;
			transform: translateY(10px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}
	</style>
	""",
	unsafe_allow_html=True,
)

with st.sidebar:
	st.title("🧭 Pipeline Navigation")
	st.write("Use Streamlit pages to navigate the project modules:")
	st.markdown("- 🧹 1_Preprocessing")
	st.markdown("- 🧠 2_Clustering")
	st.markdown("- 🎯 3_Classification")

	processed_df = st.session_state.get("processed_df")
	st.markdown("---")
	if processed_df is not None:
		st.success("Processed dataset is available in session.")
		st.caption(f"Rows: {processed_df.shape[0]} | Columns: {processed_df.shape[1]}")
	else:
		st.info("No processed dataset yet. Start from the preprocessing page.")

st.markdown(
	"""
	<div class="hero">
		<h1>📊 Data Mining Pipeline Interface</h1>
		<p>
			End-to-end university mini-project dashboard for preprocessing, clustering,
			and classification with interactive visual analytics.
		</p>
	</div>
	""",
	unsafe_allow_html=True,
)

st.write("")

cards = st.columns(3)
cards[0].markdown(
	"""
	<div class="dashboard-card">
		<h4>🧹 Preprocessing</h4>
		<p>Upload CSV files, inspect data quality, handle missing values, encode
		categorical columns, scale numeric features, and export cleaned data.</p>
	</div>
	""",
	unsafe_allow_html=True,
)
cards[1].markdown(
	"""
	<div class="dashboard-card">
		<h4>🧠 Clustering</h4>
		<p>Run K-Means or K-Medoids, evaluate with elbow and silhouette metrics,
		and inspect cluster separation using PCA-based interactive visuals.</p>
	</div>
	""",
	unsafe_allow_html=True,
)
cards[2].markdown(
	"""
	<div class="dashboard-card">
		<h4>🎯 Classification</h4>
		<p>Train Logistic Regression, Decision Tree, and Random Forest models,
		compare metrics, visualize confusion matrices, and save the best model.</p>
	</div>
	""",
	unsafe_allow_html=True,
)

st.write("")

info_cols = st.columns(4)
info_cols[0].metric("Modules", 3)
info_cols[1].metric("Algorithms", 5)
info_cols[2].metric("Train/Test Split", "80/20")
info_cols[3].metric("Visual Layer", "Plotly + Seaborn")

with st.expander("🚀 Suggested Workflow", expanded=True):
	st.markdown("1. Open the Preprocessing page and load your CSV dataset.")
	st.markdown("2. Clean and transform the data, then keep it in session state.")
	st.markdown("3. Explore groups in the Clustering page and validate structure quality.")
	st.markdown("4. Move to Classification to train models and compare performance.")
	st.markdown("5. Save the best trained model to models/saved_models.pkl.")

if st.session_state.get("processed_df") is not None:
	st.subheader("🗂️ Current Processed Dataset Preview")
	st.dataframe(st.session_state.processed_df.head(10), width='stretch')
