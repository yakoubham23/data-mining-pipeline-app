# Data Mining Pipeline App

Production-ready Streamlit application for a university mini-project in Data Mining.

The app provides a full interactive workflow:

1. Data Preprocessing
2. Unsupervised Learning (Clustering)
3. Supervised Learning (Classification)

Repository target:
https://github.com/yakoubham23/data-mining-pipeline-app

## Key Features

### 1) Preprocessing Module
- CSV upload and sample dataset loading
- Dataset preview: head, shape, dataframe info
- Statistical summary
- Missing values handling:
	- Drop rows with missing values
	- Fill with mean
	- Fill with median
	- Fill with mode
- Categorical encoding (one-hot)
- Feature scaling:
	- MinMaxScaler
	- StandardScaler
- Visualizations:
	- Boxplots (outlier detection)
	- Scatter plots (feature relationships)
- Processed dataset download

### 2) Clustering Module
- K-Means
- K-Medoids (scikit-learn-extra)
- Elbow method plotting
- Silhouette score evaluation
- PCA dimensionality reduction to 2D
- Interactive Plotly cluster visualization
- Clustered dataset export

### 3) Classification Module
- Train/test split (default 80/20)
- Models:
	- Logistic Regression
	- Decision Tree
	- Random Forest
- Metrics:
	- Accuracy
	- Precision
	- Recall
	- F1-score
- Confusion matrix visualization (interactive)
- Model comparison table
- Best-model persistence with joblib to models/saved_models.pkl

## Tech Stack

- Python 3.10+
- Streamlit
- pandas, numpy
- scikit-learn
- scikit-learn-extra
- matplotlib, seaborn
- plotly
- joblib

## Project Structure

data_mining_app/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample.csv
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── classification.py
│   └── visualization.py
├── models/
│   └── saved_models.pkl
└── pages/
		├── 1_Preprocessing.py
		├── 2_Clustering.py
		└── 3_Classification.py

## Installation

1. Clone the repository

```bash
git clone https://github.com/yakoubham23/data-mining-pipeline-app.git
cd data-mining-pipeline-app
```

2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Example Usage

1. Open the Preprocessing page and upload your CSV.
2. Clean missing values and apply encoding/scaling.
3. Switch to Clustering and inspect elbow + silhouette + PCA clusters.
4. Switch to Classification, select target, train models, compare metrics.
5. Save the best model to models/saved_models.pkl.

## Screenshots (Placeholders)

Replace these placeholders with real screenshots before final submission.

![Home Dashboard](docs/screenshots/home-dashboard.png)
![Preprocessing Page](docs/screenshots/preprocessing-page.png)
![Clustering Page](docs/screenshots/clustering-page.png)
![Classification Page](docs/screenshots/classification-page.png)

## Deployment Notes (Streamlit Cloud)

- Ensure requirements.txt is present at repository root.
- Ensure app.py is at repository root.
- In Streamlit Cloud, set entrypoint to app.py.
- The app is compatible with standard Streamlit Cloud deployment.

## License

This project is for educational purposes as part of a university mini-project.
