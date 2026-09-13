# CASEFILE: AI-Powered Missing Person Investigation and Location Prediction System

This repository contains a synthetic, academic Streamlit application for a missing-person investigation support system. It uses generated GPS trajectories, movement clustering, anomaly detection, location prediction, route prediction, and a search-priority scoring system.

## Project Overview

The goal of this project is to demonstrate how machine learning can support a fictional investigation workflow by:

- cleaning and processing trajectory data,
- identifying common movement patterns,
- detecting anomalies,
- predicting likely areas,
- ranking search-priority locations,
- and visualizing results on an interactive map.

This project is for educational and research use only. It uses synthetic data and should not be used for real-world decision-making.

## Folder Structure

```text
CASEFILE_MISSING_PERSON/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── __init__.py
│   ├── anomaly_detection.py
│   ├── clustering.py
│   ├── data_generation.py
│   ├── feature_engineering.py
│   ├── mapping.py
│   ├── prediction.py
│   └── preprocessing.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Features

- Synthetic trajectory data generation
- Data cleaning and preprocessing
- Movement feature engineering
- K-Means clustering for location grouping
- Isolation Forest for anomaly detection
- Random Forest location prediction
- Search-priority score generation
- Probable route representation
- Folium interactive map
- Streamlit dashboard

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Folium
- Joblib
- Plotly
- Seaborn

## Run the App Locally

### 1. Clone or open the project

```bash
cd /workspaces/casefile-missing-person-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit app

```bash
streamlit run app/app.py
```

If `streamlit` is not recognized, use:

```bash
python -m streamlit run app/app.py
```

### 5. Open the app

After the command starts, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example Run Command in This Environment

```bash
cd /workspaces/casefile-missing-person-ml
/home/codespace/.python/current/bin/python -m streamlit run app/app.py --server.headless true --server.port 8501
```

## Important Notes

- This project uses synthetic datasets only.
- Predicted areas are probabilistic and for demonstration purposes.
- The model should not be used for real-world missing-person investigations.
- Model outputs are intended to support academic learning and reporting.

## File Responsibilities

- `app/app.py`: main Streamlit dashboard
- `src/preprocessing.py`: load, clean, and transform trajectory data
- `src/feature_engineering.py`: create supervised training data
- `src/clustering.py`: perform movement clustering
- `src/anomaly_detection.py`: detect unusual movement patterns
- `src/prediction.py`: train and use the location prediction model
- `src/mapping.py`: build interactive map objects
- `src/data_generation.py`: generate synthetic case and trajectory datasets

## Optional Next Improvements

- Add real GeoLife dataset support
- Use SHAP for explainability
- Add confusion matrix and model metrics charts
- Add route transition probability analysis
- Convert the project into a more production-ready architecture

## License

This project is intended for academic use and demonstration only.
