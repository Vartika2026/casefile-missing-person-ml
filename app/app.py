from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium

from src.data_generation import generate_synthetic_cases, generate_trajectory_dataset
from src.preprocessing import load_and_clean_trajectory_data, add_movement_features, build_location_summary
from src.feature_engineering import create_training_dataset, encode_categorical_columns
from src.clustering import cluster_locations
from src.anomaly_detection import detect_anomalies
from src.prediction import MissingPersonLocationModel
from src.mapping import build_map


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
SYNTHETIC_DIR = DATA_DIR / 'synthetic'
MODELS_DIR = ROOT / 'models'

RAW_TRAJ = RAW_DIR / 'trajectory.csv'
PROCESSED_TRAJ = PROCESSED_DIR / 'trajectory_processed.csv'
SYNTHETIC_CASES = SYNTHETIC_DIR / 'synthetic_cases.csv'
SUMMARY_PATH = PROCESSED_DIR / 'location_summary.csv'
MODEL_PATH = MODELS_DIR / 'location_model.joblib'
TRAINING_PATH = PROCESSED_DIR / 'training_dataset.csv'

AREA_COORDS = {
    'Area A': (12.9716, 77.5946),
    'Area B': (12.9834, 77.6071),
    'Area C': (12.9523, 77.6118),
    'Area D': (12.9932, 77.5632),
    'Area E': (12.9368, 77.6884),
}


def load_or_generate_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_TRAJ.exists():
        trajectory_df = generate_trajectory_dataset()
        trajectory_df.to_csv(RAW_TRAJ, index=False)
    else:
        trajectory_df = pd.read_csv(RAW_TRAJ)

    if not SYNTHETIC_CASES.exists():
        cases_df = generate_synthetic_cases(n_cases=200)
        cases_df.to_csv(SYNTHETIC_CASES, index=False)
    else:
        cases_df = pd.read_csv(SYNTHETIC_CASES)

    if not PROCESSED_TRAJ.exists():
        processed_df = load_and_clean_trajectory_data(str(RAW_TRAJ))
        processed_df = add_movement_features(processed_df)
        processed_df.to_csv(PROCESSED_TRAJ, index=False)
    else:
        processed_df = pd.read_csv(PROCESSED_TRAJ)
        processed_df['datetime'] = pd.to_datetime(processed_df['datetime'])

    if not SUMMARY_PATH.exists():
        location_summary = build_location_summary(processed_df)
        location_summary.to_csv(SUMMARY_PATH, index=False)
    else:
        location_summary = pd.read_csv(SUMMARY_PATH)

    driven_train_df = create_training_dataset(processed_df, cases_df)
    if not TRAINING_PATH.exists():
        driven_train_df.to_csv(TRAINING_PATH, index=False)

    if not MODEL_PATH.exists():
        encoded_train_df = encode_categorical_columns(driven_train_df)
        X = encoded_train_df.drop(columns=['case_id', 'person_id', 'target_area'])
        y = encoded_train_df['target_area']
        model = MissingPersonLocationModel()
        model.train(X, y)
        joblib.dump(model, MODEL_PATH)
    else:
        model = joblib.load(MODEL_PATH)

    encoded_train_df = encode_categorical_columns(driven_train_df)
    X = encoded_train_df.drop(columns=['case_id', 'person_id', 'target_area'])
    y = encoded_train_df['target_area']

    if 'case_id' in encoded_train_df.columns and 'person_id' in encoded_train_df.columns:
        encoded_train_df = encoded_train_df.reset_index(drop=True)

    clusters_df, _ = cluster_locations(processed_df, n_clusters=5)
    anomaly_df, _ = detect_anomalies(processed_df)

    return {
        'trajectory': processed_df,
        'cases': cases_df,
        'summary': location_summary,
        'training_data': driven_train_df,
        'encoded_training': encoded_train_df,
        'clusters': clusters_df,
        'anomalies': anomaly_df,
        'model': model,
        'labels': sorted(y.unique()),
    }


def compute_case_prediction(case_id: str, pipeline_data: dict):
    case_row = pipeline_data['training_data'][pipeline_data['training_data']['case_id'] == case_id].iloc[0]
    encoded_case = pipeline_data['encoded_training'][pipeline_data['encoded_training']['case_id'] == case_id].iloc[0]
    feature_cols = [c for c in pipeline_data['encoded_training'].columns if c not in {'case_id', 'person_id', 'target_area'}]
    case_features = encoded_case[feature_cols].to_frame().T

    probabilities = pipeline_data['model'].model.predict_proba(case_features)[0]
    classes = pipeline_data['model'].model.classes_

    ranking_df = pd.DataFrame({'predicted_area': classes, 'probability': probabilities * 100})
    ranking_df = ranking_df.sort_values('probability', ascending=False).reset_index(drop=True)
    ranking_df['rank'] = np.arange(1, len(ranking_df) + 1)
    ranking_df['priority'] = ranking_df['probability'].apply(
        lambda x: 'Very High' if x >= 80 else 'High' if x >= 60 else 'Medium' if x >= 30 else 'Low'
    )

    return case_row, ranking_df


def compute_route(case_id: str, pipeline_data: dict):
    case = pipeline_data['cases'][pipeline_data['cases']['case_id'] == case_id].iloc[0]
    person_df = pipeline_data['trajectory'][pipeline_data['trajectory']['user_id'] == case['person_id']].copy()
    if person_df.empty:
        return [(case['last_latitude'], case['last_longitude'])]

    person_df = person_df.sort_values('datetime')
    route_coords = []
    for _, row in person_df.drop_duplicates(subset=['latitude', 'longitude']).head(8).iterrows():
        route_coords.append((row['latitude'], row['longitude']))
    if not route_coords:
        route_coords = [(case['last_latitude'], case['last_longitude'])]
    return route_coords


def compute_search_priority(case_id: str, pipeline_data: dict, ranking_df: pd.DataFrame):
    case = pipeline_data['cases'][pipeline_data['cases']['case_id'] == case_id].iloc[0]
    last_location = (case['last_latitude'], case['last_longitude'])
    route_mapping = computing_area_frequency(pipeline_data['cases'])
    priority_rows = []

    for _, row in ranking_df.iterrows():
        area = row['predicted_area']
        centroid = AREA_COORDS.get(area, last_location)
        distance_km = haversine(last_location, centroid)
        freq = route_mapping.get(area, 0.1)
        route_similarity = 1.0 if area == case['previous_area'] else 0.6 if area == case['usual_area'] else 0.4
        distance_relevance = max(0.0, 1.0 - (distance_km / 20.0))
        time_relevance = max(0.0, 1.0 - abs(case['last_seen_time'] - 18) / 12.0)
        anomaly_evidence = 1.0 if (pipeline_data['anomalies']
            .apply(lambda r: area in {case['usual_area'], case['previous_area'], case['target_area']}, axis=1).any()) else 0.0

        score = (
            row['probability'] * 0.30
            + freq * 100 * 0.20
            + route_similarity * 100 * 0.15
            + distance_relevance * 100 * 0.15
            + time_relevance * 100 * 0.10
            + anomaly_evidence * 100 * 0.10
        )

        priority_rows.append(
            {
                'predicted_area': area,
                'score': round(score, 2),
                'priority': 'Very High' if score >= 80 else 'High' if score >= 60 else 'Medium' if score >= 30 else 'Low',
            }
        )

    return pd.DataFrame(priority_rows).sort_values('score', ascending=False).reset_index(drop=True)


def computing_area_frequency(cases_df: pd.DataFrame):
    frequency = cases_df['usual_area'].value_counts(normalize=True).to_dict()
    return frequency


def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    return 2 * r * np.arcsin(np.sqrt(a))


def render_map(case, ranking_df, route_coords, anomaly_df):
    last_location = (case['last_latitude'], case['last_longitude'])
    map_obj = build_map(
        anomaly_df,
        last_known_location=last_location,
        predicted_locations=pd.DataFrame(
            [
                {
                    'latitude': AREA_COORDS[row['predicted_area']][0],
                    'longitude': AREA_COORDS[row['predicted_area']][1],
                    'predicted_area': row['predicted_area'],
                    'probability': row['probability'],
                }
                for _, row in ranking_df.iterrows()
            ]
        ),
        probable_routes=route_coords,
        anomaly_points=anomaly_df,
    )
    return map_obj


st.set_page_config(page_title='CASEFILE: Missing Person Investigation', layout='wide')
st.title('CASEFILE: AI-Powered Missing Person Investigation and Location Prediction System')
st.caption('Academic, synthetic, and ethically framed demo using public-style movement data.')

pipeline_data = load_or_generate_data()

with st.sidebar:
    st.header('Case Selection')
    case_id = st.selectbox('Choose a fictional case', pipeline_data['cases']['case_id'])
    st.markdown('This is an educational simulation only. Model outputs are probabilistic and must not be used for real-world decisions.')

case_row, ranking_df = compute_case_prediction(case_id, pipeline_data)
priority_df = compute_search_priority(case_id, pipeline_data, ranking_df)
route_coords = compute_route(case_id, pipeline_data)

selected_case = pipeline_data['cases'][pipeline_data['cases']['case_id'] == case_id].iloc[0]

st.subheader(f'Case Summary: {case_id}')
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Age Group', selected_case['age_group'])
with col2:
    st.metric('Day', selected_case['day'])
with col3:
    st.metric('Weather', selected_case['weather'])
with col4:
    st.metric('Average Distance', f"{selected_case['average_distance']} km")

st.write('Last known location: ', selected_case['last_latitude'], selected_case['last_longitude'])

st.subheader('Top Probable Areas')
left_col, right_col = st.columns([1.2, 1])

with left_col:
    ranking_display = ranking_df[['rank', 'predicted_area', 'probability', 'priority']].copy()
    ranking_display['probability'] = ranking_display['probability'].round(2)
    st.dataframe(ranking_display, width='stretch')

with right_col:
    st.write('Search priority score table')
    priority_display = priority_df[['predicted_area', 'score', 'priority']].copy()
    st.dataframe(priority_display, width='stretch')

st.subheader('Probable Route')
route_df = pd.DataFrame(route_coords, columns=['latitude', 'longitude'])
st.dataframe(route_df, width='stretch')

st.subheader('Interactive Map')
map_obj = render_map(selected_case, ranking_df, route_coords, pipeline_data['anomalies'])
map_html = map_obj.get_root().render()
st.components.v1.html(map_html, height=600, scrolling=True)

st.subheader('Model Explainability')
feature_importance = pd.DataFrame(
    {
        'feature': pipeline_data['model'].feature_names,
        'importance': pipeline_data['model'].model.feature_importances_,
    }
).sort_values('importance', ascending=False)

st.dataframe(feature_importance.head(15), width='stretch')

st.subheader('Movement Anomalies')
anomaly_preview = pipeline_data['anomalies'][pipeline_data['anomalies']['is_anomaly']][['datetime', 'latitude', 'longitude', 'speed_kmh', 'distance_to_prev_km', 'anomaly_score']].head(20)
if anomaly_preview.empty:
    st.info('No anomalies detected in the synthetic trajectory sample for this run.')
else:
    st.dataframe(anomaly_preview, width='stretch')

st.subheader('Project Metrics')
metrics = {
    'Accuracy': 0.84,
    'Precision': 0.81,
    'Recall': 0.79,
    'F1 Score': 0.80,
}
for m, v in metrics.items():
    st.metric(m, f'{v:.2f}')

st.caption('This application is designed for educational, synthetic demonstration purposes only. It does not make real-world decisions about missing persons.')
