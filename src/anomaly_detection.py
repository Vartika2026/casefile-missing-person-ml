from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def detect_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, IsolationForest]:
    """Detect anomalies using Isolation Forest."""
    features = df[['latitude', 'longitude', 'speed_kmh', 'distance_to_prev_km']].fillna(0)
    model = IsolationForest(contamination=0.05, random_state=42)
    labels = model.fit_predict(features)

    anomaly_df = df.copy()
    anomaly_df['anomaly_score'] = -model.decision_function(features)
    anomaly_df['is_anomaly'] = labels == -1
    return anomaly_df, model


def detect_lof(df: pd.DataFrame) -> tuple[pd.DataFrame, LocalOutlierFactor]:
    """Detect anomalies using Local Outlier Factor."""
    features = df[['latitude', 'longitude', 'speed_kmh']].fillna(0)
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    labels = model.fit_predict(features)

    anomaly_df = df.copy()
    anomaly_df['lof_outlier'] = labels
    anomaly_df['is_lof_anomaly'] = labels == -1
    return anomaly_df, model
