from __future__ import annotations

import pandas as pd
import numpy as np


def create_training_dataset(df: pd.DataFrame, synthetic_cases: pd.DataFrame) -> pd.DataFrame:
    """Construct a supervised dataset for location prediction."""
    features = []

    for _, case in synthetic_cases.iterrows():
        person_df = df[df['user_id'] == case['person_id']].copy()
        if person_df.empty:
            continue

        nearby_locs = (
            person_df.groupby(['latitude', 'longitude'])
            .size()
            .reset_index(name='visit_count')
            .sort_values('visit_count', ascending=False)
        )

        top_locations = nearby_locs.head(5)
        top_lat = top_locations['latitude'].tolist()
        top_lon = top_locations['longitude'].tolist()
        top_counts = top_locations['visit_count'].tolist()

        row = {
            'case_id': case['case_id'],
            'person_id': case['person_id'],
            'age_group': case['age_group'],
            'gender': case['gender'],
            'day': case['day'],
            'weather': case['weather'],
            'last_latitude': case['last_latitude'],
            'last_longitude': case['last_longitude'],
            'hour': case['last_seen_time'],
            'average_distance': case['average_distance'],
            'average_speed': case['average_speed'],
            'usual_area': case['usual_area'],
            'previous_area': case['previous_area'],
            'time_since_last_seen': case['time_since_last_seen'],
            'top_location_1_lat': top_lat[0] if len(top_lat) > 0 else case['last_latitude'],
            'top_location_1_lon': top_lon[0] if len(top_lon) > 0 else case['last_longitude'],
            'top_location_2_lat': top_lat[1] if len(top_lat) > 1 else case['last_latitude'],
            'top_location_2_lon': top_lon[1] if len(top_lon) > 1 else case['last_longitude'],
            'top_location_3_lat': top_lat[2] if len(top_lat) > 2 else case['last_latitude'],
            'top_location_3_lon': top_lon[2] if len(top_lon) > 2 else case['last_longitude'],
            'top_location_1_count': top_counts[0] if len(top_counts) > 0 else 0,
            'top_location_2_count': top_counts[1] if len(top_counts) > 1 else 0,
            'top_location_3_count': top_counts[2] if len(top_counts) > 2 else 0,
            'target_area': case['target_area'],
        }
        features.append(row)

    dataset = pd.DataFrame(features)
    dataset['target_area'] = dataset['target_area'].astype(str)
    return dataset


def encode_categorical_columns(df: pd.DataFrame, categorical_cols: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode categorical columns for supervised modeling."""
    categorical_cols = categorical_cols or ['age_group', 'gender', 'day', 'weather', 'usual_area', 'previous_area']
    encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    return encoded


def add_route_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple route transition features based on repeated locations."""
    df = df.copy()
    df['prev_location_lat'] = df.groupby('user_id')['latitude'].shift(1)
    df['prev_location_lon'] = df.groupby('user_id')['longitude'].shift(1)
    df['route_transition_count'] = df.groupby('user_id').cumcount()
    return df
