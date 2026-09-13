from __future__ import annotations

import pandas as pd
import numpy as np


def load_and_clean_trajectory_data(path: str) -> pd.DataFrame:
    """Load trajectory CSV and perform basic cleaning."""
    df = pd.read_csv(path)

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str),
            errors='coerce'
        )

    df = df.dropna(subset=['datetime', 'latitude', 'longitude'])
    df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
    df = df.drop_duplicates()

    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month
    df['weekend'] = df['datetime'].dt.dayofweek >= 5

    return df.reset_index(drop=True)


def add_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create movement-related features."""
    df = df.sort_values(['user_id', 'datetime']).copy()
    df['lat_lag'] = df.groupby('user_id')['latitude'].shift(1)
    df['lon_lag'] = df.groupby('user_id')['longitude'].shift(1)
    df['time_lag'] = df.groupby('user_id')['datetime'].shift(1)

    df['distance_to_prev_km'] = np.sqrt(
        (df['latitude'] - df['lat_lag']) ** 2 + (df['longitude'] - df['lon_lag']) ** 2
    ) * 111.32

    time_delta_seconds = (df['datetime'] - df['time_lag']).dt.total_seconds()
    df['speed_kmh'] = np.where(
        time_delta_seconds > 0,
        (df['distance_to_prev_km'] / (time_delta_seconds / 3600.0)).fillna(0),
        0
    )

    df['distance_to_prev_km'] = df['distance_to_prev_km'].fillna(0)
    df['speed_kmh'] = df['speed_kmh'].fillna(0)

    df['cum_distance_km'] = df.groupby('user_id')['distance_to_prev_km'].cumsum()
    df['avg_speed_kmh'] = df.groupby('user_id')['speed_kmh'].transform('mean')
    df['max_speed_kmh'] = df.groupby('user_id')['speed_kmh'].transform('max')

    return df


def build_location_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create location summary for frequent places."""
    location_summary = df.groupby(['user_id', 'latitude', 'longitude']).agg(
        visit_count=('datetime', 'count'),
        first_seen=('datetime', 'min'),
        last_seen=('datetime', 'max'),
        avg_speed=('speed_kmh', 'mean'),
        total_time_minutes=('datetime', lambda s: (s.max() - s.min()).total_seconds() / 60.0)
    ).reset_index()
    return location_summary
