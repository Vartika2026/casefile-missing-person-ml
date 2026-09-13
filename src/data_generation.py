from __future__ import annotations

import pandas as pd
import numpy as np


def generate_synthetic_cases(n_cases: int = 200, n_users: int = 12) -> pd.DataFrame:
    """Create fictional missing-person case records."""
    rng = np.random.default_rng(42)
    case_ids = [f'MP-2026-{i:03d}' for i in range(1, n_cases + 1)]

    age_groups = ['18-25', '26-35', '36-45', '46-55', '56+']
    genders = ['Female', 'Male', 'Non-binary']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weather = ['Clear', 'Rain', 'Cloudy', 'Storm']
    usual_area = ['Area A', 'Area B', 'Area C', 'Area D', 'Area E']
    previous_area = ['Area A', 'Area B', 'Area C', 'Area D', 'Area E']
    target_area = ['Area A', 'Area B', 'Area C', 'Area D', 'Area E']

    data = []
    for i in range(n_cases):
        case = {
            'case_id': case_ids[i],
            'person_id': int(rng.integers(1, n_users + 1)),
            'age_group': rng.choice(age_groups),
            'gender': rng.choice(genders),
            'last_latitude': round(rng.uniform(12.9, 13.2), 5),
            'last_longitude': round(rng.uniform(77.5, 77.8), 5),
            'last_seen_time': int(rng.integers(8, 22)),
            'day': rng.choice(days),
            'weather': rng.choice(weather),
            'usual_area': rng.choice(usual_area),
            'average_distance': round(rng.uniform(5, 18), 2),
            'average_speed': round(rng.uniform(3, 14), 2),
            'previous_area': rng.choice(previous_area),
            'time_since_last_seen': int(rng.integers(1, 48)),
            'target_area': rng.choice(target_area),
        }
        data.append(case)

    return pd.DataFrame(data)


def generate_trajectory_dataset(n_users: int = 12, n_points_per_user: int = 200) -> pd.DataFrame:
    """Generate synthetic trajectory data for demonstration and local testing."""
    rng = np.random.default_rng(123)
    records = []

    base_locations = {
        'Area A': (12.9716, 77.5946),
        'Area B': (12.9834, 77.6071),
        'Area C': (12.9523, 77.6118),
        'Area D': (12.9932, 77.5632),
        'Area E': (12.9368, 77.6884),
    }

    for user_id in range(1, n_users + 1):
        area = list(base_locations.keys())[user_id % len(base_locations)]
        lat0, lon0 = base_locations[area]

        for idx in range(n_points_per_user):
            drift = rng.normal(0, 0.002)
            lat = min(max(lat0 + drift + (idx % 10) * 0.0003, 12.90), 13.10)
            lon = min(max(lon0 + rng.normal(0, 0.002) + (idx % 7) * 0.0002, 77.50), 77.80)

            records.append({
                'user_id': user_id,
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'date': pd.Timestamp('2026-01-01') + pd.Timedelta(days=idx % 30, hours=idx % 24),
                'time': (pd.Timestamp('2026-01-01') + pd.Timedelta(days=idx % 30, hours=idx % 24)).time(),
                'datetime': pd.Timestamp('2026-01-01') + pd.Timedelta(days=idx % 30, hours=idx % 24),
            })

    trajectory_df = pd.DataFrame(records)
    trajectory_df['datetime'] = pd.to_datetime(trajectory_df['datetime'])
    trajectory_df['date'] = trajectory_df['datetime'].dt.strftime('%Y-%m-%d')
    trajectory_df['time'] = trajectory_df['datetime'].dt.strftime('%H:%M:%S')
    return trajectory_df
