from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans, DBSCAN


def cluster_locations(df: pd.DataFrame, n_clusters: int = 6) -> tuple[pd.DataFrame, KMeans]:
    """Cluster location coordinates with KMeans."""
    coords = df[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(coords)
    coords['cluster_label'] = labels
    coords['cluster_centroid_lat'] = model.cluster_centers_[labels, 0]
    coords['cluster_centroid_lon'] = model.cluster_centers_[labels, 1]
    return coords, model


def cluster_dbscan(df: pd.DataFrame, eps: float = 0.01, min_samples: int = 5) -> tuple[pd.DataFrame, DBSCAN]:
    """Cluster with DBSCAN for density-based grouping."""
    coords = df[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(coords)
    coords['cluster_label'] = labels
    return coords, model
