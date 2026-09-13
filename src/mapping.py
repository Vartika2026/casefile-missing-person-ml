from __future__ import annotations

import pandas as pd
import folium


def build_map(
    df: pd.DataFrame,
    last_known_location: tuple[float, float],
    predicted_locations: pd.DataFrame | None = None,
    probable_routes: list[tuple[float, float]] | None = None,
    anomaly_points: pd.DataFrame | None = None,
):
    """Build an interactive folium map."""
    m = folium.Map(location=[last_known_location[0], last_known_location[1]], zoom_start=12)

    folium.Marker(
        location=[last_known_location[0], last_known_location[1]],
        popup='Last Known Location',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    if anomaly_points is not None and not anomaly_points.empty:
        for _, row in anomaly_points[anomaly_points['is_anomaly']].iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup='Anomaly',
                icon=folium.Icon(color='darkred', icon='warning-sign')
            ).add_to(m)

    if predicted_locations is not None and not predicted_locations.empty:
        for _, row in predicted_locations.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=7,
                popup=f"{row['predicted_area']} ({row['probability']}%)",
                color='blue',
                fill=True,
                fill_opacity=0.6,
            ).add_to(m)

    if probable_routes:
        route_coords = list(probable_routes)
        folium.PolyLine(route_coords, color='green', weight=3, opacity=0.8).add_to(m)

    return m
