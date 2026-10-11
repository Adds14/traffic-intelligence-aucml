"""
Congestion Mapping Module
=========================
Generates sensor-level congestion maps from predicted traffic flow values.

Features:
1. Empirical Quantile-Based Classification:
   - Free Flow (Q < 25%)
   - Moderate  (25% <= Q < 50%)
   - Heavy     (50% <= Q < 75%)
   - Severe    (Q >= 75%)
2. Interactive Geographic Map (OpenStreetMap & CartoDB tiles with Folium)
3. Topological Network Layout Visualization
4. 2D Flow Heatmap Matrix
"""

import os
import json
import numpy as np
import pandas as pd


class CongestionMapper:
    """Map predicted traffic flow to congestion categories."""
    
    def __init__(self, train_flow=None, thresholds=None):
        if thresholds is not None:
            self.thresholds = thresholds
        elif train_flow is not None:
            self.thresholds = self._compute_thresholds(train_flow)
        else:
            self.thresholds = {'q25': 83.0, 'q50': 179.0, 'q75': 315.0}
        
        self.categories = ['Free Flow', 'Moderate', 'Heavy', 'Severe']
        self.colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']  # Green, Yellow, Orange, Red
    
    def _compute_thresholds(self, train_flow):
        nonzero = train_flow[train_flow > 0]
        return {
            'q25': float(np.percentile(nonzero, 25)),
            'q50': float(np.percentile(nonzero, 50)),
            'q75': float(np.percentile(nonzero, 75)),
        }
    
    def classify(self, flow_values):
        categories = np.zeros(flow_values.shape, dtype=int)
        categories[flow_values >= self.thresholds['q25']] = 1
        categories[flow_values >= self.thresholds['q50']] = 2
        categories[flow_values >= self.thresholds['q75']] = 3
        labels = np.array(self.categories)[categories]
        return categories, labels
    
    def get_sensor_congestion(self, predicted_flow, sensor_ids=None):
        if sensor_ids is None:
            sensor_ids = list(range(len(predicted_flow)))
        categories, labels = self.classify(predicted_flow)
        sensor_info = []
        for i, (sid, flow_val, cat, label) in enumerate(zip(sensor_ids, predicted_flow, categories, labels)):
            sensor_info.append({
                'sensor_id': int(sid),
                'sensor_index': i,
                'predicted_flow': float(flow_val),
                'congestion_category': int(cat),
                'congestion_label': str(label),
                'color': self.colors[int(cat)],
            })
        return sensor_info
    
    def save_thresholds(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({
                'thresholds': self.thresholds,
                'categories': self.categories,
                'colors': self.colors,
                'description': 'Flow-based congestion thresholds (vehicles per 5 minutes)',
                'method': 'Quantile-based from training data distribution',
            }, f, indent=2)
    
    @classmethod
    def load_thresholds(cls, filepath):
        with open(filepath) as f:
            data = json.load(f)
        return cls(thresholds=data['thresholds'])


def create_folium_congestion_map(
    predicted_flow,
    coords_df,
    thresholds,
    adj_matrix=None,
    model_name="PDFormer",
    horizon_label="30 min",
    time_step=0
):
    """Generate an interactive Folium OpenStreetMap with real-time predicted congestion markers."""
    import folium

    center_lat = float(coords_df['latitude'].mean())
    center_lon = float(coords_df['longitude'].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )

    # Road network links
    if adj_matrix is not None:
        edges = np.argwhere(adj_matrix > 0)
        for i, j in edges:
            if i < j and i < len(coords_df) and j < len(coords_df):
                p1 = [float(coords_df.iloc[i]['latitude']), float(coords_df.iloc[i]['longitude'])]
                p2 = [float(coords_df.iloc[j]['latitude']), float(coords_df.iloc[j]['longitude'])]
                folium.PolyLine(
                    locations=[p1, p2],
                    color='#34495e',
                    weight=1.8,
                    opacity=0.45
                ).add_to(m)

    q25 = thresholds['q25']
    q50 = thresholds['q50']
    q75 = thresholds['q75']

    cat_names = ['Free Flow', 'Moderate', 'Heavy', 'Severe']
    cat_colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']

    for idx, row in coords_df.iterrows():
        sid = int(row['sensor_id'])
        if sid >= len(predicted_flow):
            continue
        flow_val = float(predicted_flow[sid])
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        corridor = row.get('corridor', 'Freeway')

        if flow_val < q25:
            cat_idx = 0
        elif flow_val < q50:
            cat_idx = 1
        elif flow_val < q75:
            cat_idx = 2
        else:
            cat_idx = 3

        color = cat_colors[cat_idx]
        label = cat_names[cat_idx]

        popup_html = f"""
        <div style='font-family: Arial, sans-serif; font-size: 13px; line-height: 1.5; min-width: 180px;'>
            <div style='background: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;'>
                Sensor #{sid} — {label}
            </div>
            <div style='margin-top: 8px;'>
                <b>Highway:</b> {corridor}<br>
                <b>Model:</b> {model_name}<br>
                <b>Horizon:</b> {horizon_label}<br>
                <b>Predicted Flow:</b> <span style='font-size: 15px; font-weight: bold; color: {color};'>{flow_val:.1f}</span> veh/5min<br>
                <b>Location:</b> {lat:.4f}, {lon:.4f}
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=6 + cat_idx * 2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1.5,
            tooltip=f"Sensor #{sid} | {flow_val:.1f} veh/5min ({label})",
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(m)

    return m
