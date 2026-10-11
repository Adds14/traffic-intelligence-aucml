"""
Force-Directed Sensor Network Visualization Module
===================================================
Generates interactive topological network visualizations of the PeMS04
traffic sensor network using spring-electrical force-directed embedding.
Adheres to academic standards: strictly preserves sensor graph connectivity
from road distances without fabricating fictitious GPS coordinates.
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_network_layout(project_root=None):
    """Load precomputed force-directed layout coordinates."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    layout_path = os.path.join(project_root, 'data', 'processed', 'network_layout.json')
    if os.path.exists(layout_path):
        with open(layout_path) as f:
            return json.load(f)
    return None


def get_network_edges(project_root=None):
    """Load road connectivity edges from distance.csv."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dist_path = os.path.join(project_root, 'data', 'raw', 'distance.csv')
    if os.path.exists(dist_path):
        return pd.read_csv(dist_path)
    return None


def get_sensor_connections(sensor_id, df_edges):
    """Retrieve incoming and outgoing road connections for a specific sensor."""
    if df_edges is None:
        return {'incoming': [], 'outgoing': []}

    sid = int(sensor_id)
    out_edges = df_edges[df_edges['from'] == sid]
    in_edges = df_edges[df_edges['to'] == sid]

    outgoing = [{'target': int(r['to']), 'distance_m': float(r['cost'])} for _, r in out_edges.iterrows()]
    incoming = [{'source': int(r['from']), 'distance_m': float(r['cost'])} for _, r in in_edges.iterrows()]

    return {'incoming': incoming, 'outgoing': outgoing}


def create_force_directed_network_figure(
    sensor_flow,
    thresholds=None,
    pos_dict=None,
    df_edges=None,
    selected_sensor=None,
    model_name="PDFormer",
    horizon_label="30 min",
    time_step=0
):
    """
    Generate an interactive Plotly force-directed network graph with
    a polished dark theme and accessible contrast.
    """
    if thresholds is None:
        thresholds = {'q25': 83.0, 'q50': 179.0, 'q75': 315.0}

    q25 = thresholds['q25']
    q50 = thresholds['q50']
    q75 = thresholds['q75']

    num_nodes = len(sensor_flow)
    flow_arr = np.array(sensor_flow, dtype=float)

    # Classify nodes
    categories = np.zeros(num_nodes, dtype=int)
    categories[flow_arr >= q25] = 1
    categories[flow_arr >= q50] = 2
    categories[flow_arr >= q75] = 3

    cat_names = ['Free Flow', 'Moderate', 'Heavy', 'Severe']
    cat_colors = ['#10b981', '#eab308', '#f97316', '#ef4444']  # Green, Yellow, Orange, Red

    fig = go.Figure()

    # 1. Plot Network Edges
    if df_edges is not None and pos_dict is not None:
        edge_x = []
        edge_y = []
        for _, row in df_edges.iterrows():
            u = str(int(row['from']))
            v = str(int(row['to']))
            if u in pos_dict and v in pos_dict:
                p1 = pos_dict[u]
                p2 = pos_dict[v]
                edge_x.extend([p1[0], p2[0], None])
                edge_y.extend([p1[1], p2[1], None])

        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(color='#475569', width=1.2),  # Visible light slate against dark surface
            opacity=0.6,
            hoverinfo='none',
            name='Road Network Links (340 Edges)',
            showlegend=True
        ))

    # 2. Plot Nodes by Congestion Category
    if pos_dict is not None:
        in_deg = {}
        out_deg = {}
        if df_edges is not None:
            for s in range(num_nodes):
                in_deg[s] = int((df_edges['to'] == s).sum())
                out_deg[s] = int((df_edges['from'] == s).sum())
        else:
            for s in range(num_nodes):
                in_deg[s] = 0
                out_deg[s] = 0

        for cat_idx in range(4):
            mask = (categories == cat_idx)
            count = mask.sum()
            if count == 0:
                continue

            node_indices = np.where(mask)[0]
            x_pts = [pos_dict[str(s)][0] for s in node_indices if str(s) in pos_dict]
            y_pts = [pos_dict[str(s)][1] for s in node_indices if str(s) in pos_dict]
            flows = [flow_arr[s] for s in node_indices if str(s) in pos_dict]
            sizes = [max(8, min(22, 7 + (f / 350.0) * 12)) for f in flows]

            hover_texts = [
                f"<b>Sensor #{s}</b><br>"
                f"Congestion: <span style='color:{cat_colors[cat_idx]}'><b>{cat_names[cat_idx]}</b></span><br>"
                f"Predicted Flow: <b>{flow_arr[s]:.1f}</b> veh/5min<br>"
                f"Network In-Degree: {in_deg[s]} | Out-Degree: {out_deg[s]}<br>"
                f"Forecast Horizon: {horizon_label}"
                for s in node_indices if str(s) in pos_dict
            ]

            fig.add_trace(go.Scatter(
                x=x_pts,
                y=y_pts,
                mode='markers',
                name=f"{cat_names[cat_idx]} ({count})",
                marker=dict(
                    size=sizes,
                    color=cat_colors[cat_idx],
                    line=dict(width=1.2, color='#0f172a'),
                    opacity=0.92
                ),
                text=hover_texts,
                hoverinfo='text'
            ))

        # 3. Highlight Selected Sensor
        if selected_sensor is not None and str(selected_sensor) in pos_dict:
            sel_s = int(selected_sensor)
            sel_p = pos_dict[str(sel_s)]
            sel_flow = flow_arr[sel_s]
            sel_cat = categories[sel_s]

            fig.add_trace(go.Scatter(
                x=[sel_p[0]],
                y=[sel_p[1]],
                mode='markers',
                name=f"Inspected: Sensor #{sel_s}",
                marker=dict(
                    size=28,
                    color='rgba(56, 189, 248, 0.25)',  # Cyan glow
                    line=dict(width=3, color='#38bdf8'),
                    symbol='circle'
                ),
                hoverinfo='text',
                text=[f"⭐ <b>Selected Sensor #{sel_s}</b><br>Flow: {sel_flow:.1f} veh/5min<br>{cat_names[sel_cat]}"]
            ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#111827',
        title=dict(
            text=f"<b>Force-Directed Sensor Network Topology</b> — {model_name} ({horizon_label}) [Step #{time_step}]",
            font=dict(size=15, color='#f8fafc')
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=620,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(17, 24, 39, 0.85)',
            bordercolor='#334155',
            borderwidth=1,
            font=dict(color='#cbd5e1', size=11)
        ),
        hoverlabel=dict(
            bgcolor='#1e293b',
            bordercolor='#334155',
            font_size=12,
            font_color='#ffffff',
            font_family="sans-serif"
        )
    )

    return fig
