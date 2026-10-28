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
    time_step=0,
    node_filter=None,
    dtw_matrix=None,
    show_semantic=False
):
    """
    Generate an interactive Plotly force-directed network graph with
    prominent highway connectivity, selected-node flow link illumination,
    and accessible contrast.
    """
    if thresholds is None:
        thresholds = {'q25': 83.0, 'q50': 179.0, 'q75': 315.0}

    q25 = thresholds['q25']
    q50 = thresholds['q50']
    q75 = thresholds['q75']

    num_nodes = len(sensor_flow)
    flow_arr = np.array(sensor_flow, dtype=float)

    # Filtered node set
    if node_filter is not None:
        visible_nodes = set(int(s) for s in node_filter)
    else:
        visible_nodes = set(range(num_nodes))

    # Classify nodes into 4 tiers
    categories = np.zeros(num_nodes, dtype=int)
    categories[flow_arr >= q25] = 1
    categories[flow_arr >= q50] = 2
    categories[flow_arr >= q75] = 3

    cat_names = ['Free Flow', 'Moderate', 'Heavy', 'Severe']
    cat_colors = ['#10b981', '#eab308', '#f97316', '#ef4444']  # Green, Yellow, Orange, Red

    fig = go.Figure()

    # 1. Plot Base Highway Network Edges (High Contrast Casing + Road Line)
    if df_edges is not None and pos_dict is not None:
        edge_x = []
        edge_y = []
        for _, row in df_edges.iterrows():
            u_id = int(row['from'])
            v_id = int(row['to'])
            if u_id in visible_nodes and v_id in visible_nodes:
                u = str(u_id)
                v = str(v_id)
                if u in pos_dict and v in pos_dict:
                    p1 = pos_dict[u]
                    p2 = pos_dict[v]
                    edge_x.extend([p1[0], p2[0], None])
                    edge_y.extend([p1[1], p2[1], None])

        if edge_x:
            # Road Casing (Dark base underlay for depth)
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(color='rgba(15, 23, 42, 0.95)', width=4.5),
                hoverinfo='none',
                showlegend=False
            ))
            # Prominent Road Line
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(color='rgba(148, 163, 184, 0.75)', width=2.2),
                hoverinfo='none',
                name=f'Highway Segments ({len(edge_x)//3} Links)',
                showlegend=True
            ))

    # 2. Optional: Plot PDFormer DTW Semantic Links (Shockwave Correlation)
    if show_semantic and dtw_matrix is not None and pos_dict is not None:
        sem_x = []
        sem_y = []
        top_k = 2
        for s in visible_nodes:
            # Get top-k semantic functional neighbors (excluding self)
            sim_scores = dtw_matrix[s].copy()
            sim_scores[s] = -np.inf
            top_neighbors = np.argsort(sim_scores)[-top_k:]
            for nb in top_neighbors:
                if nb in visible_nodes and sim_scores[nb] > 0:
                    u = str(s)
                    v = str(int(nb))
                    if u in pos_dict and v in pos_dict:
                        p1 = pos_dict[u]
                        p2 = pos_dict[v]
                        sem_x.extend([p1[0], p2[0], None])
                        sem_y.extend([p1[1], p2[1], None])

        if sem_x:
            fig.add_trace(go.Scatter(
                x=sem_x,
                y=sem_y,
                mode='lines',
                line=dict(color='rgba(192, 132, 252, 0.45)', width=1.4, dash='dot'),
                hoverinfo='none',
                name='DTW Semantic Graph (PDFormer Corridors)',
                showlegend=True
            ))

    # 3. Highlight Incident Links for Selected Sensor (Illuminated Connections)
    if selected_sensor is not None and str(selected_sensor) in pos_dict and df_edges is not None:
        sel_s = int(selected_sensor)
        sel_p = pos_dict[str(sel_s)]

        # Incoming edges: feeder nodes -> selected_sensor
        in_rows = df_edges[(df_edges['to'] == sel_s) & (df_edges['from'].isin(visible_nodes))]
        in_edge_x = []
        in_edge_y = []
        in_neighbors = []
        for _, r in in_rows.iterrows():
            u_id = int(r['from'])
            u = str(u_id)
            if u in pos_dict:
                pu = pos_dict[u]
                in_edge_x.extend([pu[0], sel_p[0], None])
                in_edge_y.extend([pu[1], sel_p[1], None])
                in_neighbors.append((u_id, pu[0], pu[1]))

        if in_edge_x:
            fig.add_trace(go.Scatter(
                x=in_edge_x,
                y=in_edge_y,
                mode='lines',
                line=dict(color='#38bdf8', width=3.8),
                name=f'Incoming Flow ⟶ Sensor #{sel_s} ({len(in_rows)} links)',
                hoverinfo='none',
                showlegend=True
            ))

        # Outgoing edges: selected_sensor -> receiver nodes
        out_rows = df_edges[(df_edges['from'] == sel_s) & (df_edges['to'].isin(visible_nodes))]
        out_edge_x = []
        out_edge_y = []
        out_neighbors = []
        for _, r in out_rows.iterrows():
            v_id = int(r['to'])
            v = str(v_id)
            if v in pos_dict:
                pv = pos_dict[v]
                out_edge_x.extend([sel_p[0], pv[0], None])
                out_edge_y.extend([sel_p[1], pv[1], None])
                out_neighbors.append((v_id, pv[0], pv[1]))

        if out_edge_x:
            fig.add_trace(go.Scatter(
                x=out_edge_x,
                y=out_edge_y,
                mode='lines',
                line=dict(color='#f59e0b', width=3.8),
                name=f'Outgoing Flow from Sensor #{sel_s} ({len(out_rows)} links)',
                hoverinfo='none',
                showlegend=True
            ))

        # Neighbor Halos
        for n_id, nx_pos, ny_pos in in_neighbors:
            fig.add_trace(go.Scatter(
                x=[nx_pos], y=[ny_pos],
                mode='markers',
                marker=dict(size=18, color='rgba(56, 189, 248, 0.25)', line=dict(color='#38bdf8', width=2)),
                hoverinfo='text',
                text=[f"🔗 <b>Upstream Feeder: Sensor #{n_id}</b> ⟶ #{sel_s}"],
                showlegend=False
            ))
        for n_id, nx_pos, ny_pos in out_neighbors:
            fig.add_trace(go.Scatter(
                x=[nx_pos], y=[ny_pos],
                mode='markers',
                marker=dict(size=18, color='rgba(245, 158, 11, 0.25)', line=dict(color='#f59e0b', width=2)),
                hoverinfo='text',
                text=[f"🔗 <b>Downstream Receiver: Sensor #{sel_s}</b> ⟶ #{n_id}"],
                showlegend=False
            ))

    # 4. Plot Nodes by Congestion Category
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
            node_indices = [s for s in np.where(mask)[0] if s in visible_nodes]
            count = len(node_indices)
            if count == 0:
                continue

            x_pts = [pos_dict[str(s)][0] for s in node_indices if str(s) in pos_dict]
            y_pts = [pos_dict[str(s)][1] for s in node_indices if str(s) in pos_dict]
            flows = [flow_arr[s] for s in node_indices if str(s) in pos_dict]
            # Balanced node sizes that do not obscure edges
            sizes = [max(7, min(16, 6 + (f / 350.0) * 8)) for f in flows]

            hover_texts = [
                f"<b>Sensor #{s}</b><br>"
                f"Congestion: <span style='color:{cat_colors[cat_idx]}'><b>{cat_names[cat_idx]}</b></span><br>"
                f"Predicted Flow: <b>{flow_arr[s]:.1f}</b> veh/5min<br>"
                f"Highway Connectivity: In: {in_deg[s]} | Out: {out_deg[s]}<br>"
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
                    opacity=0.94
                ),
                text=hover_texts,
                hoverinfo='text'
            ))

        # 5. Highlight Selected Sensor Node
        if selected_sensor is not None and str(selected_sensor) in pos_dict and int(selected_sensor) in visible_nodes:
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
                    size=24,
                    color='rgba(56, 189, 248, 0.3)',
                    line=dict(width=3, color='#38bdf8'),
                    symbol='circle'
                ),
                hoverinfo='text',
                text=[f"⭐ <b>Inspected Sensor #{sel_s}</b><br>Flow: {sel_flow:.1f} veh/5min<br>{cat_names[sel_cat]}"]
            ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#111827',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=620,
        margin=dict(l=10, r=10, t=25, b=60),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(17, 24, 39, 0.95)',
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
