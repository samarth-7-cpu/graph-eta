"""
Delhivery Logistics Network — Live Streamlit Dashboard
=======================================================
Launch:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pickle
import time
import random
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════
# 0. CONFIG & PATHS
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="Delhivery Network — Delay Risk Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ═══════════════════════════════════════════════════════
# 1. CUSTOM CSS — Premium Dark Theme
# ═══════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0ff;
    }

    /* ── Metric Cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30,30,60,0.85) 0%, rgba(20,20,50,0.95) 100%);
        border: 1px solid rgba(100,100,255,0.15);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8888cc !important;
        font-weight: 500;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.8rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15,15,35,0.6);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        color: #8888cc;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4a4ae8 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
    }

    /* ── Section Headers ── */
    .section-header {
        background: linear-gradient(135deg, rgba(74,74,232,0.15), rgba(124,58,237,0.10));
        border-left: 4px solid #7c3aed;
        border-radius: 0 12px 12px 0;
        padding: 12px 20px;
        margin: 16px 0;
        color: #c8c8ff;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* ── Risk Badges ── */
    .risk-low {
        background: linear-gradient(135deg, #065f46, #047857);
        color: #6ee7b7; padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 0.85rem; display: inline-block;
    }
    .risk-medium {
        background: linear-gradient(135deg, #92400e, #b45309);
        color: #fcd34d; padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 0.85rem; display: inline-block;
    }
    .risk-high {
        background: linear-gradient(135deg, #991b1b, #dc2626);
        color: #fca5a5; padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 0.85rem; display: inline-block;
    }

    /* ── Live Pulse Dot ── */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.7); }
        70% { box-shadow: 0 0 0 10px rgba(34,197,94,0); }
        100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
    }
    .live-dot {
        width: 12px; height: 12px; border-radius: 50%;
        background: #22c55e; display: inline-block;
        animation: pulse 2s infinite; margin-right: 8px;
        vertical-align: middle;
    }

    /* ── Dataframe styling ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Plotly chart containers ── */
    .stPlotlyChart { border-radius: 12px; overflow: hidden; }

    /* ── Hide default streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# 2. LOAD CACHED DATA & MODELS
# ═══════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load serialized ML models."""
    with open(OUTPUT_DIR / 'baseline_xgb.pkl', 'rb') as f:
        baseline = pickle.load(f)
    with open(OUTPUT_DIR / 'graph_xgb.pkl', 'rb') as f:
        graph_model = pickle.load(f)
    with open(OUTPUT_DIR / 'le_src.pkl', 'rb') as f:
        le_src = pickle.load(f)
    with open(OUTPUT_DIR / 'le_dest.pkl', 'rb') as f:
        le_dest = pickle.load(f)
    with open(OUTPUT_DIR / 'baseline_features.pkl', 'rb') as f:
        baseline_features = pickle.load(f)
    with open(OUTPUT_DIR / 'graph_features.pkl', 'rb') as f:
        graph_features = pickle.load(f)
    return baseline, graph_model, le_src, le_dest, baseline_features, graph_features


@st.cache_data
def load_data():
    """Load graph, hub features, and corridor stats."""
    with open(OUTPUT_DIR / 'corridor_graph.pkl', 'rb') as f:
        G = pickle.load(f)
    hub_df = pd.read_pickle(OUTPUT_DIR / 'hub_features.pkl')
    corridor_df = pd.read_pickle(OUTPUT_DIR / 'corridor_stats.pkl')
    return G, hub_df, corridor_df


def check_artifacts_exist():
    required = [
        'baseline_xgb.pkl', 'graph_xgb.pkl', 'le_src.pkl', 'le_dest.pkl',
        'corridor_graph.pkl', 'hub_features.pkl', 'corridor_stats.pkl',
        'baseline_features.pkl', 'graph_features.pkl',
    ]
    missing = [f for f in required if not (OUTPUT_DIR / f).exists()]
    return missing


missing = check_artifacts_exist()
if missing:
    st.error(f"❌ Missing model artifacts: {', '.join(missing)}")
    st.info("Run `python src/train_and_save_models.py` first to train and save models.")
    st.stop()

baseline_model, graph_model, le_src, le_dest, baseline_features, graph_features = load_models()
G, hub_df, corridor_df = load_data()
hub_index = hub_df.set_index('hub')


# ═══════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════

def get_risk_badge(delay_ratio):
    if delay_ratio < 1.15:
        return '<span class="risk-low">● Low Risk</span>'
    elif delay_ratio < 1.35:
        return '<span class="risk-medium">▲ Medium Risk</span>'
    else:
        return '<span class="risk-high">◆ High Risk</span>'


def get_risk_label(delay_ratio):
    if delay_ratio < 1.15:
        return "Low"
    elif delay_ratio < 1.35:
        return "Medium"
    else:
        return "High"


def get_risk_color(delay_ratio):
    if delay_ratio < 1.15:
        return "#22c55e"
    elif delay_ratio < 1.35:
        return "#f59e0b"
    else:
        return "#ef4444"


# ═══════════════════════════════════════════════════════
# 4. SIDEBAR
# ═══════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🚚 Delhivery Network")
    st.markdown("**Graph-Based ETA Dashboard**")
    st.markdown("---")

    st.markdown("### Network Stats")
    st.metric("Total Hubs", f"{G.number_of_nodes():,}")
    st.metric("Total Corridors", f"{G.number_of_edges():,}")

    avg_delay = corridor_df['delay_ratio'].mean()
    chronic_count = len(corridor_df[corridor_df['delay_ratio'] > 1.2])
    st.metric("Avg Delay Ratio", f"{avg_delay:.2f}x")
    st.metric("Chronic Corridors (>20%)", f"{chronic_count:,}")

    st.markdown("---")
    st.markdown("### Filters")
    min_betweenness = st.slider(
        "Min Hub Betweenness",
        min_value=0.0, max_value=0.20,
        value=0.005, step=0.001, format="%.3f",
        help="Filter to show only hubs above this betweenness centrality"
    )
    delay_filter = st.selectbox(
        "Corridor Filter",
        ["All Corridors", "Chronic (>20% delay)", "Severe (>50% delay)"],
        index=0
    )
    hub_search = st.text_input(
        "🔍 Search Hub ID",
        placeholder="e.g., IND562132AAA",
        help="Highlight a specific hub on the network"
    )

    st.markdown("---")
    st.markdown(
        f"<div style='color:#666; font-size:0.75rem; text-align:center;'>"
        f"Last updated: {datetime.now().strftime('%H:%M:%S')}</div>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════
# 5. MAIN CONTENT — TABS
# ═══════════════════════════════════════════════════════

st.markdown(
    "<h1 style='text-align:center; background: linear-gradient(90deg, #4a4ae8, #7c3aed, #ec4899);"
    " -webkit-background-clip: text; -webkit-text-fill-color: transparent;"
    " font-weight: 800; font-size: 2.2rem; margin-bottom: 4px;'>"
    "Delhivery Logistics Network</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#8888cc; font-size:1rem; margin-bottom:24px;'>"
    "Real-Time Delay Risk Monitoring & Graph-Enhanced ETA Prediction</p>",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 Network Explorer",
    "📡 Live Monitor",
    "🎯 ETA Calculator",
    "📊 Model Insights",
])


# ─────────────────────────────────────────────────
# TAB 1: NETWORK EXPLORER
# ─────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Interactive Logistics Network Graph</div>',
                unsafe_allow_html=True)

    # KPI row
    top5_betweenness = hub_df.nlargest(5, 'betweenness')
    severe_corridors = len(corridor_df[corridor_df['delay_ratio'] > 1.5])
    total_trips = int(corridor_df['trip_count'].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏢 Active Hubs", f"{G.number_of_nodes():,}")
    c2.metric("🛤️ Active Corridors", f"{G.number_of_edges():,}")
    c3.metric("⚠️ Severe Corridors", f"{severe_corridors:,}")
    c4.metric("📦 Total Trips", f"{total_trips:,}")

    # Build filtered subgraph
    filtered_hubs = hub_df[hub_df['betweenness'] >= min_betweenness]['hub'].tolist()
    subG = G.subgraph([n for n in G.nodes() if n in filtered_hubs]).copy()

    if delay_filter == "Chronic (>20% delay)":
        edges_to_remove = [(u, v) for u, v, d in subG.edges(data=True) if d.get('weight', 1) <= 1.2]
        subG.remove_edges_from(edges_to_remove)
        # Remove isolated nodes
        isolates = list(nx.isolates(subG))
        subG.remove_nodes_from(isolates)
    elif delay_filter == "Severe (>50% delay)":
        edges_to_remove = [(u, v) for u, v, d in subG.edges(data=True) if d.get('weight', 1) <= 1.5]
        subG.remove_edges_from(edges_to_remove)
        isolates = list(nx.isolates(subG))
        subG.remove_nodes_from(isolates)

    if subG.number_of_nodes() == 0:
        st.warning("No hubs match the current filters. Try lowering the betweenness threshold.")
    else:
        # Compute layout
        pos = nx.spring_layout(subG, seed=42, k=2.5, iterations=50)

        # Build Plotly traces
        # --- Edges ---
        edge_traces = []
        for u, v, d in subG.edges(data=True):
            w = d.get('weight', 1.0)
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            if w > 1.5:
                color = '#ef4444'
                width = 2.0
            elif w > 1.2:
                color = '#f59e0b'
                width = 1.2
            else:
                color = '#22c55e'
                width = 0.6

            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=width, color=color),
                opacity=0.5,
                hoverinfo='skip',
                showlegend=False,
            ))

        # --- Nodes ---
        node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
        betweenness_dict = hub_index['betweenness'].to_dict()
        top5_hubs = sorted(betweenness_dict, key=betweenness_dict.get, reverse=True)[:5]

        for node in subG.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            b = betweenness_dict.get(node, 0)
            pr = hub_index.loc[node, 'pagerank'] if node in hub_index.index else 0
            deg = hub_index.loc[node, 'degree_centrality'] if node in hub_index.index else 0

            node_text.append(
                f"<b>{node}</b><br>"
                f"Betweenness: {b:.4f}<br>"
                f"PageRank: {pr:.5f}<br>"
                f"Degree Centrality: {deg:.4f}"
            )
            node_size.append(8 + b * 600)

            if hub_search and hub_search.upper() in node.upper():
                node_color.append('#ec4899')  # pink highlight for search
            elif node in top5_hubs:
                node_color.append('#ef4444')  # red for top 5
            elif b > 0.02:
                node_color.append('#f59e0b')  # orange for high
            else:
                node_color.append('#6366f1')  # indigo for normal

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers',
            marker=dict(size=node_size, color=node_color,
                        line=dict(width=1, color='rgba(255,255,255,0.3)')),
            text=node_text, hoverinfo='text',
            showlegend=False,
        )

        # Assemble figure
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            plot_bgcolor='#0f0f23',
            paper_bgcolor='#0f0f23',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=40, b=0),
            height=620,
            title=dict(
                text=f"Network — {subG.number_of_nodes()} hubs · {subG.number_of_edges()} corridors",
                font=dict(color='#c8c8ff', size=14),
                x=0.5
            ),
            # Legend annotations
            annotations=[
                dict(x=0.01, y=0.99, xref='paper', yref='paper', showarrow=False,
                     text="<b>Legend</b>", font=dict(color='#c8c8ff', size=11)),
                dict(x=0.01, y=0.95, xref='paper', yref='paper', showarrow=False,
                     text="🔴 Top-5 Bottleneck  🟠 High Betweenness  🟣 Normal Hub",
                     font=dict(color='#8888cc', size=10)),
                dict(x=0.01, y=0.91, xref='paper', yref='paper', showarrow=False,
                     text="<span style='color:#ef4444'>━</span> Severe(>50%)  "
                          "<span style='color:#f59e0b'>━</span> Chronic(>20%)  "
                          "<span style='color:#22c55e'>━</span> On-Time",
                     font=dict(color='#8888cc', size=10)),
            ]
        )
        st.plotly_chart(fig, use_container_width=True, key="network_graph")

    # Top 5 bottleneck hubs table
    st.markdown('<div class="section-header">Top 5 Bottleneck Hubs</div>', unsafe_allow_html=True)
    top5_df = hub_df.nlargest(5, 'betweenness')[['hub', 'betweenness', 'pagerank', 'degree_centrality', 'clustering']]
    top5_df.columns = ['Hub ID', 'Betweenness', 'PageRank', 'Degree Centrality', 'Clustering']
    st.dataframe(top5_df, use_container_width=True, hide_index=True)

    # Top delayed corridors
    st.markdown('<div class="section-header">Most Delayed Corridors</div>', unsafe_allow_html=True)
    top_delayed = corridor_df.nlargest(10, 'delay_ratio')[
        ['source_center', 'destination_center', 'delay_ratio', 'avg_actual_time', 'avg_osrm_time', 'trip_count']
    ].copy()
    top_delayed.columns = ['Source', 'Destination', 'Delay Ratio', 'Avg Actual (min)', 'Avg OSRM (min)', 'Trips']
    top_delayed['Delay Ratio'] = top_delayed['Delay Ratio'].round(2)
    top_delayed['Avg Actual (min)'] = top_delayed['Avg Actual (min)'].round(1)
    top_delayed['Avg OSRM (min)'] = top_delayed['Avg OSRM (min)'].round(1)
    st.dataframe(top_delayed, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────
# TAB 2: LIVE SHIPMENT MONITOR
# ─────────────────────────────────────────────────
with tab2:
    st.markdown(
        '<div class="section-header">'
        '<span class="live-dot"></span> Live Shipment Monitor — Simulated Stream'
        '</div>',
        unsafe_allow_html=True
    )

    all_hubs = list(G.nodes())
    all_edges = list(G.edges(data=True))

    # Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        n_shipments = st.selectbox("Shipments per batch", [5, 10, 20], index=1)
    with ctrl_col2:
        refresh_interval = st.selectbox("Refresh interval (s)", [2, 3, 5], index=1)
    with ctrl_col3:
        st.markdown("")
        run_sim = st.toggle("▶️ Start Simulation", value=False)

    # Summary metrics placeholder
    metrics_placeholder = st.empty()
    # Table placeholder
    table_placeholder = st.empty()
    # Chart placeholder
    chart_placeholder = st.empty()

    if run_sim:
        # Initialize session state for cumulative data
        if 'sim_history' not in st.session_state:
            st.session_state.sim_history = []

        for iteration in range(50):  # max 50 iterations
            shipments = []
            for _ in range(n_shipments):
                # Pick a random real corridor
                edge = random.choice(all_edges)
                src, dst = edge[0], edge[1]
                edge_data = edge[2]

                # Simulate realistic features
                hour = random.randint(0, 23)
                dow = random.randint(0, 6)
                is_weekend = 1 if dow >= 5 else 0
                tod = 0 if 5 <= hour < 12 else (1 if 12 <= hour < 18 else (2 if 18 <= hour < 22 else 3))
                route_ftl = random.choice([0, 1])
                route_carting = 1 - route_ftl

                osrm_time = edge_data.get('avg_osrm_time', 60)
                # Add noise to osrm values
                osrm_time_noisy = max(5, osrm_time * random.uniform(0.8, 1.2))
                osrm_dist = osrm_time_noisy * random.uniform(0.8, 1.5)

                # Get graph features
                src_b = hub_index.loc[src, 'betweenness'] if src in hub_index.index else 0
                src_pr = hub_index.loc[src, 'pagerank'] if src in hub_index.index else 0
                src_dc = hub_index.loc[src, 'degree_centrality'] if src in hub_index.index else 0
                src_cl = hub_index.loc[src, 'clustering'] if src in hub_index.index else 0

                dst_b = hub_index.loc[dst, 'betweenness'] if dst in hub_index.index else 0
                dst_pr = hub_index.loc[dst, 'pagerank'] if dst in hub_index.index else 0
                dst_dc = hub_index.loc[dst, 'degree_centrality'] if dst in hub_index.index else 0
                dst_cl = hub_index.loc[dst, 'clustering'] if dst in hub_index.index else 0

                # Encode hubs
                try:
                    src_enc = le_src.transform([src])[0]
                except ValueError:
                    src_enc = 0
                try:
                    dst_enc = le_dest.transform([dst])[0]
                except ValueError:
                    dst_enc = 0

                features = {
                    'osrm_time': osrm_time_noisy,
                    'osrm_distance': osrm_dist,
                    'segment_osrm_time': osrm_time_noisy * random.uniform(0.3, 0.7),
                    'segment_osrm_distance': osrm_dist * random.uniform(0.3, 0.7),
                    'actual_distance_to_destination': osrm_dist * random.uniform(0.9, 1.1),
                    'start_scan_to_end_scan': osrm_time_noisy * random.uniform(1.0, 2.0),
                    'segment_factor': random.uniform(0.8, 2.5),
                    'factor': random.uniform(0.8, 2.5),
                    'route_Carting': route_carting,
                    'route_FTL': route_ftl,
                    'is_cutoff': random.choice([0, 1]),
                    'cutoff_factor': random.uniform(0.5, 1.5),
                    'hour': hour,
                    'day_of_week': dow,
                    'is_weekend': is_weekend,
                    'time_of_day': tod,
                    'source_enc': src_enc,
                    'destination_enc': dst_enc,
                    'src_degree_centrality': src_dc,
                    'src_betweenness': src_b,
                    'src_pagerank': src_pr,
                    'src_clustering': src_cl,
                    'dst_degree_centrality': dst_dc,
                    'dst_betweenness': dst_b,
                    'dst_pagerank': dst_pr,
                    'dst_clustering': dst_cl,
                }

                # Predict
                X_pred = pd.DataFrame([features])[graph_features]
                pred_time = graph_model.predict(X_pred)[0]
                delay_ratio = pred_time / max(osrm_time_noisy, 1)
                risk = get_risk_label(delay_ratio)

                shipments.append({
                    'Time': datetime.now().strftime('%H:%M:%S'),
                    'Source': src,
                    'Destination': dst,
                    'Route': 'FTL' if route_ftl else 'Carting',
                    'OSRM ETA (min)': round(osrm_time_noisy, 1),
                    'Predicted ETA (min)': round(max(pred_time, 0), 1),
                    'Delay Ratio': round(delay_ratio, 2),
                    'Risk': risk,
                })

            batch_df = pd.DataFrame(shipments)
            st.session_state.sim_history.extend(shipments)

            # Keep last 100
            if len(st.session_state.sim_history) > 100:
                st.session_state.sim_history = st.session_state.sim_history[-100:]

            history_df = pd.DataFrame(st.session_state.sim_history)

            # Summary metrics
            with metrics_placeholder.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("📦 Total Monitored", len(history_df))
                high_risk_count = len(history_df[history_df['Risk'] == 'High'])
                m2.metric("🔴 High Risk", high_risk_count)
                med_risk_count = len(history_df[history_df['Risk'] == 'Medium'])
                m3.metric("🟡 Medium Risk", med_risk_count)
                avg_dr = history_df['Delay Ratio'].mean()
                m4.metric("📈 Avg Delay Ratio", f"{avg_dr:.2f}x")

            # Live table (colored by risk)
            with table_placeholder.container():
                def color_risk(val):
                    if val == 'High':
                        return 'background-color: rgba(239,68,68,0.25); color: #fca5a5; font-weight: 600'
                    elif val == 'Medium':
                        return 'background-color: rgba(245,158,11,0.25); color: #fcd34d; font-weight: 600'
                    else:
                        return 'background-color: rgba(34,197,94,0.25); color: #6ee7b7; font-weight: 600'

                styled = batch_df.style.map(color_risk, subset=['Risk'])
                st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

            # Risk distribution chart
            with chart_placeholder.container():
                risk_counts = history_df['Risk'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
                fig_risk = go.Figure(go.Bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    marker_color=['#22c55e', '#f59e0b', '#ef4444'],
                    text=risk_counts.values,
                    textposition='outside',
                    textfont=dict(color='#c8c8ff', size=14, family='Inter'),
                ))
                fig_risk.update_layout(
                    title=dict(text="Cumulative Risk Distribution", font=dict(color='#c8c8ff', size=14)),
                    plot_bgcolor='#0f0f23', paper_bgcolor='#0f0f23',
                    xaxis=dict(title='', tickfont=dict(color='#8888cc')),
                    yaxis=dict(title=dict(text='Count', font=dict(color='#8888cc')),
                               tickfont=dict(color='#8888cc')),
                    height=300, margin=dict(l=40, r=20, t=50, b=30),
                )
                st.plotly_chart(fig_risk, use_container_width=True, key=f"risk_dist_{iteration}")

            time.sleep(refresh_interval)
    else:
        st.info("Toggle **▶️ Start Simulation** in the controls above to begin the live shipment stream.")


# ─────────────────────────────────────────────────
# TAB 3: ETA CALCULATOR
# ─────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🎯 ETA & Delay Risk Prediction Tool</div>',
                unsafe_allow_html=True)

    sorted_hubs = sorted(hub_index.index.tolist())

    col_a, col_b = st.columns(2)
    with col_a:
        src_hub = st.selectbox("Source Hub", sorted_hubs, index=0, key="calc_src")
        hour_input = st.slider("Departure Hour", 0, 23, 10)
        osrm_time_input = st.number_input("OSRM Time (min)", min_value=1.0, value=60.0, step=5.0)
        seg_osrm_time = st.number_input("Segment OSRM Time (min)", min_value=0.1, value=30.0, step=5.0)
        seg_factor_input = st.number_input("Segment Factor", min_value=0.1, value=1.0, step=0.1)

    with col_b:
        dst_hub = st.selectbox("Destination Hub", sorted_hubs, index=min(1, len(sorted_hubs) - 1), key="calc_dst")
        dow_input = st.selectbox("Day of Week", list(range(7)),
                                  format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
        osrm_dist_input = st.number_input("OSRM Distance (km)", min_value=1.0, value=100.0, step=10.0)
        seg_osrm_dist = st.number_input("Segment OSRM Distance (km)", min_value=0.1, value=50.0, step=5.0)
        route_type_input = st.selectbox("Route Type", ["FTL", "Carting"])

    if st.button("🚀 Predict ETA & Risk", type="primary", use_container_width=True):
        is_weekend = 1 if dow_input >= 5 else 0
        tod = 0 if 5 <= hour_input < 12 else (1 if 12 <= hour_input < 18 else (2 if 18 <= hour_input < 22 else 3))

        try:
            src_enc = le_src.transform([src_hub])[0]
        except ValueError:
            src_enc = 0
        try:
            dst_enc = le_dest.transform([dst_hub])[0]
        except ValueError:
            dst_enc = 0

        base_input = {
            'osrm_time': osrm_time_input,
            'osrm_distance': osrm_dist_input,
            'segment_osrm_time': seg_osrm_time,
            'segment_osrm_distance': seg_osrm_dist,
            'actual_distance_to_destination': osrm_dist_input,
            'start_scan_to_end_scan': osrm_time_input * 1.2,
            'segment_factor': seg_factor_input,
            'factor': seg_factor_input,
            'route_Carting': 1 if route_type_input == 'Carting' else 0,
            'route_FTL': 1 if route_type_input == 'FTL' else 0,
            'is_cutoff': 0,
            'cutoff_factor': 1.0,
            'hour': hour_input,
            'day_of_week': dow_input,
            'is_weekend': is_weekend,
            'time_of_day': tod,
            'source_enc': src_enc,
            'destination_enc': dst_enc,
        }

        # Graph features
        graph_input = {**base_input}
        for prefix, hub in [('src', src_hub), ('dst', dst_hub)]:
            if hub in hub_index.index:
                graph_input[f'{prefix}_degree_centrality'] = hub_index.loc[hub, 'degree_centrality']
                graph_input[f'{prefix}_betweenness'] = hub_index.loc[hub, 'betweenness']
                graph_input[f'{prefix}_pagerank'] = hub_index.loc[hub, 'pagerank']
                graph_input[f'{prefix}_clustering'] = hub_index.loc[hub, 'clustering']
            else:
                graph_input[f'{prefix}_degree_centrality'] = 0
                graph_input[f'{prefix}_betweenness'] = 0
                graph_input[f'{prefix}_pagerank'] = 0
                graph_input[f'{prefix}_clustering'] = 0

        X_base = pd.DataFrame([base_input])[baseline_features]
        X_graph = pd.DataFrame([graph_input])[graph_features]

        pred_base = max(baseline_model.predict(X_base)[0], 0)
        pred_graph = max(graph_model.predict(X_graph)[0], 0)
        delay_ratio_graph = pred_graph / max(osrm_time_input, 1)

        st.markdown("---")

        # Results cards
        r1, r2, r3 = st.columns(3)
        r1.metric("📐 OSRM Estimate", f"{osrm_time_input:.0f} min")
        r2.metric("📊 Baseline XGBoost", f"{pred_base:.1f} min",
                   delta=f"{pred_base - osrm_time_input:+.1f} min")
        r3.metric("🧠 Graph-Enhanced", f"{pred_graph:.1f} min",
                   delta=f"{pred_graph - osrm_time_input:+.1f} min")

        # Risk display
        st.markdown("---")
        risk_col1, risk_col2 = st.columns([1, 2])
        with risk_col1:
            st.markdown(f"### Delay Risk Score")
            st.markdown(f"# {delay_ratio_graph:.2f}x")
            st.markdown(get_risk_badge(delay_ratio_graph), unsafe_allow_html=True)

        with risk_col2:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=delay_ratio_graph,
                delta={'reference': 1.0, 'increasing': {'color': '#ef4444'}, 'decreasing': {'color': '#22c55e'}},
                gauge={
                    'axis': {'range': [0, 4], 'tickcolor': '#8888cc',
                             'tickfont': {'color': '#8888cc'}},
                    'bar': {'color': get_risk_color(delay_ratio_graph)},
                    'bgcolor': '#1a1a3e',
                    'steps': [
                        {'range': [0, 1.15], 'color': 'rgba(34,197,94,0.15)'},
                        {'range': [1.15, 1.35], 'color': 'rgba(245,158,11,0.15)'},
                        {'range': [1.35, 4], 'color': 'rgba(239,68,68,0.15)'},
                    ],
                    'threshold': {
                        'line': {'color': '#ffffff', 'width': 3},
                        'thickness': 0.8,
                        'value': delay_ratio_graph,
                    },
                },
                title={'text': "Delay Risk Ratio", 'font': {'color': '#c8c8ff', 'size': 14}},
                number={'font': {'color': '#ffffff', 'size': 28}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor='#0f0f23', height=280,
                margin=dict(l=30, r=30, t=60, b=10),
                font=dict(color='#c8c8ff'),
            )
            st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")

        # Hub info
        st.markdown("---")
        st.markdown('<div class="section-header">Hub Graph Features Used</div>', unsafe_allow_html=True)
        hub_info = []
        for label, hub in [("Source", src_hub), ("Destination", dst_hub)]:
            if hub in hub_index.index:
                row = hub_index.loc[hub]
                hub_info.append({
                    'Role': label, 'Hub': hub,
                    'Betweenness': f"{row['betweenness']:.4f}",
                    'PageRank': f"{row['pagerank']:.5f}",
                    'Degree Centrality': f"{row['degree_centrality']:.4f}",
                    'Clustering': f"{row['clustering']:.3f}",
                })
            else:
                hub_info.append({'Role': label, 'Hub': hub,
                                 'Betweenness': '0', 'PageRank': '0',
                                 'Degree Centrality': '0', 'Clustering': '0'})
        st.dataframe(pd.DataFrame(hub_info), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────
# TAB 4: MODEL INSIGHTS
# ─────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">📊 Model Performance & Feature Importance</div>',
                unsafe_allow_html=True)

    # Feature importance comparison
    imp_base = pd.Series(baseline_model.feature_importances_, index=baseline_features)
    imp_graph = pd.Series(graph_model.feature_importances_, index=graph_features)

    ins_col1, ins_col2 = st.columns(2)

    with ins_col1:
        st.markdown("#### Baseline Model — Top 15 Features")
        top15_base = imp_base.sort_values(ascending=True).tail(15)
        fig_imp1 = go.Figure(go.Bar(
            y=top15_base.index, x=top15_base.values,
            orientation='h',
            marker_color='#6366f1',
            text=[f"{v:.3f}" for v in top15_base.values],
            textposition='outside',
            textfont=dict(color='#c8c8ff', size=10),
        ))
        fig_imp1.update_layout(
            plot_bgcolor='#0f0f23', paper_bgcolor='#0f0f23',
            xaxis=dict(title=dict(text='Importance', font=dict(color='#8888cc')),
                       tickfont=dict(color='#8888cc')),
            yaxis=dict(tickfont=dict(color='#c8c8ff', size=10)),
            height=450, margin=dict(l=180, r=60, t=20, b=40),
        )
        st.plotly_chart(fig_imp1, use_container_width=True, key="imp_base")

    with ins_col2:
        st.markdown("#### Graph-Enhanced Model — Top 15 Features")
        top15_graph = imp_graph.sort_values(ascending=True).tail(15)

        # Highlight graph features with a different color
        graph_feat_names = [
            'src_degree_centrality', 'src_betweenness', 'src_pagerank', 'src_clustering',
            'dst_degree_centrality', 'dst_betweenness', 'dst_pagerank', 'dst_clustering',
        ]
        colors = ['#ec4899' if f in graph_feat_names else '#f59e0b' for f in top15_graph.index]

        fig_imp2 = go.Figure(go.Bar(
            y=top15_graph.index, x=top15_graph.values,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.3f}" for v in top15_graph.values],
            textposition='outside',
            textfont=dict(color='#c8c8ff', size=10),
        ))
        fig_imp2.update_layout(
            plot_bgcolor='#0f0f23', paper_bgcolor='#0f0f23',
            xaxis=dict(title=dict(text='Importance', font=dict(color='#8888cc')),
                       tickfont=dict(color='#8888cc')),
            yaxis=dict(tickfont=dict(color='#c8c8ff', size=10)),
            height=450, margin=dict(l=180, r=60, t=20, b=40),
        )
        st.plotly_chart(fig_imp2, use_container_width=True, key="imp_graph")

    st.markdown(
        "<p style='text-align:center; color:#666; font-size:0.8rem;'>"
        "🟣 Baseline features  ·  🩷 Graph-enhanced features (pink)</p>",
        unsafe_allow_html=True
    )

    # Delay ratio distribution across corridors
    st.markdown('<div class="section-header">Corridor Delay Ratio Distribution</div>',
                unsafe_allow_html=True)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=corridor_df['delay_ratio'],
        nbinsx=60,
        marker_color='#7c3aed',
        opacity=0.8,
        name='Delay Ratio',
    ))
    fig_dist.add_vline(x=1.0, line_dash="dash", line_color="#22c55e",
                       annotation_text="On-Time (1.0x)", annotation_font_color="#22c55e")
    fig_dist.add_vline(x=1.2, line_dash="dash", line_color="#f59e0b",
                       annotation_text="Chronic (1.2x)", annotation_font_color="#f59e0b")
    fig_dist.add_vline(x=1.5, line_dash="dash", line_color="#ef4444",
                       annotation_text="Severe (1.5x)", annotation_font_color="#ef4444")
    fig_dist.update_layout(
        plot_bgcolor='#0f0f23', paper_bgcolor='#0f0f23',
        xaxis=dict(title=dict(text='Delay Ratio (actual / OSRM)', font=dict(color='#8888cc')),
                   tickfont=dict(color='#8888cc'), range=[0, 8]),
        yaxis=dict(title=dict(text='# of Corridors', font=dict(color='#8888cc')),
                   tickfont=dict(color='#8888cc')),
        height=350, margin=dict(l=60, r=20, t=30, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True, key="delay_dist")

    # Scatter: Trip Count vs Delay Ratio
    st.markdown('<div class="section-header">Trip Volume vs Delay Severity</div>',
                unsafe_allow_html=True)
    fig_scatter = px.scatter(
        corridor_df,
        x='trip_count', y='delay_ratio',
        size='trip_count', color='delay_ratio',
        color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444'],
        hover_data=['source_center', 'destination_center'],
        labels={'trip_count': 'Trip Count', 'delay_ratio': 'Delay Ratio'},
        range_color=[0.5, 5],
    )
    fig_scatter.update_layout(
        plot_bgcolor='#0f0f23', paper_bgcolor='#0f0f23',
        xaxis=dict(tickfont=dict(color='#8888cc'),
                   title=dict(font=dict(color='#8888cc'))),
        yaxis=dict(tickfont=dict(color='#8888cc'),
                   title=dict(font=dict(color='#8888cc')),
                   range=[0, 8]),
        height=400, margin=dict(l=60, r=20, t=30, b=50),
        coloraxis_colorbar=dict(tickfont=dict(color='#8888cc'),
                                 title=dict(text='Delay', font=dict(color='#8888cc'))),
    )
    st.plotly_chart(fig_scatter, use_container_width=True, key="trip_scatter")
