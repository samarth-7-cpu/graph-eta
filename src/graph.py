import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─────────────────────────────────────────
# 0. PATHS
# ─────────────────────────────────────────

project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "raw" / "delivery_data.csv"
hub_path = project_root / "data" / "processed" / "graph_features.csv"
output_dir = project_root / "outputs"
output_dir.mkdir(exist_ok=True)

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(data_path)
df['delay_ratio'] = df['actual_time'] / df['osrm_time']

# ─────────────────────────────────────────
# 2. BUILD GRAPH
# ─────────────────────────────────────────

print("Building graph...")

# Edge weights = median delay ratio per corridor
edge_df = (
    df.groupby(['source_center', 'destination_center'])['delay_ratio']
    .median()
    .reset_index()
    .rename(columns={'delay_ratio': 'weight'})
)

G = nx.DiGraph()
for _, row in edge_df.iterrows():
    G.add_edge(row['source_center'], row['destination_center'],
               weight=row['weight'])

print(f"Graph — Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# ─────────────────────────────────────────
# 3. LOAD HUB FEATURES
# ─────────────────────────────────────────

hub_df = pd.read_csv(hub_path)
hub_index = hub_df.set_index('hub')

betweenness = hub_index['betweenness'].to_dict()
pagerank = nx.pagerank(G, weight='weight')

# ─────────────────────────────────────────
# 4. IDENTIFY KEY ELEMENTS
# ─────────────────────────────────────────

# Top 5 bottleneck hubs
top5_hubs = sorted(betweenness, key=betweenness.get, reverse=True)[:5]
print(f"\nTop 5 bottleneck hubs: {top5_hubs}")

# Chronic delay corridors (delay ratio > 1.2)
chronic_edges = [
    (u, v) for u, v, d in G.edges(data=True)
    if d['weight'] > 1.2
]
print(f"Chronic corridors (>20% delay): {len(chronic_edges)}")

# ─────────────────────────────────────────
# 5. COMPUTE LAYOUT
# Use spring layout but only on top connected nodes
# to keep the graph readable
# ─────────────────────────────────────────

print("\nComputing layout (this may take a moment)...")

# Keep only top 100 nodes by betweenness for readability
top_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:100]
subG = G.subgraph(top_nodes).copy()

pos = nx.spring_layout(subG, seed=42, k=2)

# ─────────────────────────────────────────
# 6. PREPARE NODE SIZES & COLORS
# ─────────────────────────────────────────

nodes = list(subG.nodes())

# Node size = betweenness centrality (scaled)
node_sizes = []
for n in nodes:
    b = betweenness.get(n, 0)
    node_sizes.append(300 + b * 15000)  # scale up for visibility

# Node color
node_colors = []
for n in nodes:
    if n in top5_hubs:
        node_colors.append('#e74c3c')  # red = top 5 bottleneck
    elif betweenness.get(n, 0) > 0.02:
        node_colors.append('#e67e22')  # orange = high betweenness
    else:
        node_colors.append('#3498db')  # blue = normal hub

# ─────────────────────────────────────────
# 7. PREPARE EDGE COLORS
# ─────────────────────────────────────────

edges = list(subG.edges(data=True))
edge_colors = []
edge_widths = []

for u, v, d in edges:
    w = d.get('weight', 1.0)
    if w > 1.5:
        edge_colors.append('#e74c3c')  # red = severely delayed
        edge_widths.append(2.0)
    elif w > 1.2:
        edge_colors.append('#f39c12')  # orange = moderately delayed
        edge_widths.append(1.5)
    else:
        edge_colors.append('#2ecc71')  # green = on time
        edge_widths.append(0.8)

# ─────────────────────────────────────────
# 8. PLOT FULL NETWORK
# ─────────────────────────────────────────

print("Plotting full network visualization...")

fig, ax = plt.subplots(figsize=(20, 14))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Draw edges
nx.draw_networkx_edges(
    subG, pos,
    edgelist=[(u, v) for u, v, d in edges],
    edge_color=edge_colors,
    width=edge_widths,
    alpha=0.6,
    arrows=True,
    arrowsize=8,
    ax=ax
)

# Draw nodes
nx.draw_networkx_nodes(
    subG, pos,
    nodelist=nodes,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.9,
    ax=ax
)

# Label only top 5 hubs
top5_labels = {n: n for n in top5_hubs if n in subG.nodes()}
nx.draw_networkx_labels(
    subG, pos,
    labels=top5_labels,
    font_size=7,
    font_color='white',
    font_weight='bold',
    ax=ax
)

# ─────────────────────────────────────────
# 9. LEGEND
# ─────────────────────────────────────────

legend_elements = [
    mpatches.Patch(color='#e74c3c', label='Top 5 Bottleneck Hubs'),
    mpatches.Patch(color='#e67e22', label='High Betweenness Hubs'),
    mpatches.Patch(color='#3498db', label='Normal Hubs'),
    mpatches.Patch(color='#e74c3c', label='Severely Delayed Corridor (>50%)'),
    mpatches.Patch(color='#f39c12', label='Chronically Delayed Corridor (>20%)'),
    mpatches.Patch(color='#2ecc71', label='On-Time Corridor'),
]

ax.legend(
    handles=legend_elements,
    loc='upper left',
    fontsize=9,
    facecolor='#2c3e50',
    labelcolor='white',
    edgecolor='white',
    framealpha=0.8
)

plt.title(
    "Delhivery Logistics Network — Bottleneck Hubs & Delay Corridors\n"
    "(Node size = Betweenness Centrality | Edge color = Delay Severity)",
    fontsize=14,
    color='white',
    fontweight='bold',
    pad=20
)

plt.axis('off')
plt.tight_layout()
plt.savefig(output_dir / "network_visualization.png", dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close()
print("Saved: network_visualization.png")

# ─────────────────────────────────────────
# 10. CHRONIC CORRIDORS ONLY PLOT
# ─────────────────────────────────────────

print("\nPlotting chronic corridors...")

# Build subgraph of only chronic corridors
chronic_G = nx.DiGraph()
for u, v, d in G.edges(data=True):
    if d['weight'] > 1.2:
        chronic_G.add_edge(u, v, weight=d['weight'])

# Keep top 80 nodes by betweenness
chronic_nodes = sorted(
    [n for n in chronic_G.nodes() if n in betweenness],
    key=lambda n: betweenness.get(n, 0),
    reverse=True
)[:80]

chronic_subG = chronic_G.subgraph(chronic_nodes).copy()
pos2 = nx.spring_layout(chronic_subG, seed=42, k=2.5)

# Edge colors by severity
c_edges = list(chronic_subG.edges(data=True))
c_colors = ['#e74c3c' if d['weight'] > 1.5 else '#f39c12'
            for u, v, d in c_edges]
c_widths = [2.5 if d['weight'] > 1.5 else 1.5
            for u, v, d in c_edges]

# Node colors
c_nodes = list(chronic_subG.nodes())
c_node_colors = ['#e74c3c' if n in top5_hubs else '#e67e22'
                 for n in c_nodes]
c_node_sizes = [300 + betweenness.get(n, 0) * 15000 for n in c_nodes]

fig2, ax2 = plt.subplots(figsize=(16, 12))
fig2.patch.set_facecolor('#1a1a2e')
ax2.set_facecolor('#1a1a2e')

nx.draw_networkx_edges(
    chronic_subG, pos2,
    edge_color=c_colors,
    width=c_widths,
    alpha=0.7,
    arrows=True,
    arrowsize=10,
    ax=ax2
)

nx.draw_networkx_nodes(
    chronic_subG, pos2,
    node_color=c_node_colors,
    node_size=c_node_sizes,
    alpha=0.9,
    ax=ax2
)

# Label top 5 hubs
top5_chronic_labels = {n: n for n in top5_hubs if n in chronic_subG.nodes()}
nx.draw_networkx_labels(
    chronic_subG, pos2,
    labels=top5_chronic_labels,
    font_size=7,
    font_color='white',
    font_weight='bold',
    ax=ax2
)

legend2 = [
    mpatches.Patch(color='#e74c3c', label='Severely Delayed (>50% over OSRM)'),
    mpatches.Patch(color='#f39c12', label='Chronically Delayed (>20% over OSRM)'),
    mpatches.Patch(color='#e74c3c', label='Top 5 Bottleneck Hub'),
    mpatches.Patch(color='#e67e22', label='High Risk Hub'),
]

ax2.legend(
    handles=legend2,
    loc='upper left',
    fontsize=9,
    facecolor='#2c3e50',
    labelcolor='white',
    edgecolor='white'
)

plt.title(
    "Chronic Delay Corridors — Delhivery Logistics Network\n"
    "(Only corridors where actual time exceeds OSRM by >20%)",
    fontsize=13,
    color='white',
    fontweight='bold',
    pad=20
)

plt.axis('off')
plt.tight_layout()
plt.savefig(output_dir / "chronic_delay_corridors.png", dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("Saved: chronic_delay_corridors.png")

print("\n✅ All network visualizations done!")
print(f"Check your outputs folder: {output_dir}")