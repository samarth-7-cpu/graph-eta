"""
Train and serialize the Baseline + Graph-Enhanced XGBoost models.

Run once:
    python src/train_and_save_models.py

Saves to outputs/:
    baseline_xgb.pkl, graph_xgb.pkl, le_src.pkl, le_dest.pkl,
    corridor_graph.pkl, hub_features.pkl
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pickle
import networkx as nx

# ─── PATHS ───────────────────────────────────────────
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "raw" / "delivery_data.csv"
hub_path = project_root / "data" / "processed" / "graph_features.csv"
corridor_path = project_root / "data" / "processed" / "corridor_stats.csv"
output_dir = project_root / "outputs"
output_dir.mkdir(exist_ok=True)

# ─── 1. LOAD DATA ───────────────────────────────────
print("Loading data...")
df = pd.read_csv(data_path)
df = pd.get_dummies(df, columns=['route_type'], prefix='route', dtype=int)
df['od_start_time'] = pd.to_datetime(df['od_start_time'])
df['od_end_time'] = pd.to_datetime(df['od_end_time'])

# ─── 2. FEATURE ENGINEERING ─────────────────────────
df['hour'] = df['od_start_time'].dt.hour
df['day_of_week'] = df['od_start_time'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)


def time_bucket(h):
    if 5 <= h < 12:
        return 0  # morning
    elif 12 <= h < 18:
        return 1  # afternoon
    elif 18 <= h < 22:
        return 2  # evening
    else:
        return 3  # night


df['time_of_day'] = df['hour'].apply(time_bucket)
df['delay_ratio'] = df['actual_time'] / df['osrm_time']

# ─── 3. ENCODE CATEGORICALS ─────────────────────────
le_src = LabelEncoder()
le_dest = LabelEncoder()
df['source_enc'] = le_src.fit_transform(df['source_center'])
df['destination_enc'] = le_dest.fit_transform(df['destination_center'])

# ─── 4. BASELINE FEATURES & TARGET ──────────────────
TARGET = 'actual_time'

baseline_features = [
    'osrm_time', 'osrm_distance',
    'segment_osrm_time', 'segment_osrm_distance',
    'actual_distance_to_destination', 'start_scan_to_end_scan',
    'segment_factor', 'factor',
    'route_Carting', 'route_FTL',
    'is_cutoff', 'cutoff_factor',
    'hour', 'day_of_week', 'is_weekend', 'time_of_day',
    'source_enc', 'destination_enc',
]

# ─── 5. TRAIN / TEST SPLIT ──────────────────────────
train_df = df[df['data'] == 'training']
test_df = df[df['data'] == 'test']

X_train = train_df[baseline_features]
y_train = train_df[TARGET]
X_test = test_df[baseline_features]
y_test = test_df[TARGET]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ─── 6. TRAIN BASELINE MODEL ────────────────────────
print("Training Baseline XGBoost...")
baseline = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

mae_base = mean_absolute_error(y_test, y_pred_base)
print(f"Baseline MAE: {mae_base:.2f} mins")

# ─── 7. MAP HUB GRAPH FEATURES ──────────────────────
print("Loading hub features...")
hub_df = pd.read_csv(hub_path)
hub_index = hub_df.set_index('hub')

for prefix, col in [('src', 'source_center'), ('dst', 'destination_center')]:
    df[f'{prefix}_degree_centrality'] = df[col].map(hub_index['degree_centrality']).fillna(0)
    df[f'{prefix}_betweenness'] = df[col].map(hub_index['betweenness']).fillna(0)
    df[f'{prefix}_pagerank'] = df[col].map(hub_index['pagerank']).fillna(0)
    df[f'{prefix}_clustering'] = df[col].map(hub_index['clustering']).fillna(0)

# ─── 8. GRAPH FEATURES ──────────────────────────────
graph_features = baseline_features + [
    'src_degree_centrality', 'src_betweenness', 'src_pagerank', 'src_clustering',
    'dst_degree_centrality', 'dst_betweenness', 'dst_pagerank', 'dst_clustering',
]

train_df = df[df['data'] == 'training']
test_df = df[df['data'] == 'test']

X_graph_train = train_df[graph_features]
y_graph_train = train_df[TARGET]
X_graph_test = test_df[graph_features]
y_graph_test = test_df[TARGET]

# ─── 9. TRAIN GRAPH-ENHANCED MODEL ──────────────────
print("Training Graph-Enhanced XGBoost...")
graph_model = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
)
graph_model.fit(X_graph_train, y_graph_train)
y_pred_graph = graph_model.predict(X_graph_test)

mae_graph = mean_absolute_error(y_graph_test, y_pred_graph)
print(f"Graph-Enhanced MAE: {mae_graph:.2f} mins")

# ─── 10. BUILD AND SAVE CORRIDOR GRAPH ──────────────
print("Building corridor graph...")
corridor_df = pd.read_csv(corridor_path)
G = nx.DiGraph()
for _, row in corridor_df.iterrows():
    G.add_edge(
        row['source_center'], row['destination_center'],
        weight=row['delay_ratio'],
        avg_actual_time=row['avg_actual_time'],
        avg_osrm_time=row['avg_osrm_time'],
        trip_count=row['trip_count'],
    )
print(f"Corridor Graph — Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# ─── 11. SERIALIZE EVERYTHING ────────────────────────
print("Saving artifacts...")

artifacts = {
    'baseline_xgb.pkl': baseline,
    'graph_xgb.pkl': graph_model,
    'le_src.pkl': le_src,
    'le_dest.pkl': le_dest,
    'corridor_graph.pkl': G,
    'baseline_features.pkl': baseline_features,
    'graph_features.pkl': graph_features,
}

for fname, obj in artifacts.items():
    with open(output_dir / fname, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  Saved: {fname}")

# Save hub_df as pickle for fast loading
hub_df.to_pickle(output_dir / 'hub_features.pkl')
print("  Saved: hub_features.pkl")

# Save corridor_df as pickle
corridor_df.to_pickle(output_dir / 'corridor_stats.pkl')
print("  Saved: corridor_stats.pkl")

print(f"\n{'='*50}")
print(f"Baseline MAE       : {mae_base:.2f} mins")
print(f"Graph-Enhanced MAE : {mae_graph:.2f} mins")
print(f"Improvement        : {((mae_base - mae_graph) / mae_base) * 100:.2f}%")
print(f"{'='*50}")
print("\n[OK] All models and artifacts saved to outputs/")
