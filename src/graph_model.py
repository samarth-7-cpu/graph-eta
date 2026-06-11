import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path


# 0. PATHS


project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "raw" / "delivery_data.csv"
hub_path = project_root / "data" / "processed" / "graph_features.csv"  # ← change to your hub file name
output_dir = project_root / "outputs"
output_dir.mkdir(exist_ok=True)


# 1. LOAD DATA


df = pd.read_csv(data_path)
df = pd.get_dummies(df, columns=['route_type'], prefix='route', dtype=int)
df['od_start_time'] = pd.to_datetime(df['od_start_time'])
df['od_end_time'] = pd.to_datetime(df['od_end_time'])

# 2. FEATURE ENGINEERING


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


# 3. ENCODE CATEGORICALS


le_src = LabelEncoder()
le_dest = LabelEncoder()

df['source_enc'] = le_src.fit_transform(df['source_center'])
df['destination_enc'] = le_dest.fit_transform(df['destination_center'])


# 4. BASELINE FEATURES & TARGET


TARGET = 'actual_time'

baseline_features = [
    'osrm_time',
    'osrm_distance',
    'segment_osrm_time',
    'segment_osrm_distance',
    'actual_distance_to_destination',
    'start_scan_to_end_scan',
    'segment_factor',
    'factor',
    'route_Carting',
    'route_FTL',
    'is_cutoff',
    'cutoff_factor',
    'hour',
    'day_of_week',
    'is_weekend',
    'time_of_day',
    'source_enc',
    'destination_enc',
]


# 5. TRAIN / TEST SPLIT (using data column)


train_df = df[df['data'] == 'training']
test_df = df[df['data'] == 'test']

X_train = train_df[baseline_features]
y_train = train_df[TARGET]
X_test = test_df[baseline_features]
y_test = test_df[TARGET]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 6. TRAIN BASELINE MODEL


baseline = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_test)


# 7. BASELINE RESULTS


mae_base = mean_absolute_error(y_test, y_pred)
w15_base = np.mean(np.abs(y_pred - y_test.values) / y_test.values < 0.15) * 100
mae_osrm = mean_absolute_error(y_test, X_test['osrm_time'])

print("=" * 40)
print("BASELINE MODEL RESULTS")
print("=" * 40)
print(f"MAE              : {mae_base:.2f} mins")
print(f"% within 15%     : {w15_base:.2f}%")
print(f"OSRM Baseline MAE: {mae_osrm:.2f} mins")
print("=" * 40)

# Feature importance plot
importances = pd.Series(baseline.feature_importances_, index=baseline_features)
importances.sort_values().plot(kind='barh', figsize=(8, 6), color='steelblue')
plt.title("Baseline Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(output_dir / "baseline_feature_importance.png", dpi=150)
plt.close()

# Save baseline predictions
results_df = X_test.copy()
results_df['actual_time'] = y_test.values
results_df['baseline_pred'] = y_pred
results_df['abs_error'] = np.abs(y_pred - y_test.values)
results_df.to_csv(output_dir / "baseline_predictions.csv", index=False)
print("Saved: baseline_predictions.csv")


# GRAPH-ENHANCED MODEL



# 8. LOAD & MAP HUB GRAPH FEATURES


print("\nLoading hub features...")
hub_df = pd.read_csv(hub_path)
hub_index = hub_df.set_index('hub')

# Source hub features
df['src_degree_centrality'] = df['source_center'].map(hub_index['degree_centrality']).fillna(0)
df['src_betweenness'] = df['source_center'].map(hub_index['betweenness']).fillna(0)
df['src_pagerank'] = df['source_center'].map(hub_index['pagerank']).fillna(0)
df['src_clustering'] = df['source_center'].map(hub_index['clustering']).fillna(0)

# Destination hub features
df['dst_degree_centrality'] = df['destination_center'].map(hub_index['degree_centrality']).fillna(0)
df['dst_betweenness'] = df['destination_center'].map(hub_index['betweenness']).fillna(0)
df['dst_pagerank'] = df['destination_center'].map(hub_index['pagerank']).fillna(0)
df['dst_clustering'] = df['destination_center'].map(hub_index['clustering']).fillna(0)

print("Hub features mapped!")
print(f"NaN check: {df[['src_betweenness', 'dst_betweenness']].isna().sum().sum()} nulls")


# 9. GRAPH FEATURE LIST


graph_features = baseline_features + [
    'src_degree_centrality',
    'src_betweenness',
    'src_pagerank',
    'src_clustering',
    'dst_degree_centrality',
    'dst_betweenness',
    'dst_pagerank',
    'dst_clustering',
]

# Re-split after adding graph features to df
train_df = df[df['data'] == 'training']
test_df = df[df['data'] == 'test']

X_graph_train = train_df[graph_features]
y_graph_train = train_df[TARGET]
X_graph_test = test_df[graph_features]
y_graph_test = test_df[TARGET]

print(f"\nGraph — Train: {X_graph_train.shape}, Test: {X_graph_test.shape}")

# 10. TRAIN GRAPH-ENHANCED MODEL


print("\nTraining Graph-Enhanced XGBoost...")

graph_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

graph_model.fit(X_graph_train, y_graph_train)
y_pred_graph = graph_model.predict(X_graph_test)

print("Graph model trained!")


# 11. COMPARE BOTH MODELS


mae_graph = mean_absolute_error(y_graph_test, y_pred_graph)
w15_graph = np.mean(np.abs(y_pred_graph - y_graph_test.values) / y_graph_test.values < 0.15) * 100

improvement_mae = ((mae_base - mae_graph) / mae_base) * 100
improvement_w15 = w15_graph - w15_base

print("\n" + "=" * 55)
print(f"{'Model':<25} {'MAE':>10} {'%within15':>14}")
print("=" * 55)
print(f"{'OSRM Raw':<25} {mae_osrm:>10.2f} {'N/A':>14}")
print(f"{'Baseline XGBoost':<25} {mae_base:>10.2f} {w15_base:>13.2f}%")
print(f"{'Graph-Enhanced XGBoost':<25} {mae_graph:>10.2f} {w15_graph:>13.2f}%")
print("=" * 55)
print(f"MAE improvement      : {improvement_mae:.2f}%")
print(f"%within15 improvement: {improvement_w15:.2f}pp")
print("=" * 55)


# 12. GRAPH FEATURE IMPORTANCE PLOT


imp_graph = pd.Series(graph_model.feature_importances_, index=graph_features)
imp_graph.sort_values().tail(20).plot(kind='barh', figsize=(9, 7), color='darkorange')
plt.title("Graph-Enhanced Model — Top 20 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(output_dir / "graph_feature_importance.png", dpi=150)
plt.close()
print("Saved: graph_feature_importance.png")


# 13. SAVE GRAPH PREDICTIONS


results_graph = pd.DataFrame({
    'actual_time': y_graph_test.values,
    'osrm_time': X_graph_test['osrm_time'].values,
    'baseline_pred': y_pred,
    'graph_pred': y_pred_graph,
    'base_abs_error': np.abs(y_pred - y_test.values),
    'graph_abs_error': np.abs(y_pred_graph - y_graph_test.values),
})
results_graph.to_csv(output_dir / "graph_predictions.csv", index=False)
print("Saved: graph_predictions.csv")




# Calculate errors
error_base  = y_pred       - y_test.values   # baseline errors
error_graph = y_pred_graph - y_graph_test.values  # graph model errors

plt.figure(figsize=(10, 5))

plt.hist(error_base,  bins=100, alpha=0.5, color='steelblue',
         label='Baseline', range=(-500, 500))
plt.hist(error_graph, bins=100, alpha=0.5, color='darkorange',
         label='Graph-Enhanced', range=(-500, 500))

plt.axvline(x=0, color='red', linestyle='--', label='Perfect prediction')
plt.xlabel("Prediction Error (mins)")
plt.ylabel("Number of Trips")
plt.title("Error Distribution — Baseline vs Graph-Enhanced Model")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "error_distribution.png", dpi=150)
plt.close()




plt.figure(figsize=(10, 5))

# Sample 2000 points so plot isn't too crowded
idx = np.random.choice(len(y_test), size=2000, replace=False)

# Baseline
plt.subplot(1, 2, 1)
plt.scatter(y_test.values[idx], y_pred[idx],
            alpha=0.3, color='steelblue', s=10)
plt.plot([0, y_test.max()], [0, y_test.max()],
         color='red', linestyle='--', label='Perfect')
plt.xlabel("Actual Time (mins)")
plt.ylabel("Predicted Time (mins)")
plt.title("Baseline — Actual vs Predicted")
plt.legend()

# Graph model
plt.subplot(1, 2, 2)
plt.scatter(y_graph_test.values[idx], y_pred_graph[idx],
            alpha=0.3, color='darkorange', s=10)
plt.plot([0, y_graph_test.max()], [0, y_graph_test.max()],
         color='red', linestyle='--', label='Perfect')
plt.xlabel("Actual Time (mins)")
plt.ylabel("Predicted Time (mins)")
plt.title("Graph-Enhanced — Actual vs Predicted")
plt.legend()

plt.tight_layout()
plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
plt.close()


print("Saved: actual_vs_predicted.png")

print("\n All done!")
