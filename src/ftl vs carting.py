import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pathlib import Path



project_root = Path(__file__).resolve().parents[1]
data_path    = project_root / "data" / "raw" / "delivery_data.csv"
hub_path     = project_root / "data" / "processed" / "graph_features.csv"
output_dir   = project_root / "outputs"
output_dir.mkdir(exist_ok=True)



print("Loading data...")
df_ftl = pd.read_csv(data_path)
print(f"Shape before drop: {df_ftl.shape}")

df_ftl = df_ftl[df_ftl['route_type'].notna()].copy()
print(f"Shape after dropping missing route_type: {df_ftl.shape}")




df_ftl['od_start_time'] = pd.to_datetime(df_ftl['od_start_time'], errors='coerce')
df_ftl['hour']          = df_ftl['od_start_time'].dt.hour
df_ftl['day_of_week']   = df_ftl['od_start_time'].dt.dayofweek
df_ftl['is_weekend']    = df_ftl['day_of_week'].isin([5, 6]).astype(int)

def time_bucket(h):
    if 5  <= h < 12:   return 0
    elif 12 <= h < 18: return 1
    elif 18 <= h < 22: return 2
    else:              return 3

df_ftl['time_of_day'] = df_ftl['hour'].apply(time_bucket)
df_ftl['delay_ratio'] = df_ftl['actual_time'] / df_ftl['osrm_time']
df_ftl['is_cutoff']   = df_ftl['is_cutoff'].astype(int)

# 3. BUILD TARGET COLUMN


print(f"\nroute_type values: {df_ftl['route_type'].unique()}")

df_ftl['is_FTL'] = (df_ftl['route_type'] == 'FTL').astype(int)

print(f"FTL trips    : {df_ftl['is_FTL'].sum()} ({df_ftl['is_FTL'].mean()*100:.1f}%)")
print(f"Carting trips: {(df_ftl['is_FTL']==0).sum()} ({(df_ftl['is_FTL']==0).mean()*100:.1f}%)")


# 4. MAP HUB GRAPH FEATURES


print("\nMapping hub graph features...")
hub_df    = pd.read_csv(hub_path)
hub_index = hub_df.set_index('hub')

df_ftl['src_degree_centrality'] = df_ftl['source_center'].map(hub_index['degree_centrality']).fillna(0)
df_ftl['src_betweenness']       = df_ftl['source_center'].map(hub_index['betweenness']).fillna(0)
df_ftl['src_pagerank']          = df_ftl['source_center'].map(hub_index['pagerank']).fillna(0)
df_ftl['src_clustering']        = df_ftl['source_center'].map(hub_index['clustering']).fillna(0)

df_ftl['dst_degree_centrality'] = df_ftl['destination_center'].map(hub_index['degree_centrality']).fillna(0)
df_ftl['dst_betweenness']       = df_ftl['destination_center'].map(hub_index['betweenness']).fillna(0)
df_ftl['dst_pagerank']          = df_ftl['destination_center'].map(hub_index['pagerank']).fillna(0)
df_ftl['dst_clustering']        = df_ftl['destination_center'].map(hub_index['clustering']).fillna(0)

print("Graph features mapped!")


# 5. FEATURE LIST


ftl_features = [
    'osrm_distance',
    'osrm_time',
    'actual_distance_to_destination',
    'segment_osrm_distance',
    'start_scan_to_end_scan',
    'factor',
    'segment_factor',
    'cutoff_factor',
    'is_cutoff',
    'hour',
    'day_of_week',
    'is_weekend',
    'time_of_day',
    'delay_ratio',
    'src_degree_centrality',
    'src_betweenness',
    'src_pagerank',
    'src_clustering',
    'dst_degree_centrality',
    'dst_betweenness',
    'dst_pagerank',
    'dst_clustering',
]

TARGET = 'is_FTL'

# Safety check
missing = [c for c in ftl_features if c not in df_ftl.columns]
if missing:
    print(f"  Missing columns: {missing}")
    ftl_features = [c for c in ftl_features if c in df_ftl.columns]


# 6. TRAIN / TEST SPLIT (using data column)


train_df = df_ftl[df_ftl['data'] == 'training']
test_df  = df_ftl[df_ftl['data'] == 'test']

X_train = train_df[ftl_features]
y_train = train_df[TARGET]
X_test  = test_df[ftl_features]
y_test  = test_df[TARGET]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")


# 7. TRAIN FTL vs CARTING CLASSIFIER


print("\nTraining FTL vs Carting Classifier...")

ftl_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
)

ftl_model.fit(X_train, y_train)
y_pred       = ftl_model.predict(X_test)
y_pred_proba = ftl_model.predict_proba(X_test)[:, 1]

print("Classifier trained!")

# 8. EVALUATE


acc    = accuracy_score(y_test, y_pred)
auc    = roc_auc_score(y_test, y_pred_proba)
report = classification_report(y_test, y_pred, target_names=['Carting', 'FTL'])

print("\n" + "=" * 45)
print("FTL vs CARTING CLASSIFIER RESULTS")
print("=" * 45)
print(f"Accuracy : {acc*100:.2f}%")
print(f"ROC-AUC  : {auc:.4f}")
print("=" * 45)
print("\nClassification Report:")
print(report)


# 9. CONFUSION MATRIX PLOT


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Carting', 'FTL'],
    yticklabels=['Carting', 'FTL']
)
plt.title("FTL vs Carting — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(output_dir / "ftl_confusion_matrix.png", dpi=150)
plt.close()
print("Saved: ftl_confusion_matrix.png")


imp = pd.Series(ftl_model.feature_importances_, index=ftl_features)
imp.sort_values().tail(15).plot(kind='barh', figsize=(9, 6), color='green')
plt.title("FTL vs Carting — Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(output_dir / "ftl_feature_importance.png", dpi=150)
plt.close()
print("Saved: ftl_feature_importance.png")

# ─────────────────────────────────────────
# 11. TIME-COST TRADE-OFF BY DISTANCE
# ─────────────────────────────────────────

df_ftl['distance_bucket'] = pd.cut(
    df_ftl['osrm_distance'],
    bins=[0, 100, 300, 600, 1000, 99999],
    labels=['<100km', '100-300km', '300-600km', '600-1000km', '>1000km']
)

tradeoff = df_ftl.groupby(['distance_bucket', 'route_type']).agg(
    avg_actual_time = ('actual_time', 'mean'),
    avg_delay_ratio = ('delay_ratio', 'mean'),
    trip_count      = ('actual_time', 'count')
).reset_index()

print("\n" + "=" * 60)
print("TIME-COST TRADE-OFF: FTL vs Carting by Distance Bucket")
print("=" * 60)
print(tradeoff.to_string(index=False))
print("=" * 60)

tradeoff.to_csv(output_dir / "ftl_carting_tradeoff.csv", index=False)
print("Saved: ftl_carting_tradeoff.csv")



results = pd.DataFrame({
    'source_center':      test_df['source_center'].values,
    'destination_center': test_df['destination_center'].values,
    'actual_route':       y_test.values,
    'predicted_route':    y_pred,
    'ftl_probability':    y_pred_proba,
    'osrm_distance':      X_test['osrm_distance'].values,
    'delay_ratio':        X_test['delay_ratio'].values,
})
results['recommended'] = results['ftl_probability'].apply(
    lambda p: 'FTL' if p >= 0.5 else 'Carting'
)
results.to_csv(output_dir / "ftl_predictions.csv", index=False)
print("Saved: ftl_predictions.csv")

print("\n FTL vs Carting classifier done!")