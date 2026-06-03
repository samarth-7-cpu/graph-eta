import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

data_path = project_root / "data" / "raw" / "delivery_data.csv"

df = pd.read_csv(data_path)
df = pd.get_dummies(df, columns=['route_type'], prefix='route', dtype=int)
df['od_start_time'] = pd.to_datetime(df['od_start_time'])
df['od_end_time']   = pd.to_datetime(df['od_end_time'])

df['hour']        = df['od_start_time'].dt.hour
df['day_of_week'] = df['od_start_time'].dt.dayofweek
df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)

# Time of day bucket
def time_bucket(h):
    if 5 <= h < 12:  return 0   # morning
    elif 12 <= h < 18: return 1  # afternoon
    elif 18 <= h < 22: return 2  # evening
    else: return 3               # night

df['time_of_day'] = df['hour'].apply(time_bucket)

# Delay ratio (used later for graph, good to have now)
df['delay_ratio'] = df['actual_time'] / df['osrm_time']





le_src  = LabelEncoder()
le_dest = LabelEncoder()

df['source_enc']      = le_src.fit_transform(df['source_center'])
df['destination_enc'] = le_dest.fit_transform(df['destination_center'])



feature_cols = [
    'osrm_time',                    # OSRM prediction
    'osrm_distance',                # route distance
    'segment_osrm_time',            # segment-level OSRM
    'segment_osrm_distance',        # segment-level distance
    'actual_distance_to_destination',
    'start_scan_to_end_scan',       # dwell time proxy
    'segment_factor',               # segment delay factor
    'factor',                       # overall factor
    'route_Carting',                # route type flags
    'route_FTL',
    'is_cutoff',                    # cutoff flag
    'cutoff_factor',
    'hour',
    'day_of_week',
    'is_weekend',
    'time_of_day',
    'source_enc',
    'destination_enc',
]

TARGET = 'actual_time'

X = df[feature_cols]
y = df[TARGET]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")



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


mae = mean_absolute_error(y_test, y_pred)

within_15 = np.mean(
    np.abs(y_pred - y_test) / y_test < 0.15
) * 100

print("=" * 40)
print("BASELINE MODEL RESULTS")
print("=" * 40)
print(f"MAE              : {mae:.2f} mins")
print(f"% within 15%     : {within_15:.2f}%")
print(f"OSRM Baseline MAE: {mean_absolute_error(y_test, X_test['osrm_time']):.2f} mins")
print("=" * 40)



importances = pd.Series(baseline.feature_importances_, index=feature_cols)
importances.sort_values().plot(kind='barh', figsize=(8, 6), color='steelblue')

output_dir = project_root / "outputs"
output_dir.mkdir(exist_ok=True)

plt.title("Baseline Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(
    output_dir / "baseline_feature_importance.png",
    dpi=150
)
plt.show()



results_df = X_test.copy()
results_df['actual_time']   = y_test.values
results_df['baseline_pred'] = y_pred
results_df['abs_error']     = np.abs(y_pred - y_test.values)
results_df.to_csv(
    output_dir / "baseline_predictions.csv",
    index=False
)

print("Saved: baseline_predictions.csv")