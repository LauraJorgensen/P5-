import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Konfiguration ---
csv_path = 'pv_production_june1.csv'
target_col = 'pv_production'

period_15min = 96              # antal 15-min punkter pr. dag
test_hours = 36                # 1.5 dag = 36 timer
test_15min = test_hours * 4    # 36 * 4 = 144 punkter

# --- Data indlæsning ---
df = pd.read_csv(csv_path, parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
y = df[target_col].astype(float)

# --- Train/Test split (15-min data) ---
y_train = y.iloc[:-test_15min]
y_test = y.iloc[-test_15min:]          # 1.5 dag test

# Den sidste dag (til plotting)
last_day_test = y_test.iloc[-period_15min:]

# --- Resample til hourly ---
y_hour = y.resample('H').mean()

# --- Hourly train/test split ---
test_hours = int(test_15min / 4)       # antal timer i test
y_train_hour = y_hour.iloc[:-test_hours]
y_test_hour = y_hour.iloc[-test_hours:]

# -------------------------------
# --- RANDOM WALK – PER HOUR ---
# --- + 95% CONFIDENCE INTERVALS
# -------------------------------
preds_hour = []
preds_hour_lower = []
preds_hour_upper = []
idxs = []

for t_idx in y_test_hour.index:
    h = t_idx.hour

    # træningspunkter for denne hour-of-day
    idx_train_h = y_train_hour.index[y_train_hour.index.hour == h]
    y_train_h = y_train_hour.loc[idx_train_h].dropna()

    if len(y_train_h) < 2:
        preds_hour.append(np.nan)
        preds_hour_lower.append(np.nan)
        preds_hour_upper.append(np.nan)
        idxs.append(t_idx)
        continue

    # sidste træningsværdi og tid
    last_val = y_train_h.iloc[-1]
    last_time = y_train_h.index[-1]

    # sigma = std af differencer (RW innovation variance)
    diffs = y_train_h.diff().dropna()
    sigma = diffs.std()
    if pd.isna(sigma) or sigma == 0:
        sigma = 1e-6  # fallback

    # beregn hvor mange skridt (dage) frem vi prognoserer for denne hour
    if pd.isna(last_time) or t_idx <= last_time:
        k = 1
    else:
        # for samme hour-of-day er forekomster dagligt => skridt = antal dage
        days = (t_idx - last_time) / pd.Timedelta(days=1)
        k = max(1, int(round(days)))

    # 95% CI for k-step ahead: last_val ± 1.96 * sigma * sqrt(k)
    margin = 1.96 * sigma * np.sqrt(k)
    ci_lower = last_val - margin
    ci_upper = last_val + margin

    preds_hour.append(float(last_val))
    preds_hour_lower.append(float(ci_lower))
    preds_hour_upper.append(float(ci_upper))
    idxs.append(t_idx)

preds_hour_series = pd.Series(preds_hour, index=idxs)
preds_hour_lower_series = pd.Series(preds_hour_lower, index=idxs)
preds_hour_upper_series = pd.Series(preds_hour_upper, index=idxs)

# --- Map hourly forecast + CI til 15-min ---
preds_15min = preds_hour_series.reindex(y_test.index, method='ffill')
preds_15min_lower = preds_hour_lower_series.reindex(y_test.index, method='ffill')
preds_15min_upper = preds_hour_upper_series.reindex(y_test.index, method='ffill')

# --- Beregn fejl over HELE 1.5 dag test-set ---
rmse = math.sqrt(mean_squared_error(y_test, preds_15min))
mae = mean_absolute_error(y_test, preds_15min)

print(f"\nRMSE 15-min test-set (1.5 dag): {rmse:.3f}")
print(f"MAE 15-min test-set (1.5 dag): {mae:.3f}")

# --- Plot KUN sidste dag ---
plt.figure(figsize=(12, 5))
plt.plot(last_day_test.index, last_day_test, label='Test set')
plt.plot(last_day_test.index, preds_15min.loc[last_day_test.index], label='Prediction')

# plot CI som shaded area
plt.fill_between(
    last_day_test.index,
    preds_15min_lower.loc[last_day_test.index],
    preds_15min_upper.loc[last_day_test.index],
    alpha=0.25,
    label='95% CI'
)

plt.legend()
plt.tight_layout()
plt.show()
