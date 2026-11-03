# Databricks notebook source
# MAGIC %md
# MAGIC # Next-day peak model
# MAGIC Predicts daily peak demand from temperature, calendar and prior-day load, and benchmarks against a naive baseline and ISO-NE's own day-ahead forecast.
# MAGIC
# MAGIC Target-day temperatures are observed values, i.e. a perfect weather forecast. Real-world error would be higher.

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

CATALOG = spark.sql("SELECT current_catalog()").first()[0]
SCHEMA = "grid"
TEST_DAYS = 90

# COMMAND ----------

df = (
    spark.table(f"{CATALOG}.{SCHEMA}.daily")
    .filter("is_complete_day")
    .orderBy("date_local")
    .toPandas()
)
df["date_local"] = pd.to_datetime(df["date_local"])
df = df.set_index("date_local").asfreq("D")

for c in ["is_weekend", "is_holiday"]:
    df[c] = df[c].fillna(False).astype(bool)

df["dow"] = df.index.dayofweek
df["is_offday"] = (df["is_weekend"] | df["is_holiday"]).astype(int)
df["prev_peak"] = df["peak_mwh"].shift(1)
df["prev_avg_temp"] = df["avg_temp_f"].shift(1)
df["cdd_sq"] = df["cdd"] ** 2

GBM_FEATURES = ["max_temp_f", "min_temp_f", "avg_temp_f", "hdd", "cdd", "avg_humidity_pct", "prev_peak", "prev_avg_temp", "is_offday", "dow"]
LIN_FEATURES = ["hdd", "cdd", "cdd_sq", "prev_peak", "is_offday"]

data = df.dropna(subset=list(set(GBM_FEATURES + LIN_FEATURES)) + ["peak_mwh", "forecast_peak_mwh"]).copy()
train, test = data.iloc[:-TEST_DAYS], data.iloc[-TEST_DAYS:]
print(f"train {train.index.min().date()} → {train.index.max().date()} ({len(train)} days)")
print(f"test  {test.index.min().date()} → {test.index.max().date()} ({len(test)} days)")

# COMMAND ----------

lin = LinearRegression().fit(train[LIN_FEATURES], train["peak_mwh"])
gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, random_state=7)
gbm.fit(train[GBM_FEATURES], train["peak_mwh"])

data["pred_naive"] = data["prev_peak"]
data["pred_linear"] = lin.predict(data[LIN_FEATURES])
data["pred_gbm"] = gbm.predict(data[GBM_FEATURES])
data["pred_iso"] = data["forecast_peak_mwh"]
data["split"] = np.where(data.index.isin(test.index), "test", "train")

# COMMAND ----------

def score(frame, col):
    err = frame[col] - frame["peak_mwh"]
    return {
        "mae_mwh": err.abs().mean(),
        "mape_pct": (err.abs() / frame["peak_mwh"]).mean() * 100,
        "bias_mwh": err.mean(),
        "worst_miss_mwh": err.abs().max(),
    }

MODELS = {
    "Naive (yesterday's peak)": "pred_naive",
    "Linear (HDD/CDD + calendar)": "pred_linear",
    "Gradient boosting": "pred_gbm",
    "ISO-NE day-ahead (EIA)": "pred_iso",
}

results = pd.DataFrame({name: score(data[data["split"] == "test"], col) for name, col in MODELS.items()}).T.round(1)
display(results.reset_index(names="model"))

# COMMAND ----------

full = LinearRegression().fit(data[["hdd", "cdd", "is_offday"]], data["peak_mwh"])
coef = dict(zip(["hdd", "cdd", "is_offday"], full.coef_))
print(f"Each cooling degree day adds ~{coef['cdd']:,.0f} MWh to the daily peak")
print(f"Each heating degree day adds ~{coef['hdd']:,.0f} MWh to the daily peak")
print(f"Weekends/holidays shift the peak by ~{coef['is_offday']:,.0f} MWh")

# COMMAND ----------

t = data[data["split"] == "test"]
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t.index, t["peak_mwh"], label="Actual", linewidth=2, color="black")
ax.plot(t.index, t["pred_gbm"], label="Gradient boosting", alpha=0.8)
ax.plot(t.index, t["pred_iso"], label="ISO-NE day-ahead", alpha=0.8, linestyle="--")
ax.set_ylabel("Daily peak (MWh)")
ax.set_title("Next-day peak: test period")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
display(fig)

# COMMAND ----------

out = (
    data.reset_index()[
        ["date_local", "split", "peak_mwh", "pred_naive", "pred_linear", "pred_gbm", "pred_iso", "max_temp_f", "min_temp_f", "is_offday"]
    ]
    .rename(columns={"peak_mwh": "actual_peak_mwh"})
)
out["date_local"] = out["date_local"].dt.date
out["gbm_error_mwh"] = out["pred_gbm"] - out["actual_peak_mwh"]
out["iso_error_mwh"] = out["pred_iso"] - out["actual_peak_mwh"]

(
    spark.createDataFrame(out)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.peak_forecast_eval")
)
