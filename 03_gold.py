# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: reporting tables
# MAGIC Builds `daily`, `temp_curve` and `hourly_profile` from `grid.hourly`. Full rebuild each run.

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CATALOG = spark.sql("SELECT current_catalog()").first()[0]
SCHEMA = "grid"
SRC = f"{CATALOG}.{SCHEMA}.hourly"
BASE_F = 65.0

h = spark.table(SRC)

def publish(df, name):
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.{name}")

# COMMAND ----------

hol = USFederalHolidayCalendar().holidays(start="2020-01-01", end="2030-12-31")
holidays = spark.createDataFrame(
    pd.DataFrame({"date_local": [d.date() for d in hol], "is_holiday": True})
)

season = (
    F.when(F.month("date_local").isin(12, 1, 2), "Winter")
    .when(F.month("date_local").isin(3, 4, 5), "Spring")
    .when(F.month("date_local").isin(6, 7, 8), "Summer")
    .otherwise("Fall")
)

ape = F.abs(F.col("forecast_mwh") - F.col("demand_mwh")) / F.col("demand_mwh")

daily = (
    h.groupBy("date_local")
    .agg(
        F.count("demand_mwh").alias("demand_hours"),
        F.max("demand_mwh").alias("peak_mwh"),
        F.max_by("hour_local", "demand_mwh").alias("peak_hour"),
        F.min("demand_mwh").alias("min_mwh"),
        F.avg("demand_mwh").alias("avg_mwh"),
        F.sum("demand_mwh").alias("total_mwh"),
        F.max("forecast_mwh").alias("forecast_peak_mwh"),
        (F.avg(ape) * 100).alias("mape_pct"),
        F.avg(F.col("forecast_mwh") - F.col("demand_mwh")).alias("bias_mwh"),
        F.avg("temp_f").alias("avg_temp_f"),
        F.max("temp_f").alias("max_temp_f"),
        F.min("temp_f").alias("min_temp_f"),
        F.avg("humidity_pct").alias("avg_humidity_pct"),
    )
    .withColumn("hdd", F.greatest(F.lit(0.0), F.lit(BASE_F) - F.col("avg_temp_f")))
    .withColumn("cdd", F.greatest(F.lit(0.0), F.col("avg_temp_f") - F.lit(BASE_F)))
    .withColumn("peak_error_mwh", F.col("forecast_peak_mwh") - F.col("peak_mwh"))
    .withColumn("day_of_week", F.date_format("date_local", "EEE"))
    .withColumn("is_weekend", F.dayofweek("date_local").isin(1, 7))
    .withColumn("month", F.month("date_local"))
    .withColumn("season", season)
    .join(F.broadcast(holidays), "date_local", "left")
    .fillna({"is_holiday": False})
    .withColumn("is_complete_day", F.col("demand_hours") >= 23)
    .orderBy("date_local")
)

publish(daily, "daily")

# COMMAND ----------

complete = h.filter(F.col("demand_mwh").isNotNull() & F.col("temp_f").isNotNull())

temp_curve = (
    complete.withColumn("temp_bin_f", (F.floor(F.col("temp_f") / 2) * 2).cast("int"))
    .groupBy("temp_bin_f")
    .agg(
        F.count("*").alias("hours"),
        F.avg("demand_mwh").alias("avg_mwh"),
        F.percentile_approx("demand_mwh", 0.9).alias("p90_mwh"),
        F.max("demand_mwh").alias("max_mwh"),
    )
    .filter("hours >= 5")
    .orderBy("temp_bin_f")
)

publish(temp_curve, "temp_curve")

# COMMAND ----------

hourly_profile = (
    complete.withColumn("season", season)
    .withColumn("is_weekend", F.dayofweek("date_local").isin(1, 7))
    .groupBy("season", "is_weekend", "hour_local")
    .agg(
        F.avg("demand_mwh").alias("avg_mwh"),
        F.avg("temp_f").alias("avg_temp_f"),
        F.count("*").alias("hours"),
    )
    .orderBy("season", "is_weekend", "hour_local")
)

publish(hourly_profile, "hourly_profile")

# COMMAND ----------

d = spark.table(f"{CATALOG}.{SCHEMA}.daily").filter("is_complete_day")

display(
    d.agg(
        F.count("*").alias("complete_days"),
        F.round(F.max("peak_mwh")).alias("max_peak_mwh"),
        F.round(F.avg("mape_pct"), 2).alias("avg_forecast_mape_pct"),
        F.round(F.corr("cdd", "peak_mwh"), 3).alias("corr_cdd_peak"),
        F.round(F.corr("hdd", "peak_mwh"), 3).alias("corr_hdd_peak"),
    )
)

display(
    d.orderBy(F.desc("peak_mwh"))
    .select("date_local", "day_of_week", "peak_mwh", "peak_hour", "max_temp_f", "min_temp_f", "forecast_peak_mwh", "peak_error_mwh")
    .limit(10)
)
