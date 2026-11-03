# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: hourly demand + weather
# MAGIC Parses bronze JSON, pivots EIA demand/forecast, population-weights six metros into one regional weather series, and upserts into `grid.hourly` via MERGE.

# COMMAND ----------

from pyspark.sql import functions as F, Window

spark.conf.set("spark.sql.session.timeZone", "UTC")

CATALOG = spark.sql("SELECT current_catalog()").first()[0]
SCHEMA = "grid"
VOL_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
TARGET = f"{CATALOG}.{SCHEMA}.hourly"
TZ = "America/New_York"
N_LOCATIONS = 6

# Approximate metro-area populations (millions), used as load weights.
METRO_POP = {
    "boston": 4.9,
    "providence": 1.68,
    "hartford": 1.2,
    "worcester": 0.98,
    "portland_me": 0.56,
    "manchester_nh": 0.42,
}

# COMMAND ----------

latest = lambda *keys: Window.partitionBy(*keys).orderBy(F.desc("fetched_at_utc"))

eia = (
    spark.read.option("multiLine", True).json(f"{VOL_ROOT}/bronze/eia/region_data/")
    .select("fetched_at_utc", F.explode("data").alias("r"))
    .select(
        F.to_timestamp("r.period", "yyyy-MM-dd'T'HH").alias("ts_utc"),
        F.col("r.type").alias("type"),
        F.col("r.value").cast("double").alias("value"),
        "fetched_at_utc",
    )
    .withColumn("rn", F.row_number().over(latest("ts_utc", "type")))
    .filter("rn = 1")
    .groupBy("ts_utc")
    .pivot("type", ["D", "DF"])
    .agg(F.first("value"))
    .withColumnRenamed("D", "demand_mwh")
    .withColumnRenamed("DF", "forecast_mwh")
)

# COMMAND ----------

wx_points = (
    spark.read.option("multiLine", True).json(f"{VOL_ROOT}/bronze/weather/open_meteo/")
    .select("fetched_at_utc", F.explode("data").alias("d"))
    .select(
        "fetched_at_utc",
        F.col("d.location").alias("location"),
        F.col("d.hourly.time").alias("t"),
        F.col("d.hourly.temperature_2m").alias("temp"),
        F.col("d.hourly.apparent_temperature").alias("app"),
        F.col("d.hourly.relative_humidity_2m").alias("rh"),
        F.col("d.hourly.dew_point_2m").alias("dew"),
    )
    .select("fetched_at_utc", "location", F.explode(F.arrays_zip("t", "temp", "app", "rh", "dew")).alias("h"))
    .select(
        "fetched_at_utc",
        "location",
        F.to_timestamp("h.t", "yyyy-MM-dd'T'HH:mm").alias("ts_utc"),
        F.col("h.temp").cast("double").alias("temp_f"),
        F.col("h.app").cast("double").alias("apparent_temp_f"),
        F.col("h.rh").cast("double").alias("humidity_pct"),
        F.col("h.dew").cast("double").alias("dew_point_f"),
    )
    .withColumn("rn", F.row_number().over(latest("ts_utc", "location")))
    .filter("rn = 1")
)

pop = spark.createDataFrame(list(METRO_POP.items()), "location STRING, pop DOUBLE")

def weighted(col):
    return F.sum(F.col(col) * F.col("pop")) / F.sum(F.when(F.col(col).isNotNull(), F.col("pop")))

wx = (
    wx_points.join(F.broadcast(pop), "location")
    .groupBy("ts_utc")
    .agg(
        weighted("temp_f").alias("temp_f"),
        weighted("apparent_temp_f").alias("apparent_temp_f"),
        weighted("humidity_pct").alias("humidity_pct"),
        weighted("dew_point_f").alias("dew_point_f"),
        F.count("temp_f").cast("int").alias("wx_locations"),
    )
)

# COMMAND ----------

staged = (
    eia.join(wx, "ts_utc", "full")
    .withColumn("ts_local", F.from_utc_timestamp("ts_utc", TZ))
    .withColumn("date_local", F.to_date("ts_local"))
    .withColumn("hour_local", F.hour("ts_local"))
    .withColumn("updated_at", F.current_timestamp())
    .select(
        "ts_utc", "ts_local", "date_local", "hour_local",
        "demand_mwh", "forecast_mwh",
        "temp_f", "apparent_temp_f", "humidity_pct", "dew_point_f", "wx_locations",
        "updated_at",
    )
)

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TARGET} (
        ts_utc TIMESTAMP NOT NULL,
        ts_local TIMESTAMP,
        date_local DATE,
        hour_local INT,
        demand_mwh DOUBLE,
        forecast_mwh DOUBLE,
        temp_f DOUBLE,
        apparent_temp_f DOUBLE,
        humidity_pct DOUBLE,
        dew_point_f DOUBLE,
        wx_locations INT,
        updated_at TIMESTAMP
    )
    CLUSTER BY (date_local)
""")

staged.createOrReplaceTempView("staged_hourly")

spark.sql(f"""
    MERGE INTO {TARGET} t
    USING staged_hourly s
    ON t.ts_utc = s.ts_utc
    WHEN MATCHED AND NOT (
        t.demand_mwh <=> s.demand_mwh AND
        t.forecast_mwh <=> s.forecast_mwh AND
        t.temp_f <=> s.temp_f AND
        t.apparent_temp_f <=> s.apparent_temp_f AND
        t.humidity_pct <=> s.humidity_pct AND
        t.dew_point_f <=> s.dew_point_f AND
        t.wx_locations <=> s.wx_locations
    ) THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

display(
    spark.sql(f"DESCRIBE HISTORY {TARGET} LIMIT 1")
    .select("version", "timestamp", "operation", "operationMetrics")
)

# COMMAND ----------

t = spark.table(TARGET)

checks = t.agg(
    F.count("*").alias("rows"),
    (F.count("*") - F.countDistinct("ts_utc")).alias("dup_keys"),
    ((F.unix_timestamp(F.max("ts_utc")) - F.unix_timestamp(F.min("ts_utc"))) / 3600 + 1 - F.count("*"))
        .cast("int").alias("missing_hours"),
    F.sum(F.col("demand_mwh").isNull().cast("int")).alias("null_demand"),
    F.sum(F.col("forecast_mwh").isNull().cast("int")).alias("null_forecast"),
    F.sum(F.col("temp_f").isNull().cast("int")).alias("null_temp"),
    F.sum((F.col("demand_mwh") <= 0).cast("int")).alias("nonpositive_demand"),
    F.sum((F.col("wx_locations") < N_LOCATIONS).cast("int")).alias("partial_weather"),
    F.min("ts_utc").alias("first_hour_utc"),
    F.max("ts_utc").alias("last_hour_utc"),
)
display(checks)

r = checks.first()
assert r["dup_keys"] == 0, f"{r['dup_keys']} duplicate ts_utc keys in {TARGET}"
