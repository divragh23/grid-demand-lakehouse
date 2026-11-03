# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze ingestion
# MAGIC ISO-NE hourly demand (EIA) and New England weather (Open-Meteo), landed to Azure Blob and mirrored to a UC volume.

# COMMAND ----------

import json, os, time, uuid
import datetime as dt
import requests
from azure.storage.blob import ContainerClient

dbutils.widgets.text("start_date", "")
dbutils.widgets.text("end_date", "")

# COMMAND ----------

CATALOG = spark.sql("SELECT current_catalog()").first()[0]
SCHEMA = "grid"
VOL_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/raw"

EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
WX_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WX_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WX_VARS = "temperature_2m,apparent_temperature,relative_humidity_2m,dew_point_2m"

LOCATIONS = [
    ("boston", 42.3601, -71.0589),
    ("worcester", 42.2626, -71.8023),
    ("providence", 41.8240, -71.4128),
    ("hartford", 41.7658, -72.6734),
    ("manchester_nh", 42.9956, -71.4548),
    ("portland_me", 43.6591, -70.2568),
]

EIA_KEY = dbutils.secrets.get("grid", "eia_key")
container = ContainerClient.from_container_url(
    "https://divgridlake01.blob.core.windows.net/lake?" + dbutils.secrets.get("grid", "blob_sas")
)

RUN_ID = str(uuid.uuid4())
TODAY_UTC = dt.datetime.now(dt.timezone.utc).date()

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.ingest_log (
        run_id STRING, source STRING, day DATE, row_count INT, rel_path STRING, fetched_at TIMESTAMP
    )
""")

# COMMAND ----------

def resolve_window():
    end_s = dbutils.widgets.get("end_date").strip()
    start_s = dbutils.widgets.get("start_date").strip()
    end = dt.date.fromisoformat(end_s) if end_s else TODAY_UTC - dt.timedelta(days=1)
    start = dt.date.fromisoformat(start_s) if start_s else end - dt.timedelta(days=6)
    assert start <= end, f"start_date {start} > end_date {end}"
    return start, end


def get_json(url, params, tries=5):
    for attempt in range(tries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    r.raise_for_status()


def land(rel_path, payload):
    body = json.dumps(payload, separators=(",", ":")).encode()
    container.upload_blob(rel_path, body, overwrite=True)
    path = f"{VOL_ROOT}/{rel_path}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(body)


def envelope(source, day, params, data):
    return {
        "source": source,
        "day_utc": day.isoformat(),
        "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "request_params": params,
        "data": data,
    }

# COMMAND ----------

def fetch_eia_day(day):
    params = {
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "ISNE",
        "facets[type][]": ["D", "DF"],
        "start": f"{day}T00",
        "end": f"{day}T23",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    rows, offset = [], 0
    while True:
        resp = get_json(EIA_URL, {**params, "api_key": EIA_KEY, "offset": offset})["response"]
        batch = resp.get("data", [])
        rows.extend(batch)
        offset += len(batch)
        if not batch or offset >= int(resp.get("total", 0)):
            break
    return envelope("eia_region_data", day, params, rows), len(rows)


def fetch_weather_day(day):
    url = WX_FORECAST_URL if (TODAY_UTC - day).days <= 80 else WX_ARCHIVE_URL
    params = {
        "latitude": ",".join(str(lat) for _, lat, _ in LOCATIONS),
        "longitude": ",".join(str(lon) for _, _, lon in LOCATIONS),
        "hourly": WX_VARS,
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
        "timezone": "UTC",
        "temperature_unit": "fahrenheit",
    }
    payload = get_json(url, params)
    payload = payload if isinstance(payload, list) else [payload]
    data = [{"location": name, **p} for (name, _, _), p in zip(LOCATIONS, payload)]
    n = sum(len(d.get("hourly", {}).get("time", [])) for d in data)
    return envelope("open_meteo_hourly", day, {**params, "endpoint": url}, data), n

# COMMAND ----------

SOURCES = [
    ("eia_region_data", fetch_eia_day, "eia/region_data"),
    ("open_meteo_hourly", fetch_weather_day, "weather/open_meteo"),
]

start, end = resolve_window()
log_rows = []
day = start
while day <= end:
    for source, fetch, folder in SOURCES:
        payload, n = fetch(day)
        rel_path = f"bronze/{folder}/date={day}/{source}_{day}.json"
        land(rel_path, payload)
        log_rows.append((RUN_ID, source, day, n, rel_path, dt.datetime.now(dt.timezone.utc)))
        time.sleep(0.2)
    day += dt.timedelta(days=1)

# COMMAND ----------

log_df = spark.createDataFrame(
    log_rows, "run_id STRING, source STRING, day DATE, row_count INT, rel_path STRING, fetched_at TIMESTAMP"
)
log_df.write.mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.ingest_log")

from pyspark.sql import functions as F

display(
    log_df.groupBy("source").agg(
        F.count("day").alias("days"),
        F.sum("row_count").alias("total_rows"),
        F.min("row_count").alias("min_rows_per_day"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resync volume from Blob

# COMMAND ----------

RUN_RESYNC = False

if RUN_RESYNC:
    for blob in container.list_blobs(name_starts_with="bronze/"):
        if blob.name.endswith(".json"):
            path = f"{VOL_ROOT}/{blob.name}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(container.download_blob(blob.name).readall())
