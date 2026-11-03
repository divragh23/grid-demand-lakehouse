# Grid Demand vs. Weather Lakehouse

How much does weather drive electricity demand in New England, and how well can tomorrow's peak be predicted? I built this project to find out.

It's a daily ETL pipeline. Hourly ISO New England demand (EIA) and regional weather (Open-Meteo) land in Azure Blob Storage, flow through bronze/silver/gold Delta tables on Databricks, feed a next-day peak model in pandas and scikit-learn, and end up in an Excel report and a Power BI dashboard. The whole thing runs on its own every morning.

## Architecture

```
EIA API ─────┐
             ├─► 01 bronze ──► Azure Blob (raw JSON, system of record)
Open-Meteo ──┘                     │
                                   ▼
                           UC volume mirror
                                   │
                    02 silver ─► grid.hourly (Delta, MERGE upsert)
                                   │
                    03 gold ───► grid.daily · grid.temp_curve · grid.hourly_profile
                                   │
                    04 model ──► grid.peak_forecast_eval
                                   │
                    05 export ─► grid_report.xlsx (Blob) ──► Power BI
```

A Databricks Job runs the five notebooks in order at 7:00 AM ET, with an email alert if any step fails. Power BI refreshes from the exported workbook at 8:00 AM.

| Layer | What happens |
|---|---|
| **Bronze** | One raw JSON file per source per UTC day, written to Azure Blob and mirrored to a Unity Catalog volume. EIA revises recent hours, so each run re-pulls the last 7 days and overwrites those files. Reruns are idempotent. |
| **Silver** | Parses EIA actual demand (`D`) and the day-ahead forecast (`DF`). Weather from six metros (Boston, Worcester, Providence, Hartford, Manchester, Portland) is population-weighted into one regional series. Handles the UTC → Eastern conversion, including DST. Upserts into Delta on `ts_utc` with `MERGE`, then checks for duplicate keys, missing hours, nulls and partial weather. |
| **Gold** | Daily peak and peak hour, total energy, heating and cooling degree days (base 65°F), ISO forecast error, holiday and weekend flags. Also builds a demand-by-temperature curve and a load shape by hour and season. |
| **Model** | Next-day peak forecast from a naive baseline, a linear regression on HDD/CDD, and gradient boosting, all benchmarked against ISO-NE's own day-ahead forecast on a 90-day holdout. |
| **Export** | Excel workbook with named tables, KPI formulas and charts, written to Blob for Power BI. |

## Findings (Nov 2025 – Nov 2026)

- **8,808 hourly rows**, with 0 duplicate keys and 0 missing hours.
- The highest daily peak was **25,289 MWh** on July 2, 2026. All top-10 peak days fall in June–August.
- Demand vs. temperature is U-shaped and lowest at 50–60°F. Heat moves the peak far more than cold: correlation with the daily peak is **0.65** for cooling degree days vs **0.14** for heating degree days. That fits a region that heats mostly with gas and oil rather than electricity.

Next-day peak, 90-day test period:

| Model | MAPE |
|---|---|
| Naive (yesterday's peak) | 5.5% |
| Linear (HDD/CDD + calendar) | 4.5% |
| Gradient boosting | **3.8%** |
| ISO-NE day-ahead forecast | 1.2% |

Gradient boosting cuts error by about 30% compared with the naive baseline. ISO-NE's operational forecast is still about 3× more accurate, which is what I expected given the data and tooling they have.

**Limitations**
- The model gets the target day's *observed* temperatures, which acts like a perfect weather forecast. Real-world error would be higher.
- The biggest misses come on days when the weather turns. The model leans on the previous day's peak, so it lags a day when conditions shift suddenly. Day-over-day temperature change is the next feature I'd add.

## Design decisions

- **Volume mirror:** Databricks Free Edition runs serverless on AWS and can't read `abfss://` paths with Azure credentials. Raw data lands in Blob through the Azure SDK and is mirrored into a Unity Catalog volume for Spark. A resync cell rebuilds the volume from Blob if it ever drifts.
- **Secrets:** The EIA API key and a container-scoped SAS token live in a Databricks secret scope, never in code. Power BI reads Blob with an organizational account through a read-only Azure role, so it stores no keys.
- **Retries:** API calls back off exponentially on timeouts and on 429/5xx responses. I added this after a slow Open-Meteo archive request killed the first full-year backfill.

## Repo layout

```
notebooks/
  01_bronze_ingest.py   extract → Blob + volume, ingest log
  02_silver_hourly.py   parse, DST, weighted weather, MERGE, quality checks
  03_gold.py            daily / temp curve / hourly profile tables
  04_peak_model.py      pandas + scikit-learn peak models, evaluation
  05_excel_export.py    Excel report for Power BI
```

The notebooks are in Databricks source format. Import them via Workspace → Import.

## Running it

1. Azure: an ADLS Gen2 storage account with a `lake` container.
2. Databricks: a secret scope `grid` with keys `eia_key` (free from eia.gov/opendata) and `blob_sas`.
3. Import the notebooks and run `01` with `start_date` set for the initial backfill. Then create a job that runs 01 → 05 daily.

## Stack

Python · PySpark · Delta Lake · Databricks (Unity Catalog, Jobs) · Azure Blob Storage / ADLS Gen2 · pandas · scikit-learn · Excel · Power BI
