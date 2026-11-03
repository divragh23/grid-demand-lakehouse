# Databricks notebook source
# MAGIC %md
# MAGIC # Excel export
# MAGIC Builds `grid_report.xlsx` (summary KPIs, weekly planner, gold tables as named Excel tables) for Excel users and Power BI. Saved to the UC volume and Azure Blob.

# COMMAND ----------

# MAGIC %pip install -q xlsxwriter

# COMMAND ----------

import io, os
import datetime as dt
import pandas as pd
from azure.storage.blob import ContainerClient

CATALOG = spark.sql("SELECT current_catalog()").first()[0]
SCHEMA = "grid"
EXPORT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/raw/exports"
FILE_NAME = "grid_report.xlsx"

container = ContainerClient.from_container_url(
    "https://divgridlake01.blob.core.windows.net/lake?" + dbutils.secrets.get("grid", "blob_sas")
)

def load(name):
    return spark.table(f"{CATALOG}.{SCHEMA}.{name}").toPandas()

# COMMAND ----------

daily = load("daily")
daily = daily[daily["is_complete_day"]].drop(columns=["is_complete_day"]).sort_values("date_local")
daily["date_local"] = pd.to_datetime(daily["date_local"])

temp_curve = load("temp_curve").sort_values("temp_bin_f")
hourly_profile = load("hourly_profile").sort_values(["season", "is_weekend", "hour_local"])

evals = load("peak_forecast_eval").sort_values("date_local")
evals["date_local"] = pd.to_datetime(evals["date_local"])
evals["gbm_ape_pct"] = (evals["gbm_error_mwh"].abs() / evals["actual_peak_mwh"]) * 100
evals["iso_ape_pct"] = (evals["iso_error_mwh"].abs() / evals["actual_peak_mwh"]) * 100

weekly = (
    daily.assign(week_start=daily["date_local"] - pd.to_timedelta(daily["date_local"].dt.dayofweek, unit="D"))
    .groupby("week_start")
    .agg(
        days=("date_local", "count"),
        week_peak_mwh=("peak_mwh", "max"),
        avg_daily_peak_mwh=("peak_mwh", "mean"),
        total_mwh=("total_mwh", "sum"),
        max_temp_f=("max_temp_f", "max"),
        min_temp_f=("min_temp_f", "min"),
        cdd=("cdd", "sum"),
        hdd=("hdd", "sum"),
        iso_mape_pct=("mape_pct", "mean"),
    )
    .reset_index()
    .sort_values("week_start", ascending=False)
)

SHEETS = {
    "Weekly": weekly,
    "Daily": daily,
    "TempCurve": temp_curve,
    "HourlyProfile": hourly_profile,
    "ForecastEval": evals,
}
SHEETS = {k: v.round({c: 2 for c in v.select_dtypes("number").columns}) for k, v in SHEETS.items()}

# COMMAND ----------

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as xw:
    wb = xw.book
    bold = wb.add_format({"bold": True})
    title = wb.add_format({"bold": True, "font_size": 16})
    label = wb.add_format({"font_color": "#555555"})
    num = wb.add_format({"num_format": "#,##0", "bold": True, "font_size": 13})
    pct = wb.add_format({"num_format": "0.0\"%\"", "bold": True, "font_size": 13})
    date = wb.add_format({"num_format": "yyyy-mm-dd", "bold": True, "font_size": 13})

    summary = wb.add_worksheet("Summary")
    xw.sheets["Summary"] = summary

    for name, df in SHEETS.items():
        df.to_excel(xw, sheet_name=name, index=False, header=False, startrow=1)
        ws = xw.sheets[name]
        rows, cols = df.shape
        ws.add_table(0, 0, rows, cols - 1, {
            "name": name,
            "columns": [{"header": c} for c in df.columns],
            "style": "Table Style Medium 2",
        })
        ws.set_column(0, cols - 1, 15)
        ws.freeze_panes(1, 1)

    summary.set_column(0, 0, 34)
    summary.set_column(1, 1, 18)
    summary.write(0, 0, "ISO New England — Demand vs. Weather", title)
    summary.write(1, 0, f"Generated {dt.date.today():%Y-%m-%d}", label)

    D, E = SHEETS["Daily"], SHEETS["ForecastEval"]
    test = E[E["split"] == "test"]
    serial = lambda ts: (pd.Timestamp(ts) - pd.Timestamp("1899-12-30")).days
    kpis = [
        ("First day", "=MIN(Daily[date_local])", date, serial(D["date_local"].min())),
        ("Last day", "=MAX(Daily[date_local])", date, serial(D["date_local"].max())),
        ("Days covered", "=ROWS(Daily[date_local])", num, len(D)),
        ("Highest daily peak (MWh)", "=MAX(Daily[peak_mwh])", num, D["peak_mwh"].max()),
        ("Date of highest peak", "=INDEX(Daily[date_local],MATCH(MAX(Daily[peak_mwh]),Daily[peak_mwh],0))", date,
         serial(D.loc[D["peak_mwh"].idxmax(), "date_local"])),
        ("Average daily peak (MWh)", "=AVERAGE(Daily[peak_mwh])", num, D["peak_mwh"].mean()),
        ("Total energy (MWh)", "=SUM(Daily[total_mwh])", num, D["total_mwh"].sum()),
        ("ISO day-ahead MAPE, all days", "=AVERAGE(Daily[mape_pct])", pct, D["mape_pct"].mean()),
        ("Model peak MAPE, test period", '=AVERAGEIFS(ForecastEval[gbm_ape_pct],ForecastEval[split],"test")', pct,
         test["gbm_ape_pct"].mean()),
        ("ISO peak MAPE, test period", '=AVERAGEIFS(ForecastEval[iso_ape_pct],ForecastEval[split],"test")', pct,
         test["iso_ape_pct"].mean()),
    ]
    for i, (k, f, fmt, v) in enumerate(kpis, start=3):
        summary.write(i, 0, k, label)
        summary.write_formula(i, 1, f, fmt, float(v))

    n = len(SHEETS["Daily"])
    line = wb.add_chart({"type": "line"})
    line.add_series({
        "name": "Daily peak (MWh)",
        "categories": ["Daily", 1, 0, n, 0],
        "values": ["Daily", 1, list(SHEETS["Daily"].columns).index("peak_mwh"), n, list(SHEETS["Daily"].columns).index("peak_mwh")],
        "line": {"width": 1.25},
    })
    line.set_title({"name": "Daily peak demand"})
    line.set_legend({"none": True})
    line.set_x_axis({"date_axis": True, "num_format": "mmm yy"})
    line.set_size({"width": 760, "height": 300})
    summary.insert_chart("D3", line)

    m = len(SHEETS["TempCurve"])
    tc_cols = list(SHEETS["TempCurve"].columns)
    scatter = wb.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
    scatter.add_series({
        "name": "Avg demand",
        "categories": ["TempCurve", 1, tc_cols.index("temp_bin_f"), m, tc_cols.index("temp_bin_f")],
        "values": ["TempCurve", 1, tc_cols.index("avg_mwh"), m, tc_cols.index("avg_mwh")],
    })
    scatter.add_series({
        "name": "P90 demand",
        "categories": ["TempCurve", 1, tc_cols.index("temp_bin_f"), m, tc_cols.index("temp_bin_f")],
        "values": ["TempCurve", 1, tc_cols.index("p90_mwh"), m, tc_cols.index("p90_mwh")],
    })
    scatter.set_title({"name": "Hourly demand vs. temperature"})
    scatter.set_x_axis({"name": "Temperature (°F)"})
    scatter.set_y_axis({"name": "MWh"})
    scatter.set_size({"width": 760, "height": 300})
    summary.insert_chart("D19", scatter)

body = buf.getvalue()

# COMMAND ----------

os.makedirs(EXPORT_DIR, exist_ok=True)
with open(f"{EXPORT_DIR}/{FILE_NAME}", "wb") as f:
    f.write(body)

container.upload_blob(f"gold/exports/{FILE_NAME}", body, overwrite=True)
container.upload_blob(f"gold/exports/archive/grid_report_{dt.date.today():%Y-%m-%d}.xlsx", body, overwrite=True)

print(f"{len(body)/1024:,.0f} KB → {EXPORT_DIR}/{FILE_NAME} and blob gold/exports/{FILE_NAME}")
print({k: len(v) for k, v in SHEETS.items()})
