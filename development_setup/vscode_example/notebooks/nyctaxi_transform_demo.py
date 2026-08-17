# Databricks notebook source
# MAGIC %md
# MAGIC # NYC Taxi Transformation — VS Code Extension Demo
# MAGIC
# MAGIC This notebook is designed to run via the **Databricks VS Code Extension**
# MAGIC ("Run File on Databricks"). The whole file is uploaded as a workspace file
# MAGIC and executes **entirely on the cluster** — your laptop is just an editor.
# MAGIC
# MAGIC Contrast this with `src/ml_test/`, where Python runs locally via
# MAGIC Databricks Connect and only Spark calls are remote.
# MAGIC
# MAGIC **How to run from VS Code:**
# MAGIC 1. Open this file in VS Code with the Databricks Extension installed
# MAGIC 2. Make sure the extension's cluster picker shows an attached cluster
# MAGIC 3. Click the ▶ icon in the editor title bar, or right-click → "Run File on Databricks"
# MAGIC 4. Output streams back to the VS Code terminal / Databricks output pane

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Where is this code actually running?
# MAGIC
# MAGIC When run via the extension, all of this — including the Python interpreter — lives
# MAGIC on the cluster. The runtime version, executor count, and library set are the
# MAGIC cluster's, not your laptop's.

# COMMAND ----------

import os
import sys
import platform

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("DATABRICKS_RUNTIME_VERSION:", os.environ.get("DATABRICKS_RUNTIME_VERSION", "<not set — you are NOT on a cluster>"))
print("Spark version:", spark.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Read raw data
# MAGIC
# MAGIC `samples.nyctaxi.trips` is a public table available in every Databricks workspace.

# COMMAND ----------

from pyspark.sql import functions as F

trips = spark.table("samples.nyctaxi.trips")
trips.printSchema()
print(f"Row count: {trips.count():,}")
display(trips.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Clean + derive features
# MAGIC
# MAGIC Filter out implausible records, then derive `trip_duration_min` and `pickup_hour`.
# MAGIC These are pure Spark transformations — they run on the cluster's executors.

# COMMAND ----------

clean = (
    trips
    .where(F.col("fare_amount").between(2.5, 200))
    .where(F.col("trip_distance") > 0)
    .withColumn(
        "trip_duration_min",
        (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0,
    )
    .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    .where(F.col("trip_duration_min").between(0.5, 180))
)

print(f"Clean row count: {clean.count():,}")
display(clean.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Aggregate: fare and distance by pickup hour

# COMMAND ----------

by_hour = (
    clean.groupBy("pickup_hour")
    .agg(
        F.count("*").alias("trips"),
        F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
        F.round(F.avg("trip_distance"), 2).alias("avg_distance"),
        F.round(F.avg("trip_duration_min"), 2).alias("avg_duration_min"),
    )
    .orderBy("pickup_hour")
)

display(by_hour)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Quick visualization
# MAGIC
# MAGIC `display()` lets you switch to a chart view in the Databricks output pane.
# MAGIC The matplotlib alternative below works because the cluster's ML Runtime
# MAGIC already has matplotlib installed — on your laptop you'd need to `uv add` it.

# COMMAND ----------

import matplotlib.pyplot as plt

pdf = by_hour.toPandas()

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(pdf["pickup_hour"], pdf["trips"])
ax.set_xlabel("Pickup hour")
ax.set_ylabel("Trips")
ax.set_title("NYC taxi trips by pickup hour")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. What this demo shows
# MAGIC
# MAGIC - The cell outputs above (`Python: 3.x.x`, `DATABRICKS_RUNTIME_VERSION: ...`,
# MAGIC   `Spark master: ...`) confirm the entire script ran on the cluster.
# MAGIC - The `spark` and `display()` symbols are injected by the Databricks runtime —
# MAGIC   no `from databricks.connect import DatabricksSession` boilerplate needed.
# MAGIC - Cluster-side libraries (matplotlib, pandas, etc.) are available without any
# MAGIC   local install.
# MAGIC
# MAGIC ### Compare to Databricks Connect (`src/ml_test/`)
# MAGIC
# MAGIC | | This notebook (Extension) | `src/ml_test/train.py` (DBC) |
# MAGIC |---|---|---|
# MAGIC | Python process | Cluster | Your laptop |
# MAGIC | Need `databricks-connect` locally | No | Yes |
# MAGIC | Need `spark = DatabricksSession.builder.getOrCreate()` | No (auto-injected) | Yes |
# MAGIC | Breakpoints / step debugger | No (cluster-side) | Yes (VS Code debugger) |
# MAGIC | Cluster needs your deps installed | Yes | No |
# MAGIC | Best for | Notebook-style exploration | Modular packaged code, CI/CD |
