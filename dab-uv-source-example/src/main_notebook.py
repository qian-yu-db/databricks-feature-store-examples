# Databricks notebook source
# MAGIC %md
# MAGIC # Main Job Notebook — source-deployed via DAB (no wheel)
# MAGIC
# MAGIC This notebook is deployed as a plain .py file.
# MAGIC The package version is managed in `pyproject.toml`.

# COMMAND ----------

import numpy as np
import pandas as pd

# ── Read version from pyproject.toml (via importlib.metadata) ──
try:
    from importlib.metadata import version
    pkg_version = version("dab-uv-source-example")
except Exception:
    pkg_version = "unknown (not installed as editable)"

print(f"Running dab-uv-source-example v{pkg_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: process data with numpy + pandas

# COMMAND ----------

df = pd.DataFrame({
    "feature_a": np.random.randn(100),
    "feature_b": np.random.randn(100),
})
df["score"] = df["feature_a"] * 0.6 + df["feature_b"] * 0.4

print(f"Processed {len(df)} rows. Mean score: {df['score'].mean():.4f}")
display(spark.createDataFrame(df))
