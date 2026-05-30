# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Inference: Score Loan Applications
# MAGIC
# MAGIC Two inference modes:
# MAGIC 1. **Batch with on-demand features** — `fe.score_batch()` resolves FeatureLookups + FeatureFunctions
# MAGIC    automatically. Returns a single `prediction` column (DOUBLE = overall_risk_score).
# MAGIC 2. **Direct pyfunc for full breakdown** — `mlflow.pyfunc.load_model()` with manual feature
# MAGIC    enrichment gives access to all 5 sub-scores + risk decision.

# COMMAND ----------

# MAGIC %pip install xgboost databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "fins_genai"
SCHEMA = "classic_ml"
ENSEMBLE_MODEL_NAME = "ensemble_risk_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Batch Inference with `fe.score_batch()`
# MAGIC
# MAGIC On-demand UC functions (`compute_debt_to_income`, etc.) and payment feature lookups
# MAGIC are resolved automatically. The `prediction` column is the `overall_risk_score` (DOUBLE).

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F
import mlflow

fe = FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")

registered_model_name = f"{CATALOG}.{SCHEMA}.{ENSEMBLE_MODEL_NAME}"

# Get latest model version
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
versions = list(w.model_versions.list(full_name=registered_model_name))
latest_version = str(max(int(v.version) for v in versions))
model_uri = f"models:/{registered_model_name}/{latest_version}"
print(f"Using model: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score existing loan applications

# COMMAND ----------

# Join with payment history — payment fields are raw input (no online table lookup)
apps_to_score = spark.table(f"{CATALOG}.{SCHEMA}.loan_applications").limit(20)
payments = spark.table(f"{CATALOG}.{SCHEMA}.payment_history")
apps_to_score = apps_to_score.join(payments, on="application_id", how="left")
apps_to_score = apps_to_score.withColumn(
    "ltv_ratio",
    F.when(F.col("property_value") > 0, F.col("loan_amount") / F.col("property_value")).otherwise(0.0),
)

# prediction column = overall_risk_score (DOUBLE)
scored = fe.score_batch(model_uri=model_uri, df=apps_to_score)

# Add risk decision from the score
scored_with_decision = scored.withColumn(
    "risk_decision",
    F.when(F.col("prediction") < 0.3, "APPROVE")
     .when(F.col("prediction") > 0.6, "DENY")
     .otherwise("REVIEW")
)

display(scored_with_decision.select(
    "application_id", "annual_income", "credit_score",
    F.col("prediction").alias("overall_risk_score"),
    "risk_decision",
    "debt_to_income", "credit_utilization", "payment_velocity", "risk_category",
))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score new applications (not in Delta)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

new_apps_schema = StructType([
    StructField("application_id", StringType()),
    StructField("customer_id", StringType()),
    StructField("annual_income", DoubleType()),
    StructField("total_debt", DoubleType()),
    StructField("credit_score", IntegerType()),
    StructField("credit_limit", DoubleType()),
    StructField("credit_used", DoubleType()),
    StructField("loan_amount", DoubleType()),
    StructField("property_value", DoubleType()),
    StructField("loan_purpose", StringType()),
    StructField("property_type", StringType()),
    StructField("ltv_ratio", DoubleType()),
    # Payment fields — provided by caller (no online table lookup)
    StructField("payments_last_30d", IntegerType()),
    StructField("payments_last_90d", IntegerType()),
    StructField("missed_payments_12m", IntegerType()),
    StructField("on_time_pct", DoubleType()),
])

new_apps = spark.createDataFrame([
    ("NEW-001", "CUST-000001", 150000.0, 20000.0, 790, 60000.0, 6000.0,
     250000.0, 500000.0, "purchase", "single_family", 0.5,
     4, 12, 0, 0.99),
    ("NEW-002", "CUST-000002", 25000.0, 22000.0, 480, 5000.0, 4800.0,
     175000.0, 180000.0, "refinance", "multi_family", 0.97,
     0, 1, 8, 0.45),
    ("NEW-003", "CUST-000003", 65000.0, 26000.0, 660, 25000.0, 12000.0,
     250000.0, 350000.0, "home_equity", "condo", 0.71,
     2, 6, 2, 0.82),
], schema=new_apps_schema)

scored_new = fe.score_batch(model_uri=model_uri, df=new_apps)
scored_new = scored_new.withColumn(
    "risk_decision",
    F.when(F.col("prediction") < 0.3, "APPROVE")
     .when(F.col("prediction") > 0.6, "DENY")
     .otherwise("REVIEW")
)

display(scored_new.select(
    "application_id", "annual_income", "credit_score",
    F.col("prediction").alias("overall_risk_score"),
    "risk_decision",
    "debt_to_income", "credit_utilization",
))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary Statistics

# COMMAND ----------

large_batch = spark.table(f"{CATALOG}.{SCHEMA}.loan_applications").limit(1000)
large_batch = large_batch.join(payments, on="application_id", how="left")
large_batch = large_batch.withColumn(
    "ltv_ratio",
    F.when(F.col("property_value") > 0, F.col("loan_amount") / F.col("property_value")).otherwise(0.0),
)

scored_large = fe.score_batch(model_uri=model_uri, df=large_batch)
scored_large = scored_large.withColumn(
    "risk_decision",
    F.when(F.col("prediction") < 0.3, "APPROVE")
     .when(F.col("prediction") > 0.6, "DENY")
     .otherwise("REVIEW")
)

print("Decision distribution:")
scored_large.groupBy("risk_decision").count().orderBy("risk_decision").show()

print("Score statistics:")
scored_large.select(
    F.mean("prediction").alias("mean_score"),
    F.min("prediction").alias("min_score"),
    F.max("prediction").alias("max_score"),
    F.stddev("prediction").alias("std_score"),
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Deploy to Model Serving Endpoint
# MAGIC
# MAGIC Creates or updates the serving endpoint. Takes ~10-15 min to become READY.

# COMMAND ----------

ENDPOINT_NAME = "credit-risk-ensemble"

from mlflow.deployments import get_deploy_client

mlflow.set_registry_uri("databricks-uc")
client = get_deploy_client("databricks")

entity_name = f"{CATALOG}.{SCHEMA}.{ENSEMBLE_MODEL_NAME}"

try:
    client.update_endpoint(
        endpoint=ENDPOINT_NAME,
        config={
            "served_entities": [{
                "entity_name": entity_name,
                "entity_version": latest_version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }],
        },
    )
    print(f"Updated endpoint '{ENDPOINT_NAME}' to version {latest_version}")
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e) or "not found" in str(e).lower():
        client.create_endpoint(
            name=ENDPOINT_NAME,
            config={
                "served_entities": [{
                    "entity_name": entity_name,
                    "entity_version": latest_version,
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                }],
            },
        )
        print(f"Created endpoint '{ENDPOINT_NAME}'")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check endpoint status
# MAGIC Re-run this cell until state is READY.

# COMMAND ----------

ep = w.serving_endpoints.get(ENDPOINT_NAME)
print(f"State: {ep.state.ready}")
print(f"Config update: {ep.state.config_update}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query endpoint
# MAGIC Only raw fields needed — on-demand features computed by the serving layer.

# COMMAND ----------

response = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_records=[
        {
            "annual_income": 150000.0, "total_debt": 20000.0, "credit_score": 790,
            "credit_limit": 60000.0, "credit_used": 6000.0, "loan_amount": 250000.0,
            "property_value": 500000.0, "loan_purpose": "purchase", "property_type": "single_family",
            "payments_last_30d": 4, "payments_last_90d": 12,
            "missed_payments_12m": 0, "on_time_pct": 0.99,
            "ltv_ratio": 250000.0 / 500000.0,
        },
        {
            "annual_income": 25000.0, "total_debt": 22000.0, "credit_score": 480,
            "credit_limit": 5000.0, "credit_used": 4800.0, "loan_amount": 175000.0,
            "property_value": 180000.0, "loan_purpose": "refinance", "property_type": "multi_family",
            "payments_last_30d": 0, "payments_last_90d": 1,
            "missed_payments_12m": 8, "on_time_pct": 0.45,
            "ltv_ratio": 175000.0 / 180000.0,
        },
    ],
)
print(response.predictions)

# COMMAND ----------

