# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Train 5 XGBoost Models with On-Demand Features
# MAGIC
# MAGIC Uses `FeatureEngineeringClient` with `FeatureFunction` for on-demand feature computation.
# MAGIC UC Python UDFs (registered in notebook 02) compute derived features like debt-to-income
# MAGIC ratio and credit utilization at both training and serving time.
# MAGIC
# MAGIC **Option A architecture (no online table):**
# MAGIC - All raw features (loan fields + payment fields) are provided in the input
# MAGIC - On-demand `FeatureFunction` UDFs compute derived features automatically
# MAGIC - No `FeatureLookup` → no online table dependency → simpler deployment

# COMMAND ----------

# MAGIC %pip install xgboost databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "fins_genai"
SCHEMA = "classic_ml"
SEED = 42
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/model_artifacts"

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.model_artifacts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define On-Demand Feature Functions
# MAGIC
# MAGIC `FeatureFunction` binds UC Python UDFs to input columns. At training time,
# MAGIC `fe.create_training_set()` calls these UDFs to compute derived features.
# MAGIC At serving time, Model Serving calls the same UDFs automatically.
# MAGIC
# MAGIC No `FeatureLookup` is used — all raw features (including payment history)
# MAGIC are provided directly in the base DataFrame. This avoids the need for an
# MAGIC online table at serving time.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction

fe = FeatureEngineeringClient()

on_demand_features = [
    FeatureFunction(
        udf_name=f"{CATALOG}.{SCHEMA}.compute_debt_to_income",
        input_bindings={"annual_income": "annual_income", "total_debt": "total_debt"},
        output_name="debt_to_income",
    ),
    FeatureFunction(
        udf_name=f"{CATALOG}.{SCHEMA}.compute_credit_utilization",
        input_bindings={"credit_used": "credit_used", "credit_limit": "credit_limit"},
        output_name="credit_utilization",
    ),
    FeatureFunction(
        udf_name=f"{CATALOG}.{SCHEMA}.compute_payment_velocity",
        input_bindings={"payments_last_30d": "payments_last_30d", "payments_last_90d": "payments_last_90d"},
        output_name="payment_velocity",
    ),
    FeatureFunction(
        udf_name=f"{CATALOG}.{SCHEMA}.compute_risk_category",
        input_bindings={"credit_score": "credit_score", "dti_ratio": "debt_to_income"},
        output_name="risk_category",
    ),
]

print(f"Defined {len(on_demand_features)} on-demand FeatureFunctions (no FeatureLookup)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Base DataFrame
# MAGIC
# MAGIC The base DataFrame contains raw loan application fields + payment history + labels.
# MAGIC Payment data is joined directly (not via FeatureLookup) so no online table is needed.

# COMMAND ----------

import numpy as np
from pyspark.sql import functions as F

# Join loan applications with payment history
apps_df = spark.table(f"{CATALOG}.{SCHEMA}.loan_applications")
payments_df = spark.table(f"{CATALOG}.{SCHEMA}.payment_history")
apps_df = apps_df.join(payments_df, on="application_id", how="left")

# Compute LTV ratio directly (simple enough to not need a UC function)
apps_df = apps_df.withColumn(
    "ltv_ratio",
    F.when(F.col("property_value") > 0, F.col("loan_amount") / F.col("property_value")).otherwise(0.0),
)

# Generate labels (correlated with features for learnable signal)
apps_pd = apps_df.toPandas()
np.random.seed(SEED)
n = len(apps_pd)

dti = np.where(apps_pd["annual_income"] > 0, apps_pd["total_debt"] / apps_pd["annual_income"], 0)
util = np.where(apps_pd["credit_limit"] > 0, apps_pd["credit_used"] / apps_pd["credit_limit"], 0)
velocity = np.where(apps_pd["payments_last_90d"] > 0, apps_pd["payments_last_30d"] / (apps_pd["payments_last_90d"] / 3), 1.0)

default_prob = 0.3 * (dti / max(dti.max(), 1)) + 0.4 * (1 - (apps_pd["credit_score"] - 300) / 550) + 0.3 * (1 - apps_pd["on_time_pct"])
apps_pd["label_credit_risk"] = (np.random.random(n) < default_prob.clip(0, 0.8)).astype(int)
apps_pd["label_fraud_signal"] = (np.random.random(n) < (0.02 + 0.05 * np.clip(util, 0, 1)).clip(0, 0.15)).astype(int)
apps_pd["label_income_verify"] = (np.random.random(n) < (0.1 * np.clip(dti, 0, 2) / 2).clip(0, 0.3)).astype(int)

behavior_prob = 0.5 * (1 - apps_pd["on_time_pct"]) + 0.3 * (apps_pd["missed_payments_12m"] / max(apps_pd["missed_payments_12m"].max(), 1)) + 0.2 * (1 - np.clip(velocity, 0, 2) / 2)
apps_pd["label_behavior"] = (np.random.random(n) < behavior_prob.clip(0, 0.7)).astype(int)

market_prob = 0.3 * apps_pd["ltv_ratio"].clip(0, 1) + 0.2 * (apps_pd["property_type"] == "multi_family").astype(float) + 0.2 * (apps_pd["loan_amount"] / max(apps_pd["loan_amount"].max(), 1))
apps_pd["label_market_risk"] = (np.random.random(n) < market_prob.clip(0, 0.5)).astype(int)

# Rebuild Spark DataFrame with labels
label_cols = ["label_credit_risk", "label_fraud_signal", "label_income_verify", "label_behavior", "label_market_risk"]
labels_sdf = spark.createDataFrame(apps_pd[["application_id"] + label_cols])
base_df = apps_df.join(labels_sdf, on="application_id")

print(f"Base DataFrame: {base_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Training Set with On-Demand Features
# MAGIC
# MAGIC `fe.create_training_set()` calls the UC Python UDFs to compute `debt_to_income`,
# MAGIC `credit_utilization`, `payment_velocity`, and `risk_category` from raw input columns.

# COMMAND ----------

training_set = fe.create_training_set(
    df=base_df,
    feature_lookups=on_demand_features,
    label="label_credit_risk",
    exclude_columns=["application_id", "customer_id", "application_date"],
)

training_df = training_set.load_df()
display(training_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train 5 XGBoost Models
# MAGIC Each model uses a different feature subset and label. All features — both raw
# MAGIC (payment fields, credit_score) and on-demand (debt_to_income, credit_utilization) —
# MAGIC are available in the training DataFrame.

# COMMAND ----------

import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split

train_pd = training_df.toPandas()

FEATURE_SETS = {
    "credit_risk": ["credit_score", "debt_to_income", "on_time_pct", "missed_payments_12m", "credit_utilization"],
    "fraud_signal": ["credit_utilization", "loan_amount", "annual_income", "debt_to_income", "payment_velocity"],
    "income_verify": ["annual_income", "total_debt", "debt_to_income", "credit_score", "loan_amount"],
    "behavior": ["payments_last_30d", "payments_last_90d", "missed_payments_12m", "on_time_pct", "payment_velocity"],
    "market_risk": ["ltv_ratio", "loan_amount", "credit_score", "debt_to_income", "credit_utilization"],
}

LABEL_MAP = {
    "credit_risk": "label_credit_risk",
    "fraud_signal": "label_fraud_signal",
    "income_verify": "label_income_verify",
    "behavior": "label_behavior",
    "market_risk": "label_market_risk",
}

for model_name, features in FEATURE_SETS.items():
    X = train_pd[features].values
    y = train_pd[LABEL_MAP[model_name]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        objective="binary:logistic", eval_metric="auc", random_state=SEED,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test[:5])[:, 1]
    assert all(0 <= p <= 1 for p in probs)

    artifact_path = f"{VOLUME_PATH}/{model_name}.pkl"
    artifact = {"model": model, "features": features}
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"{model_name}: train_acc={model.score(X_train, y_train):.3f}, "
          f"test_acc={model.score(X_test, y_test):.3f} → {artifact_path}")

with open(f"{VOLUME_PATH}/feature_sets.pkl", "wb") as f:
    pickle.dump(FEATURE_SETS, f)

print(f"\nAll 5 models saved to {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Artifacts

# COMMAND ----------

import os
for f in sorted(os.listdir(VOLUME_PATH)):
    size = os.path.getsize(f"{VOLUME_PATH}/{f}")
    print(f"  {f}: {size:,} bytes")