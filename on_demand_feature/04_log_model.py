# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Log Ensemble Model with On-Demand Features
# MAGIC
# MAGIC Logs the EnsembleRiskModel using `FeatureEngineeringClient.log_model()` so that
# MAGIC on-demand UC functions are automatically called at serving time.
# MAGIC
# MAGIC **Key difference from vanilla `mlflow.pyfunc.log_model()`:**
# MAGIC - `fe.log_model()` captures the `FeatureFunction` bindings from the training set
# MAGIC - At serving time, the endpoint calls the UC Python UDFs automatically
# MAGIC - No need to replicate feature computation logic inside the pyfunc

# COMMAND ----------

# MAGIC %pip install xgboost databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "fins_genai"
SCHEMA = "classic_ml"
ENSEMBLE_MODEL_NAME = "ensemble_risk_model"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/model_artifacts"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Feature Specs (same as training)
# MAGIC These must match exactly what was used in `create_training_set()` in notebook 03.

# COMMAND ----------

from databricks.feature_engineering import (
    FeatureEngineeringClient,
    FeatureFunction,
)

fe = FeatureEngineeringClient()

# On-demand features only — no FeatureLookup (Online Tables are deprecated).
# Payment fields (payments_last_30d, etc.) are provided as raw input by the caller.
all_features = [
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recreate Training Set (needed for `fe.log_model()`)
# MAGIC `fe.log_model()` requires the training set object to capture feature bindings.

# COMMAND ----------

from pyspark.sql import functions as F

# Join loan applications with payment history — payment fields are raw input now (no FeatureLookup)
apps_df = spark.table(f"{CATALOG}.{SCHEMA}.loan_applications")
payments_df = spark.table(f"{CATALOG}.{SCHEMA}.payment_history")
apps_df = apps_df.join(payments_df, on="application_id", how="left")
apps_df = apps_df.withColumn(
    "ltv_ratio",
    F.when(F.col("property_value") > 0, F.col("loan_amount") / F.col("property_value")).otherwise(0.0),
)

# Add a dummy label column — can't reuse credit_score because it's an input to compute_risk_category
apps_df = apps_df.withColumn("_dummy_label", F.lit(0))

# We only need a small sample for logging — not the full training set
base_df = apps_df.limit(100)

training_set = fe.create_training_set(
    df=base_df,
    feature_lookups=all_features,
    label="_dummy_label",
    exclude_columns=["application_id", "customer_id", "application_date"],
)

print("Training set recreated for model logging")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Self-Contained Model Code
# MAGIC The pyfunc loads 5 XGBoost models and runs parallel inference.
# MAGIC On-demand feature computation is handled by the Feature Engineering serving layer —
# MAGIC the pyfunc receives **already-enriched** input with `debt_to_income`, `credit_utilization`,
# MAGIC `payment_velocity`, and `risk_category` columns already computed.

# COMMAND ----------

model_code = '''
import pickle
import concurrent.futures
import numpy as np
import pandas as pd
import mlflow


MODEL_WEIGHTS = {
    "credit_risk": 0.30, "fraud_signal": 0.20, "income_verify": 0.15,
    "behavior": 0.20, "market_risk": 0.15,
}
APPROVE_THRESHOLD = 0.3
DENY_THRESHOLD = 0.6


class EnsembleRiskModel(mlflow.pyfunc.PythonModel):
    """
    Multi-model credit risk scoring with parallel inference.

    When served via FeatureEngineeringClient, the input DataFrame arrives
    with on-demand features already computed by UC functions (debt_to_income,
    credit_utilization, payment_velocity, risk_category) and pre-materialized
    features from FeatureLookup (payments_last_30d, etc.).

    fe.score_batch() coerces output to a single DOUBLE `prediction` column,
    so predict() returns the overall_risk_score as a scalar per row.
    For the full sub-score breakdown, call predict() directly via
    mlflow.pyfunc.load_model() — see notebook 05.
    """

    def load_context(self, context):
        self.models = {}
        for name in MODEL_WEIGHTS:
            with open(context.artifacts[name], "rb") as f:
                data = pickle.load(f)
            self.models[name] = {"model": data["model"], "features": data["features"]}

    def _predict_single_model(self, model_name, row_dict):
        info = self.models[model_name]
        feature_values = [float(row_dict.get(f, 0.0) or 0.0) for f in info["features"]]
        X = np.array([feature_values])
        return float(info["model"].predict_proba(X)[0, 1])

    def predict(self, context, model_input, params=None):
        results = []
        for _, row in model_input.iterrows():
            row_dict = row.to_dict()

            # Fan-out: 5 models in parallel via ThreadPoolExecutor
            sub_scores = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_name = {
                    executor.submit(self._predict_single_model, name, row_dict): name
                    for name in MODEL_WEIGHTS
                }
                for future in concurrent.futures.as_completed(future_to_name):
                    sub_scores[future_to_name[future]] = future.result()

            # Weighted aggregation → single overall_risk_score
            overall_score = round(min(max(sum(
                sub_scores[n] * w for n, w in MODEL_WEIGHTS.items()
            ), 0.0), 1.0), 6)

            results.append(overall_score)

        return pd.DataFrame({"overall_risk_score": results})


mlflow.models.set_model(EnsembleRiskModel())
'''

model_code_path = f"{VOLUME_PATH}/ensemble_model.py"
with open(model_code_path, "w") as f:
    f.write(model_code)
print(f"Model code written to {model_code_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Model with `fe.log_model()`
# MAGIC This captures the FeatureFunction bindings so UC functions run at serving time.

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').first()[0]}/credit-risk-ensemble")

artifacts = {
    name: f"{VOLUME_PATH}/{name}.pkl"
    for name in ["credit_risk", "fraud_signal", "income_verify", "behavior", "market_risk"]
}

registered_model_name = f"{CATALOG}.{SCHEMA}.{ENSEMBLE_MODEL_NAME}"

with mlflow.start_run(run_name="ensemble_with_on_demand_features") as run:
    fe.log_model(
        model=model_code_path,      # "models from code" — self-contained .py file
        artifact_path="model",
        flavor=mlflow.pyfunc,
        training_set=training_set,   # captures FeatureFunction + FeatureLookup bindings
        artifacts=artifacts,
        registered_model_name=registered_model_name,
        extra_pip_requirements=["xgboost>=2.0", "scikit-learn"],
    )
    print(f"Run ID: {run.info.run_id}")

print(f"Registered: {registered_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify: Load Model and Check Feature Spec
# MAGIC The logged model should have a feature spec that includes the on-demand functions.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
versions = list(w.model_versions.list(full_name=registered_model_name))
latest = max(versions, key=lambda v: int(v.version))
print(f"Latest version: {latest.version}")
print(f"Status: {latest.status}")
print(f"Run ID: {latest.run_id}")