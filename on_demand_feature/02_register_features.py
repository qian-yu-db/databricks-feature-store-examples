# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Register On-Demand Feature Functions in Unity Catalog
# MAGIC Registers 5 Python UDFs that compute derived features at serving time.

# COMMAND ----------

CATALOG = "fins_genai"
SCHEMA = "classic_ml"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Functions

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.compute_debt_to_income(annual_income DOUBLE, total_debt DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON
AS $$
if annual_income is None or total_debt is None:
    return None
if annual_income <= 0:
    return 0.0
return round(total_debt / annual_income, 6)
$$
""")
print("Registered: compute_debt_to_income")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.compute_credit_utilization(credit_used DOUBLE, credit_limit DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON
AS $$
if credit_used is None or credit_limit is None:
    return None
if credit_limit <= 0:
    return 0.0
return round(min(credit_used / credit_limit, 1.0), 6)
$$
""")
print("Registered: compute_credit_utilization")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.compute_payment_velocity(payments_last_30d INT, payments_last_90d INT)
RETURNS DOUBLE
LANGUAGE PYTHON
AS $$
if payments_last_30d is None or payments_last_90d is None:
    return None
if payments_last_90d <= 0:
    return 1.0
monthly_avg = payments_last_90d / 3.0
return round(payments_last_30d / monthly_avg, 6)
$$
""")
print("Registered: compute_payment_velocity")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.compute_account_age_months(account_open_date DATE)
RETURNS INT
LANGUAGE PYTHON
AS $$
if account_open_date is None:
    return None
from datetime import date
today = date.today()
months = (today.year - account_open_date.year) * 12 + (today.month - account_open_date.month)
return max(months, 0)
$$
""")
print("Registered: compute_account_age_months")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.compute_risk_category(credit_score INT, dti_ratio DOUBLE)
RETURNS STRING
LANGUAGE PYTHON
AS $$
if credit_score is None or dti_ratio is None:
    return None
if credit_score >= 740 and dti_ratio < 0.3:
    return "low"
elif credit_score >= 620 and dti_ratio < 0.5:
    return "medium"
else:
    return "high"
$$
""")
print("Registered: compute_risk_category")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Functions

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   fins_genai.classic_ml.compute_debt_to_income(60000, 18000) as dti,
# MAGIC   fins_genai.classic_ml.compute_credit_utilization(8000, 10000) as credit_util,
# MAGIC   fins_genai.classic_ml.compute_payment_velocity(3, 9) as pay_velocity,
# MAGIC   fins_genai.classic_ml.compute_risk_category(750, 0.25) as risk_low,
# MAGIC   fins_genai.classic_ml.compute_risk_category(650, 0.4) as risk_med,
# MAGIC   fins_genai.classic_ml.compute_risk_category(550, 0.6) as risk_high,
# MAGIC   fins_genai.classic_ml.compute_debt_to_income(0, 18000) as dti_zero,
# MAGIC   fins_genai.classic_ml.compute_debt_to_income(NULL, 18000) as dti_null