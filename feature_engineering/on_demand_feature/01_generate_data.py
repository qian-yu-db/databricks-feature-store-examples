# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Generate Synthetic Loan Application Data
# MAGIC Generates realistic loan application data with correlated features and writes to Delta tables.
# MAGIC
# MAGIC **Tables created:**
# MAGIC - `customer_profiles` — 10K customers with demographics
# MAGIC - `loan_applications` — 50K applications with credit/loan details
# MAGIC - `payment_history` — Payment behavior per application

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %pip install faker
# MAGIC %restart_python

# COMMAND ----------

CATALOG = "fins_genai"
SCHEMA = "classic_ml"

N_CUSTOMERS = 10_000
N_APPLICATIONS = 50_000
N_PARTITIONS = 16

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Infrastructure

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Customer Profiles

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd

@F.pandas_udf("struct<name:string,email:string,city:string,state:string,employer:string>")
def fake_customer_details(ids: pd.Series) -> pd.DataFrame:
    from faker import Faker
    fake = Faker()
    Faker.seed(42)
    rows = []
    for _ in range(len(ids)):
        rows.append({
            "name": fake.name(),
            "email": fake.email(),
            "city": fake.city(),
            "state": fake.state_abbr(),
            "employer": fake.company(),
        })
    return pd.DataFrame(rows)


customers_df = spark.range(0, N_CUSTOMERS, numPartitions=N_PARTITIONS).select(
    F.concat(F.lit("CUST-"), F.lpad(F.col("id").cast("string"), 6, "0")).alias("customer_id"),
    F.col("id").alias("customer_idx"),
    fake_customer_details(F.col("id")).alias("details"),
).select(
    "customer_id",
    "customer_idx",
    F.col("details.name").alias("name"),
    F.col("details.email").alias("email"),
    F.col("details.city").alias("city"),
    F.col("details.state").alias("state"),
    F.col("details.employer").alias("employer"),
    (F.abs(F.randn(seed=42) * 5) + 1).cast("int").alias("employment_years"),
    (F.abs(F.randn(seed=43) * 3) + 0.5).cast("int").alias("address_years"),
)
customers_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.customer_profiles")
print(f"Wrote customer_profiles: {N_CUSTOMERS} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Loan Applications

# COMMAND ----------

import numpy as np

customer_lookup = spark.table(f"{CATALOG}.{SCHEMA}.customer_profiles").select("customer_idx", "customer_id")

@F.pandas_udf("struct<annual_income:double,total_debt:double,credit_score:int,credit_limit:double,credit_used:double,loan_amount:double,property_value:double,loan_purpose:string,property_type:string>")
def fake_loan_details(ids: pd.Series) -> pd.DataFrame:
    import numpy as np
    np.random.seed(42)
    n = len(ids)
    rows = []
    for i in range(n):
        income = np.clip(np.random.lognormal(mean=10.9, sigma=0.6), 20000, 500000)
        credit_score = int(np.clip(np.random.normal(680, 80), 300, 850))
        dti_base = 0.15 if credit_score > 720 else (0.30 if credit_score > 620 else 0.50)
        total_debt = income * np.random.uniform(dti_base * 0.5, dti_base * 1.5)
        credit_limit = income * np.random.uniform(0.3, 1.2)
        util_rate = 0.2 if credit_score > 720 else (0.5 if credit_score > 620 else 0.8)
        credit_used = credit_limit * np.random.uniform(util_rate * 0.3, min(util_rate * 1.5, 1.0))
        loan_amount = np.clip(np.random.lognormal(mean=12.0, sigma=0.5), 10000, 1000000)
        ltv = np.random.uniform(0.5, 0.95)
        property_value = loan_amount / ltv
        purpose = np.random.choice(["purchase", "refinance", "home_equity"], p=[0.5, 0.35, 0.15])
        prop_type = np.random.choice(["single_family", "condo", "multi_family", "townhouse"], p=[0.55, 0.20, 0.10, 0.15])
        rows.append({
            "annual_income": round(income, 2), "total_debt": round(total_debt, 2),
            "credit_score": credit_score, "credit_limit": round(credit_limit, 2),
            "credit_used": round(credit_used, 2), "loan_amount": round(loan_amount, 2),
            "property_value": round(property_value, 2), "loan_purpose": purpose,
            "property_type": prop_type,
        })
    return pd.DataFrame(rows)


apps_df = spark.range(0, N_APPLICATIONS, numPartitions=N_PARTITIONS).select(
    F.concat(F.lit("APP-"), F.lpad(F.col("id").cast("string"), 7, "0")).alias("application_id"),
    (F.abs(F.hash(F.col("id"))) % N_CUSTOMERS).alias("customer_idx"),
    fake_loan_details(F.col("id")).alias("details"),
    F.date_sub(F.current_date(), (F.abs(F.hash(F.col("id") + 1000)) % 365).cast("int")).alias("application_date"),
).select(
    "application_id", "customer_idx", "application_date",
    F.col("details.annual_income").alias("annual_income"),
    F.col("details.total_debt").alias("total_debt"),
    F.col("details.credit_score").alias("credit_score"),
    F.col("details.credit_limit").alias("credit_limit"),
    F.col("details.credit_used").alias("credit_used"),
    F.col("details.loan_amount").alias("loan_amount"),
    F.col("details.property_value").alias("property_value"),
    F.col("details.loan_purpose").alias("loan_purpose"),
    F.col("details.property_type").alias("property_type"),
)

apps_with_fk = apps_df.join(customer_lookup, on="customer_idx").drop("customer_idx")
apps_with_fk.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.loan_applications")
print(f"Wrote loan_applications: {N_APPLICATIONS} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Payment History

# COMMAND ----------

@F.pandas_udf("struct<payments_last_30d:int,payments_last_90d:int,missed_payments_12m:int,on_time_pct:double>")
def fake_payment_history(credit_scores: pd.Series) -> pd.DataFrame:
    import numpy as np
    np.random.seed(42)
    rows = []
    for cs in credit_scores:
        if cs > 720:
            p30, p90, missed = int(np.random.poisson(3)), int(np.random.poisson(9)), int(np.random.poisson(0.1))
            on_time = np.clip(np.random.normal(0.97, 0.02), 0.85, 1.0)
        elif cs > 620:
            p30, p90, missed = int(np.random.poisson(2)), int(np.random.poisson(6)), int(np.random.poisson(1.5))
            on_time = np.clip(np.random.normal(0.85, 0.08), 0.50, 1.0)
        else:
            p30, p90, missed = int(np.random.poisson(1)), int(np.random.poisson(3)), int(np.random.poisson(4))
            on_time = np.clip(np.random.normal(0.65, 0.15), 0.20, 1.0)
        rows.append({"payments_last_30d": p30, "payments_last_90d": max(p90, p30),
                      "missed_payments_12m": missed, "on_time_pct": round(on_time, 4)})
    return pd.DataFrame(rows)


payment_df = spark.table(f"{CATALOG}.{SCHEMA}.loan_applications").select(
    "application_id", "credit_score",
).withColumn("payment_info", fake_payment_history(F.col("credit_score"))).select(
    "application_id",
    F.col("payment_info.payments_last_30d").alias("payments_last_30d"),
    F.col("payment_info.payments_last_90d").alias("payments_last_90d"),
    F.col("payment_info.missed_payments_12m").alias("missed_payments_12m"),
    F.col("payment_info.on_time_pct").alias("on_time_pct"),
)
payment_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.payment_history")
print(f"Wrote payment_history: {N_APPLICATIONS} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

for table in ["customer_profiles", "loan_applications", "payment_history"]:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{table}").count()
    print(f"{table}: {count} rows")