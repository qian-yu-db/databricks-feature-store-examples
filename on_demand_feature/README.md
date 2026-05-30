# On-Demand Features (Managed, GA)

End-to-end example of **on-demand features** on Databricks: derived features are computed
by **Unity Catalog Python UDFs** that are bound to a model via
`FeatureEngineeringClient` + `FeatureFunction`. The same UDFs run at **both training and
serving time**, so feature logic is defined once and never drifts between offline and
online.

**Scenario:** a loan-application credit-risk service. An ensemble of 5 XGBoost models
scores each application and produces an overall risk score and an APPROVE / REVIEW / DENY
decision.

## Why this pattern (Option A — no online table)

- All **raw** features (loan fields + payment history) are supplied directly in the request.
- On-demand `FeatureFunction` UDFs compute the **derived** features (debt-to-income,
  credit utilization, etc.) automatically.
- **No `FeatureLookup` → no online table dependency** → simpler, lower-latency deployment.

Use this when the derived features are cheap, deterministic functions of the request
payload. If features must be looked up from a precomputed store (e.g. aggregates keyed by
customer), you would add `FeatureLookup` + an online table instead.

## Configuration

All notebooks share the same constants — edit them at the top of each notebook to point at
your own catalog/schema:

```python
CATALOG = "fins_genai"
SCHEMA  = "classic_ml"
```

## On-demand UC functions

Registered in `02_register_features.py` (Unity Catalog Python UDFs):

| Function | Inputs | Output |
|----------|--------|--------|
| `compute_debt_to_income` | `annual_income`, `total_debt` | DTI ratio (DOUBLE) |
| `compute_credit_utilization` | `credit_used`, `credit_limit` | utilization 0–1 (DOUBLE) |
| `compute_payment_velocity` | `payments_last_30d`, `payments_last_90d` | velocity vs. 90d avg (DOUBLE) |
| `compute_account_age_months` | `account_open_date` | months since open (INT) |
| `compute_risk_category` | `credit_score`, `dti_ratio` | `low` / `medium` / `high` (STRING) |

These are bound to model inputs as `FeatureFunction`s in notebooks 03 and 04. Note
`compute_risk_category` consumes the **output** of `compute_debt_to_income`
(`dti_ratio → debt_to_income`), demonstrating chained on-demand features.

## Notebooks

Run in order (`01` → `05`) in a Databricks workspace. Each notebook installs its own
dependencies via `%pip`.

| Notebook | Purpose |
|----------|---------|
| `01_generate_data.py` | Generate synthetic Delta tables: `customer_profiles` (10K), `loan_applications` (50K), `payment_history`. Features are correlated for learnable signal. |
| `02_register_features.py` | Register the 5 on-demand feature UDFs in Unity Catalog and verify them with a SQL smoke test. |
| `03_train_models.py` | Build a training set with `fe.create_training_set()` (UDFs compute derived features), then train 5 XGBoost models — one per risk dimension — and save artifacts to a UC Volume. |
| `04_log_model.py` | Log the `EnsembleRiskModel` pyfunc with `fe.log_model()` (models-from-code), capturing the `FeatureFunction` bindings and registering to Unity Catalog. |
| `05_inference.py` | Batch scoring with `fe.score_batch()`, scoring brand-new applications, summary stats, and deploying + querying a Model Serving endpoint. |

## The ensemble model

`04_log_model.py` logs an `mlflow.pyfunc.PythonModel` (`EnsembleRiskModel`) that loads 5
XGBoost models and fans out inference across them in parallel (`ThreadPoolExecutor`), then
combines the sub-scores with fixed weights into a single `overall_risk_score`:

| Sub-model | Weight | Feature set (raw + on-demand) |
|-----------|-------:|-------------------------------|
| `credit_risk`  | 0.30 | credit_score, debt_to_income, on_time_pct, missed_payments_12m, credit_utilization |
| `fraud_signal` | 0.20 | credit_utilization, loan_amount, annual_income, debt_to_income, payment_velocity |
| `income_verify`| 0.15 | annual_income, total_debt, debt_to_income, credit_score, loan_amount |
| `behavior`     | 0.20 | payments_last_30d/90d, missed_payments_12m, on_time_pct, payment_velocity |
| `market_risk`  | 0.15 | ltv_ratio, loan_amount, credit_score, debt_to_income, credit_utilization |

Because it is logged with `fe.log_model(training_set=...)`, the serving layer computes the
on-demand features (`debt_to_income`, `credit_utilization`, `payment_velocity`,
`risk_category`) **before** the pyfunc runs — the model code never re-implements feature
logic.

## Inference

**Batch** — `fe.score_batch()` resolves the `FeatureFunction`s automatically; the
`prediction` column is the `overall_risk_score`:

```python
scored = fe.score_batch(model_uri=model_uri, df=apps_to_score)
```

**Real-time** — deploy to a Model Serving endpoint (`credit-risk-ensemble`) and query with
only the **raw** fields; derived features are computed server-side:

```python
w.serving_endpoints.query(
    name="credit-risk-ensemble",
    dataframe_records=[{
        "annual_income": 150000.0, "total_debt": 20000.0, "credit_score": 790,
        "credit_limit": 60000.0, "credit_used": 6000.0, "loan_amount": 250000.0,
        "property_value": 500000.0, "loan_purpose": "purchase",
        "property_type": "single_family", "payments_last_30d": 4,
        "payments_last_90d": 12, "missed_payments_12m": 0, "on_time_pct": 0.99,
        "ltv_ratio": 0.5,
    }],
)
```

Risk decision thresholds applied downstream: `< 0.3` → APPROVE, `> 0.6` → DENY, otherwise
REVIEW.

## References

- [On-demand features](https://docs.databricks.com/aws/en/machine-learning/feature-store/on-demand-features)
- [Feature Engineering in Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/feature-store/)
- [Deploy and query Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/model-serving-intro)
