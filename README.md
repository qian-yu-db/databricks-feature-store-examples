# databricks-feature-store-examples

End-to-end, runnable examples of the different feature engineering and feature serving
patterns available on Databricks — from managed on-demand UDFs and declarative feature
pipelines to fully custom pyfunc serving endpoints. Each example is self-contained and
includes the notebooks/code needed to reproduce it in a Databricks workspace.

## Examples in this repo

### [`on_demand_feature/`](./on_demand_feature) — On-demand features (managed, GA)

Computes derived features at training and serving time using Unity Catalog Python UDFs
bound to the model via `FeatureEngineeringClient` + `FeatureFunction`. No online table
required — the serving endpoint calls the UDFs automatically. Uses a synthetic
loan-application credit-risk scenario.

| Notebook | Purpose |
|----------|---------|
| `01_generate_data.py` | Generate synthetic customer / loan / payment Delta tables |
| `02_register_features.py` | Register on-demand feature UDFs in Unity Catalog |
| `03_train_models.py` | Train XGBoost models with on-demand `FeatureFunction` features |
| `04_log_model.py` | Log the ensemble model with `fe.log_model()` so UDFs run at serving time |
| `05_inference.py` | Batch (`fe.score_batch`) and direct pyfunc inference |

### [`custom_feature_serving/`](./custom_feature_serving) — Custom feature serving endpoint

A fully custom, CPU-only regex/dictionary feature transformer (`claims_fe` wheel) served
through **two parallel deployment tracks** and benchmarked head-to-head:

- **Track A — Model Serving pyfunc**: `mlflow.pyfunc.PythonModel` → MLflow scoring server → Databricks Model Serving.
- **Track B — Databricks App**: FastAPI + uvicorn + orjson with a Gradio UI.

Both tracks install and call the same wheel, so any latency delta reflects only the
serving shell. Numbered notebooks `01_…06_` run the end-to-end workflow (synthetic
payloads → build wheel → deploy → test → compare); `load_testing/` provides local load
tests. See [`custom_feature_serving/README.md`](./custom_feature_serving/README.md) for
the full walkthrough.