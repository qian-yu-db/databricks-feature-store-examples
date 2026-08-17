# Databricks ML Examples

Runnable Databricks examples and reference notes for development setup, feature
engineering, feature serving, model training, hyperparameter tuning, and MLOps
project scaffolding.

Examples are grouped into four topic folders — `development_setup/`,
`feature_engineering/`, `model_training/`, and `mlops/` — each containing
self-contained sub-projects with their own README and numbered Databricks
notebooks/scripts. The root `pyproject.toml` provides a local Python environment
for notebook editing, load testing, and utility work, while the notebooks
themselves install any workspace-specific dependencies they need.

## Repository Layout

| Topic | Sub-project | Contents |
| --- | --- | --- |
| [`development_setup/`](./development_setup) | [`vscode_example/`](./development_setup/vscode_example) | Local IDE workflow with Databricks Connect, VS Code, and Databricks Asset Bundles — environment setup, `src/` package, and demo notebooks. See its `GUIDE.md`. |
| [`feature_engineering/`](./feature_engineering) | [`on_demand_feature/`](./feature_engineering/on_demand_feature) | End-to-end managed on-demand feature example using Unity Catalog Python UDFs, `FeatureEngineeringClient`, `FeatureFunction`, XGBoost, MLflow, and Databricks Model Serving. |
| | [`custom_feature_serving/`](./feature_engineering/custom_feature_serving) | Custom insurance-claims feature transformer packaged as a wheel and deployed through both Databricks Model Serving and Databricks Apps for latency comparison. |
| | [`uc_python_udf_custom_deps/`](./feature_engineering/uc_python_udf_custom_deps) | Modular, Python-driven pattern for registering Unity Catalog Python UDFs that carry custom pip dependencies via the `ENVIRONMENT` clause. |
| [`model_training/`](./model_training) | [`xgboost_optuna_example/`](./model_training/xgboost_optuna_example) | Training and hyperparameter-tuning a model on a dataset larger than a single GPU's memory using classic multi-GPU compute — three patterns (in-memory, NVMe streaming, distributed Spark) with XGBoost + Optuna. |
| [`mlops/`](./mlops) | [`mlops_stack_prep/`](./mlops/mlops_stack_prep) | Guides for simplifying and customizing Databricks MLOps Stacks / Asset Bundle templates. |
| [`docs/`](./docs) | — | *Machine Learning on Databricks* reference guide (`ml-on-databricks-guide`, `.md` + `.pdf`) covering dev setup, the ML lifecycle, platform capabilities, and MLOps promotion patterns. |
| [`imgs/`](./imgs) | — | Diagrams used by the feature store and feature serving materials. |

## Examples

### Development Setup

[`development_setup/vscode_example/`](./development_setup/vscode_example)
demonstrates a local IDE development workflow for Databricks: authoring code and
notebooks in VS Code, running them against a workspace with Databricks Connect,
and deploying with Databricks Asset Bundles (`databricks.yml`). Start with its
[`GUIDE.md`](./development_setup/vscode_example/GUIDE.md) for environment setup,
then explore the `src/` package and demo notebooks.

### On-Demand Features

[`feature_engineering/on_demand_feature/`](./feature_engineering/on_demand_feature) demonstrates a loan-application
credit-risk workflow where derived features are computed by Unity Catalog Python
UDFs at both training and serving time. The model is logged with
`fe.log_model()` so Databricks resolves the `FeatureFunction` bindings before
the pyfunc model runs.

Run the notebooks in order:

| File | Purpose |
| --- | --- |
| `01_generate_data.py` | Generate synthetic customer, loan application, and payment-history Delta tables. |
| `02_register_features.py` | Register Unity Catalog Python UDFs for derived features such as debt-to-income and credit utilization. |
| `03_train_models.py` | Build the training set and train five XGBoost risk sub-models. |
| `04_log_model.py` | Log and register the ensemble pyfunc model with on-demand feature bindings. |
| `05_inference.py` | Run batch scoring, direct pyfunc inference, and Model Serving endpoint queries. |

### Custom Feature Serving

[`feature_engineering/custom_feature_serving/`](./feature_engineering/custom_feature_serving) builds a deterministic
Guidewire ClaimCenter-style feature transformer (`claims_fe`) and serves the
same wheel through two deployment shells:

| Track | Stack | Primary files |
| --- | --- | --- |
| Model Serving pyfunc | `mlflow.pyfunc.PythonModel` deployed to Databricks Model Serving | `03_log_and_deploy.ipynb`, `04_test_fe_endpoint.ipynb` |
| Databricks App | FastAPI + uvicorn + orjson with a Gradio UI | `05_build_and_deploy_app.ipynb`, `app/` |

The workflow notebooks cover synthetic payload generation, wheel build,
deployment, endpoint testing, app deployment, and side-by-side comparison:
`01_generate_synthetic_payloads.ipynb` through `06_compare_endpoints.ipynb`.
Additional Locust-based load testing utilities live in
[`feature_engineering/custom_feature_serving/load_testing/`](./feature_engineering/custom_feature_serving/load_testing).

### UC Python UDFs with Custom Dependencies

[`feature_engineering/uc_python_udf_custom_deps/`](./feature_engineering/uc_python_udf_custom_deps)
shows how to register Unity Catalog Python UDFs that carry **custom pip
dependencies** via the `ENVIRONMENT` clause. UDF bodies live in
`udf_functions.py` and shared dependencies in `requirements.txt`; the notebook
extracts each function body with `inspect` and generates one
`CREATE OR REPLACE FUNCTION … ENVIRONMENT (…)` per UDF, so all UDFs share a
single environment. This is a UDF authoring/packaging pattern, distinct from the
on-demand feature example above.

### Training & Tuning on Large Datasets with Classic GPU Compute

[`model_training/xgboost_optuna_example/`](./model_training/xgboost_optuna_example) tackles
a recurring problem: training and hyperparameter-tuning a model when the dataset is **larger
than a single GPU's memory**, on **classic multi-GPU compute**. It compares three patterns
for spending a fixed multi-GPU budget on a search (XGBoost + Optuna), trading per-trial
capacity against search throughput:

| File | Pattern |
| --- | --- |
| `xgboost_optuna_approach_1.py` | Data held in memory; N concurrent trials, one per GPU (`MlflowSparkStudy`). |
| `xgboost_optuna_approach_2.py` | Data streamed from local NVMe; N concurrent trials, one per GPU. |
| `xgboost_optuna_approach_3.py` | Distributed `xgboost.spark`; one trial at a time across all N GPUs. |

The child README explains the VRAM math, CPU-memory pressure, and trial-parallelism
trade-offs, plus how to choose among the three.

## Local Environment

This repository uses `uv` for local dependency management:

```bash
uv sync
```

The root environment targets Python 3.12 and includes common dependencies used
across the examples, including XGBoost, MLflow, Databricks Connect, the
Databricks SDK, pandas, NumPy, Faker, scikit-learn, Locust, and matplotlib.

Use the local environment for editing notebooks, running the load-test client,
or developing supporting code. The Databricks examples themselves are intended
to run in a Databricks workspace with access to Unity Catalog, MLflow, Model
Serving, and the workspace paths/catalogs configured inside each notebook.

## Configuration Notes

- Several notebooks use example catalog and schema names such as
  `fins_genai.classic_ml`; update those constants before running in your own
  workspace.
- The on-demand feature example requires Unity Catalog and Databricks Feature
  Engineering in Unity Catalog.
- The custom feature serving example requires permissions to create/register
  models, deploy Model Serving endpoints, and deploy Databricks Apps.
- Load testing requires a service principal with `CAN_QUERY` on the target
  serving endpoint.

See each subdirectory README for the complete setup, execution order, and
deployment details.
