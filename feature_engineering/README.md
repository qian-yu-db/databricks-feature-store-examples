# Feature Engineering

Examples for building, serving, and packaging features on Databricks — from
on-demand feature functions bound to models, through custom feature transformers
served at low latency, to authoring the Unity Catalog Python UDFs that back them.

## Sub-projects

| Sub-project | Contents |
|-------------|----------|
| [`on_demand_feature/`](./on_demand_feature) | End-to-end **on-demand features**: Unity Catalog Python UDFs bound to an ensemble model via `FeatureEngineeringClient` + `FeatureFunction`, computed identically at training and serving time. Loan-application credit-risk scenario. |
| [`custom_feature_serving/`](./custom_feature_serving) | A custom insurance-claims **feature transformer** packaged as a wheel and deployed through both **Databricks Model Serving** and **Databricks Apps**, with load testing for latency comparison. |
| [`uc_python_udf_custom_deps/`](./uc_python_udf_custom_deps) | Modular, Python-driven pattern for registering **UC Python UDFs with custom pip dependencies** via the `ENVIRONMENT` clause. A UDF authoring/packaging example, distinct from the on-demand feature example above. |

Each sub-project has its own README with setup and execution details.
