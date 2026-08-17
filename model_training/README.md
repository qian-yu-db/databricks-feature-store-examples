# Model Training

Examples for training and tuning models on Databricks, including distributed and
GPU-accelerated hyperparameter search patterns.

## Sub-projects

| Sub-project | Contents |
|-------------|----------|
| [`xgboost_optuna_example/`](./xgboost_optuna_example) | Training and hyperparameter-tuning a model on a **dataset larger than a single GPU's memory** using **classic multi-GPU compute** — three patterns (in-memory, NVMe streaming, distributed Spark) with XGBoost + Optuna and their capacity/throughput trade-offs. |

Each sub-project has its own README with setup and execution details.
