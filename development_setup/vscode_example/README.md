# Databricks VSCode Example

A side-by-side demo of two ways to develop on Databricks from VS Code, using the same workspace dataset (`samples.nyctaxi.trips`):

| Path | What it shows | Where Python runs |
|---|---|---|
| **`src/ml_test/`** | Modular Python package; trains a `RandomForestRegressor`, logs to MLflow, deployable as a wheel via Asset Bundles | **Local Interactive** (Databricks Connect) |
| **`notebooks/XGBoost Wine Classification End to End.ipynb`** | Jupyter notebook; End to End ML Workflow + mlflow experiment + UC model registery | **Local Interactive** (Databricks Connect) |
| **`notebooks/nyctaxi_transform_demo.py`** | Databricks-source notebook; Spark transforms + `display()` + inline matplotlib | **On the cluster** (VS Code Databricks Extension) |

See [`GUIDE.md`](./GUIDE.md) for the full walkthrough, conceptual diagrams, and shell commands.

---

## Prerequisites

- Python 3.12 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/)
- Databricks CLI v0.200+ with a valid `DEFAULT` profile pointing at `<your-workspace>.cloud.databricks.com`
- VS Code with the [Databricks Extension](https://docs.databricks.com/aws/en/dev-tools/vscode-ext/) 

```bash
databricks auth profiles    # DEFAULT should be Valid = YES
uv sync                     # install local deps into .venv/
```

---

## Quick start

### Example A — Modular Python via Databricks Connect

```bash
uv run train
```

Reads `samples.nyctaxi.trips` (Spark-side filters + feature derivation), pulls a 20k-row sample to pandas, trains a `RandomForestRegressor` locally, and logs params/metrics/model to MLflow experiment at `/Users/<your-email>/ml_test_nyctaxi` on your Databricks worksapce. The console prints a link to show the workspace MLflow UI.

### Example C — Jupyter Notebook via Databricks Connect

Use your local virual env `.venv` for juptyer python kernel. The notebook read wine classification dataset and perform an end to end ML workflow including training, hyper-parameter optimization, experiment tracking (on Databricks) and model registration on Databricks Unity Catalog

### Example C — Databricks Notebook via VS Code Databricks Extension

1. Open `notebooks/nyctaxi_transform_demo.py` in VS Code.
2. Attach a cluster in the Databricks sidebar (DBR 15.x+ ML Runtime, or serverless interactive).
3. Click the ▶ icon in the title bar (or right-click → **Run File on Databricks** → **Run File as Workflow on Databricks**).

The entire file uploads to the workspace and executes on the cluster. The first cell prints the cluster's Python / runtime / Spark version to make the execution context unmistakable.

### Deploy `src/ml_test/` as a job

Make changes to `databricks.yml` file for your workspace, then use the following command to deploy a databricks asset bundle for Example C as a job

```bash
databricks bundle validate
databricks bundle deploy
databricks bundle run train_job
```

Builds the `ml_test` wheel, uploads it, and runs the `train` entry point as a serverless `python_wheel_task` in the `[dev] train_job` job.

---

## Project layout

```
databricks_vscode_example/
├── src/
│   ├── ml_test/
│   │   ├── data_utils.py            # get_spark() + load_nyctaxi_pandas()
│   │   └── train.py                 # MLflow run: RandomForest + log model
│   └── utils/dev_utils.py
├── notebooks/
│   ├── XGBoost Wine Classification End to End.ipynb  # Jupyter notebook: end-to-end ML via Databricks Connect
│   └── nyctaxi_transform_demo.py    # VS Code extension demo (cluster-side execution)
├── pyproject.toml                   # `train` entry point + deps
├── databricks.yml                   # Asset Bundle: train_job
└── GUIDE.md                         # Full demo guide
```

For the full conceptual explanation (Databricks Connect vs Run on Databricks, when to pick which, live-demo script, AI-agent integration), read [`GUIDE.md`](./GUIDE.md).
