# Databricks VSCode Examples

## 1. Introduction

This project demonstrates three complementary ways to develop on Databricks from VS Code:

- **`src/ml_test/`** — a **modular Python** package that trains a regression model on `samples.nyctaxi.trips`, logs the experiment to MLflow, and ships as a wheel deployed via Asset Bundles. Runs **locally via Databricks Connect**.
- **`notebooks/XGBoost Wine Classification End to End.ipynb`** — a **Jupyter notebook** that runs an end-to-end ML workflow (EDA → Optuna tuning → MLflow logging → UC model registration). Runs **locally via Databricks Connect** in the Jupyter kernel; interactive cell-by-cell development.
- **`notebooks/nyctaxi_transform_demo.py`** — a **Databricks-source notebook** that does data transformation but runs **entirely on the cluster** via the VS Code Databricks Extension's "Run File on Databricks".

Same workspace, three distinct execution models — the contrast is the whole point of the demo.

**Key tools:**

- **VS Code + Databricks Extension** — local IDE with cluster attach, variable explorer, and remote debugging
- **Databricks Connect** — execute PySpark code locally against a remote cluster (or serverless)
- **Asset Bundles** — declarative YAML-based deployment of jobs, wheels, and resources
- **Claude Code** — AI coding agent that can navigate, modify, and run the pipeline

**Why modular Python over notebooks?**

- Full IDE support: autocomplete, go-to-definition, refactoring, debugger
- Standard code review via pull request diffs (no JSON notebook diffs)
- Testable with pytest and runnable in CI/CD
- Each module has a clear entry point — easy to run, test, and compose

---

## 2. Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.12 (see `.python-version`) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Databricks CLI | v0.200+ configured with a `DEFAULT` profile |
| VS Code | With [Databricks Extension](https://marketplace.visualstudio.com/items?itemName=databricks.databricks) |
| Workspace | `<your-workspace>.cloud.databricks.com` |
| Demo data | `samples.nyctaxi.trips` (pre-loaded in every workspace; no setup needed) |

Verify your Databricks CLI profile:

```bash
databricks auth profiles
# Should show DEFAULT pointing to your workspace (Valid = YES)
```

---

## 3. Project Structure

```
databricks_vscode_example/
├── src/
│   ├── ml_test/
│   │   ├── __init__.py        # Package marker
│   │   ├── data_utils.py      # get_spark() + load_nyctaxi_pandas()
│   │   └── train.py           # MLflow run: train RandomForest, log params/metrics/model
│   └── utils/
│       └── dev_utils.py       # is_running_in_databricks()
├── notebooks/
│   ├── XGBoost Wine Classification End to End.ipynb  # Jupyter notebook: end-to-end ML via Databricks Connect
│   └── nyctaxi_transform_demo.py  # Databricks-source notebook for VS Code extension demo
├── pyproject.toml             # Package definition with `train` entry point
├── databricks.yml             # Asset Bundle config for remote deployment
├── .python-version            # Pins Python 3.12
└── .vscode/                   # VS Code / Databricks Extension settings
```

**Key design decisions:**

- `data_utils.get_spark()` auto-detects environment: returns a Databricks Connect session locally, a regular SparkSession when running on a cluster.
- `train.py` is the single entry point. It sets `mlflow.set_tracking_uri("databricks")` only when running locally so logging works in both contexts.
- `pyproject.toml` defines a `[project.scripts]` entry — `train = "ml_test.train:main"` — that Asset Bundles invoke as `python_wheel_task`.

---

## 3b. Understanding the Execution Models

There are two distinct ways to run Python code against Databricks from VS Code. This project includes a concrete example of **both** — `src/ml_test/` uses Databricks Connect; `notebooks/nyctaxi_transform_demo.py` uses the VS Code Extension.

### Databricks Connect (example: `src/ml_test/`)

Your Python process runs **locally**. Only Spark operations are sent to the remote cluster.

```
Your Laptop                              Databricks
┌──────────────────────────┐             ┌──────────────────────────┐
│ Python process            │             │ Cluster / Serverless     │
│                           │             │                          │
│  argparse, config     (local)           │                          │
│  sklearn.fit()        (local)           │                          │
│  pandas operations    (local)           │                          │
│  mlflow.log_model()   (local→REST)      │                          │
│                           │             │                          │
│  spark.table(...)     ──────────────►   │  Executes Spark SQL      │
│  df.filter().toPandas() ────────────►   │  Returns results         │
└──────────────────────────┘             └──────────────────────────┘
```

**What you install locally (`uv sync`):** sklearn, mlflow client, databricks-connect, databricks-sdk, pandas, etc.

**What you install on the cluster:** Nothing. The cluster already has PySpark and the ML Runtime pre-installed. Databricks Connect uses the cluster purely as a Spark execution engine.

**How to run:** From the VS Code integrated terminal:

```bash
uv run train
```

Or set a breakpoint in `train.py` and use VS Code's debugger with a launch config that runs the `train` module.

### VS Code Extension "Run on Databricks" (example: `notebooks/nyctaxi_transform_demo.py`)

The Databricks VS Code Extension has a mode where it **uploads** your `.py` file to the workspace and executes it **entirely on the cluster**, similar to running a notebook.

```
Your Laptop                              Databricks
┌──────────────────────────┐             ┌──────────────────────────┐
│ VS Code                   │  upload     │ Cluster                  │
│                           │ ────────►   │                          │
│  Edit code locally        │             │  ALL code runs here      │
│  View output in terminal  │  ◄────────  │  sklearn, mlflow, etc.   │
│                           │  results    │  needs %pip install      │
└──────────────────────────┘             └──────────────────────────┘
```

In this mode, the cluster needs **all** dependencies (sklearn, your `ml_test` package, etc.). You'd have to install them on the cluster first (via init scripts, cluster libraries, or `%pip install`). The notebook demo intentionally uses only libraries that ship with the ML Runtime (PySpark, pandas, matplotlib) so it runs with no extra setup.

### Which to use and why

| | Databricks Connect | Run on Databricks |
|---|---|---|
| **Code runs** | Locally (Spark ops remote) | Entirely on cluster |
| **Local deps needed** | Yes (`uv sync`) | No (just an editor) |
| **Cluster deps needed** | No (uses ML Runtime) | Yes (must install sklearn, etc.) |
| **Debugging** | Full VS Code debugger with breakpoints | Limited (print-based, no stepping) |
| **Best for** | Modular Python packages, iterative dev | Notebook-style scripts, quick experiments |
| **PandasUDFs** | Work — serialized and sent to cluster workers | Work — run directly on cluster |
| **Example in this repo** | `src/ml_test/` | `notebooks/nyctaxi_transform_demo.py` |

**Bottom line:** `uv sync` sets up your local environment. The cluster (or serverless) provides Spark. The DBC path needs no package installation on the cluster; the Extension path runs as if you were in a Databricks notebook.

### What the VS Code Databricks Extension gives you (regardless of which mode)

- **Workspace browser** — browse UC tables, MLflow experiments, and models in the sidebar
- **Cluster management** — start/stop clusters, view logs
- **Profile configuration** — manages your `~/.databrickscfg` connection
- **Sync status** — shows which workspace you're connected to
- **Run File on Databricks** — the upload-and-execute mode used by the notebook demo

The extension is **not required** to run the DBC example — `uv run` + Databricks Connect handles execution independently — but it is required for the notebook demo.

---

## 4. Getting Started

### Step 1: Install local dependencies

```bash
cd development_setup/vscode_example
uv sync
```

This installs sklearn, mlflow, databricks-connect, etc. into `.venv/`. Nothing is installed on any cluster.

### Step 2: Verify your Databricks profile

The code uses the `DEFAULT` profile from `~/.databrickscfg`. Confirm it exists and is valid:

```bash
databricks auth profiles
```

If your DEFAULT profile has `serverless_compute_id = auto`, Databricks Connect will use serverless compute automatically — no cluster to manage. Otherwise, you need a running classic cluster with DBR 16.1+ ML Runtime.

### Step 3: Verify connectivity

```bash
uv run python -c "from ml_test.data_utils import get_spark; print(get_spark())"
```

You should see a `DatabricksSession` object pointing to your workspace. If you get an import or auth error, re-check your `DEFAULT` profile.

### Step 4 (optional): Open in VS Code

1. Open the project folder in VS Code
2. Set the Python interpreter to `.venv/bin/python` (Cmd+Shift+P > "Python: Select Interpreter")
3. If you have the Databricks Extension installed, click the Databricks icon in the sidebar to browse your workspace

The extension is helpful for browsing but not required for running code.

---

## 5. Local Development Workflow

The training run executes locally via Databricks Connect. Your Python code (sklearn, mlflow) runs on your machine; the data read against `samples.nyctaxi.trips` happens on the remote cluster.

### Train and log to MLflow (~30s)

**What it does:**
- Calls `load_nyctaxi_pandas()` — reads `samples.nyctaxi.trips`, filters bad fares/distances, derives `trip_duration_min` and `pickup_hour` on the Spark side, and pulls a 20k-row sample to pandas.
- Trains a scikit-learn `RandomForestRegressor` locally.
- Creates (if needed) an MLflow experiment at `/Users/<your-email>/ml_test_nyctaxi` on the Databricks tracking server, then logs params, metrics (MAE, R²), and the sklearn model.

```bash
uv run train
```

**What to expect:**
- Console prints the run ID, MAE, R², and a link to the run in the workspace MLflow UI
- A new experiment under your user folder (first run only)
- A logged model artifact attached to the run

**Debugging in VS Code:**
1. Open `src/ml_test/train.py`
2. Set a breakpoint in `main()`
3. Press F5 / "Run and Debug" using the Python debugger
4. Step through training and inspect `X_train`, `y_pred`, etc.

---

## 5b. Jupyter Notebook Workflow (Databricks Connect)

The example is `notebooks/XGBoost Wine Classification End to End.ipynb`. Unlike the cluster-side `.py` notebook in 5c, this Jupyter notebook runs **locally in your `.venv` kernel** — same execution model as `src/ml_test/`, but in an interactive notebook UI so you can iterate cell by cell.

### What the notebook does (end-to-end ML workflow)

1. **Bootstrap** — Cell 0 calls `is_running_in_databricks()`; if you're local, it creates a Databricks Connect `DatabricksSession` from the `DEFAULT` profile. If you later open the same notebook inside the Databricks workspace, it skips the setup and uses the workspace `spark`.
2. **EDA** — Loads sklearn's Wine dataset (178 samples, 13 features, 3 classes), prints class balance and descriptive stats, plots class distribution, feature correlation heatmap, and per-class distributions for the top-correlated features.
3. **Preprocessing** — 60/20/20 stratified train/val/test split, `StandardScaler` fit on train only.
4. **Hyperparameter tuning** — Optuna study with 50 trials over XGBoost params (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, regularization, etc.), scored via 5-fold stratified CV. Each trial is logged as a **nested MLflow run** under a parent `optuna_tuning` run so you can compare them in the Experiment UI.
5. **Final training** — Refits XGBoost with the best params on train+val, evaluates on the held-out test set, and logs the model with an inferred signature and input example to MLflow.
6. **Evaluation** — Prints classification report, plots confusion matrix and gain-based feature importance.
7. **UC registration** — Registers the best model to Unity Catalog as `fins_genai.classic_ml.xgboost_wine_classifier` via `mlflow.set_registry_uri("databricks-uc")`. **Update the `CATALOG` / `SCHEMA` constants to your workspace before running this cell.**

The MLflow experiment lives at `/Users/<your-email>/xgboost_wine_classification` on the Databricks tracking server.

### How to run it from VS Code

1. Open `notebooks/XGBoost Wine Classification End to End.ipynb`.
2. In the kernel picker (top-right of the notebook), select the project's `.venv/bin/python` — same env created by `uv sync`.
3. Run cells top-to-bottom. The first cell prints `Connected to Databricks using profile: DEFAULT` to confirm Databricks Connect is wired up.
4. After tuning + training completes, the cell output prints clickable links into the Databricks MLflow Experiment UI and the UC model version page.

### Why this matters

- **Interactive iteration** with the same auth/profile setup that the `src/ml_test/` script uses — no separate kernel config needed.
- **Notebook ergonomics** (inline plots, rich `display()` tables, narrative markdown) without giving up the local debugger or the `.venv` you already manage.
- **Same workflow you'd run on the cluster** — the `is_running_in_databricks()` check means moving this notebook into the workspace requires no code changes.

---

## 5c. VS Code Extension Workflow (cluster-side execution)

This workflow demonstrates the **other** execution model — your code is uploaded to the workspace and runs entirely on the cluster, like a Databricks notebook. The example is `notebooks/nyctaxi_transform_demo.py`.

### Why a separate notebook?

We keep this example out of `src/ml_test/` deliberately:
- `src/` is a packaged Python module designed to run locally (DBC) or as a wheel (Asset Bundle job). It is not a notebook.
- `notebooks/nyctaxi_transform_demo.py` is a **Databricks-source `.py` file** (first line is `# Databricks notebook source`, cells separated by `# COMMAND ----------`). The Databricks runtime auto-injects `spark` and `display()` — the file would not run locally without modification.

Mixing the two would muddy the demo: each file should make its execution model obvious from the first line.

### What the notebook does (~30s)

1. Prints `Python`, `DATABRICKS_RUNTIME_VERSION`, `Spark version`, `defaultParallelism` — proves the code is running on the cluster
2. Reads `samples.nyctaxi.trips` and shows the schema + a sample
3. Filters bad records, derives `trip_duration_min` and `pickup_hour`
4. Aggregates trips / avg fare / avg distance / avg duration by pickup hour
5. Renders a matplotlib bar chart inline (matplotlib comes free with the ML Runtime — no local install)
6. Final markdown cell compares this workflow to the DBC workflow side by side

### How to run it from VS Code

1. Install the **Databricks Extension** in VS Code and sign in to the `DEFAULT` profile.
2. In the Databricks sidebar, attach an **all-purpose cluster** (DBR 15.x+ ML Runtime works fine). Serverless interactive compute also works if your workspace has it enabled.
3. Open `notebooks/nyctaxi_transform_demo.py`.
4. Click the ▶ icon in the editor title bar, or right-click → **"Run File on Databricks"** → **"Run File as Workflow on Databricks"**.
5. The extension uploads the file as a workspace file and triggers a one-off job. Output and `display()` results stream back into the VS Code output pane; click the run link to open the same view in the Databricks UI.

### What this demonstrates that the DBC workflow can't

- The cluster's runtime version and Python version, not your laptop's
- Direct access to ML Runtime libraries (matplotlib, etc.) without `uv add`
- The `spark` and `display()` symbols injected automatically — no Databricks Connect boilerplate
- Notebook-style outputs (rich `display()` tables, inline charts) instead of stdout

### When you'd choose this mode

- Quick exploration where you don't want to maintain a local virtualenv
- Code that depends on libraries already on the cluster but not locally
- Notebook-shaped work (inline plots, `display()`, markdown narrative) you'd later promote to a scheduled notebook job
- Onboarding teammates who have VS Code but haven't set up `uv` yet

---

## 6. Deploying with Asset Bundles

Asset Bundles package the project as a wheel and deploy it as a Databricks job.

### Validate the bundle config

```bash
databricks bundle validate
```

This checks that `databricks.yml` is syntactically correct and all references resolve.

### Deploy to the workspace

```bash
databricks bundle deploy
```

This will:
1. Build the `ml_test` wheel from the project (via hatchling)
2. Upload the wheel to the workspace
3. Create (or update) the job `[dev] train_job` with one task that calls the `train` entry point in serverless environment `default` (with `scikit-learn` and `pandas` added on top of the wheel).

### Run the pipeline

```bash
databricks bundle run train_job
```

### Monitor in the Databricks UI

1. Navigate to **Workflows** in the workspace sidebar
2. Find `[dev] train_job`
3. Click into the run to see task-level progress, logs, and stdout from `train.main()`

---

## 7. Demo Script (Live Presentation)

Total time: ~10 minutes

### Step 1 — Project Overview (2 min)

Show the project structure in VS Code. Explain the modular approach: `data_utils.py` is the I/O boundary, `train.py` is the entry point. Each file has a single, well-defined responsibility.

### Step 2 — Data Loading via Databricks Connect (2 min)

Open `src/ml_test/data_utils.py`. Highlight:
- `get_spark()` auto-detects local (Databricks Connect) vs cluster runtime
- `load_nyctaxi_pandas()` does **Spark-side** filtering and feature derivation, then materializes a small slice to pandas — the heavy lift stays remote, the model fits locally

### Step 3 — Train and Log (3 min)

Run locally:

```bash
uv run train
```

While it runs, show `train.py`: the `mlflow.set_tracking_uri("databricks")` line that wires logging to the workspace, the `mlflow.start_run()` block, and the params/metrics/model logging calls. After completion, click the printed run URL to show the experiment, run, and registered model artifact in the MLflow UI.

### Step 4 — Asset Bundle Configuration (1 min)

Open `databricks.yml`. Explain:
- The `artifacts` section builds the wheel with `uv build --wheel`
- The `resources.jobs.train_job` section defines a single task using `python_wheel_task` and the `train` entry point
- The `environments` section adds extra pip deps (`scikit-learn`, `pandas`) on top of the built wheel

Run validation:

```bash
databricks bundle validate
```

### Step 5 — Deploy and Run (1 min)

Deploy the bundle and trigger a run:

```bash
databricks bundle deploy
databricks bundle run train_job
```

Show the job in the Workflows UI. Point out that the same `train.main()` function ran here as on your laptop — only the execution environment changed.

### Step 6 — AI Coding Agent (1 min)

Open Claude Code in the project. Demonstrate asking it to:
- Swap `RandomForestRegressor` for `GradientBoostingRegressor` in `train.py`
- Re-run `uv run train` to verify the change works and produces a new MLflow run
- Show how the agent understands the modular structure and can make targeted changes

---

## 8. Coding Agent Integration

This modular structure is purpose-built for AI coding agents like Claude Code.

**Modular files are easy to navigate.** Each module is a single file with clear responsibilities. An agent can read `data_utils.py` and understand the full data pipeline without cross-referencing multiple notebooks.

**Clear entry points.** `train.py` has a `main()` function exposed via `[project.scripts]`. The agent can run it directly:

```bash
uv run train
```

**Declarative deployment.** Asset Bundles use YAML, which agents handle naturally. An agent can add a new task, change cluster settings, or modify job parameters in `databricks.yml`.

**Example prompts for Claude Code:**

- "Add a `pickup_day_of_week` feature in `data_utils.py` and re-run training"
- "Switch the model in `train.py` to XGBoost and log feature importances as a CSV artifact"
- "Add a second job task in `databricks.yml` that runs a validation step after `train`"
- "Register the trained model in Unity Catalog under `main.<schema>.nyctaxi_fare` and version it"

---

## 9. Key Differences from Notebook Approach

| Aspect | Notebooks | Modular Python |
|---|---|---|
| Local debugging | Limited (notebook debugger) | Full VS Code debugger with breakpoints |
| Code review | JSON diffs, hard to read | Standard file diffs in PRs |
| Testing | Manual cell execution | pytest, automated CI/CD |
| IDE support | Basic notebook editor | Full autocomplete, go-to-definition, refactoring |
| Deployment | Job notebooks or `%run` chains | Asset Bundles (declarative YAML) |
| AI agents | Notebook-scoped, cell-by-cell | Full project context, file-level edits |
| Dependency management | `%pip install` in cells | `pyproject.toml` with lockfile |
| Reusability | Copy-paste between notebooks | Import modules, share via wheel |
| Configuration | Widgets / hardcoded values | Centralized constants / env vars |
| Version control | Notebook JSON artifacts | Plain Python files |

The modular approach does not replace notebooks entirely — `notebooks/` is still useful for exploration and visualization. This project shows how to graduate from notebook prototypes to production-ready, maintainable Python code.
