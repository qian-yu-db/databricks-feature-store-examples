# Databricks notebook source
# DBTITLE 1,Cell 1
# MAGIC %md
# MAGIC # Modular UC UDF Custom Dependencies — Test Notebook
# MAGIC
# MAGIC This notebook demonstrates the **Python-driven, modular pattern** for defining shared
# MAGIC custom dependencies across multiple Unity Catalog Python UDFs.
# MAGIC
# MAGIC **Pattern:**
# MAGIC - UDF function bodies defined in a separate **`udf_functions.py`** file (same directory)
# MAGIC - Shared dependencies listed in a **`requirements.txt`** file (same directory)
# MAGIC - The notebook imports the module, extracts function bodies via `inspect`, and registers all UDFs
# MAGIC - All UDFs share the exact same environment → environment is installed once and reused
# MAGIC
# MAGIC **File layout** (upload all three to the same workspace folder):
# MAGIC ```
# MAGIC uc_python_udf_custom_deps/
# MAGIC ├── modular_udf_dependency_test    ← this notebook
# MAGIC ├── udf_functions.py               ← UDF function definitions
# MAGIC └── requirements.txt               ← shared pip dependencies
# MAGIC ```
# MAGIC
# MAGIC **Requirements:**
# MAGIC - DBR 16.2+ (classic) OR Serverless notebooks/jobs OR Pro/Serverless SQL Warehouse
# MAGIC - Unity Catalog enabled workspace
# MAGIC - `unitycatalog-ai[databricks]` package (installed in Cell 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install SDK

# COMMAND ----------

# DBTITLE 1,Cell 3
# Install unitycatalog-ai + UDF dependencies from requirements.txt (single source of truth).
# EDIT the path below to the workspace folder where you uploaded this notebook,
# udf_functions.py, and requirements.txt (all three live together). The `%pip`
# magic runs before any Python, so this path must be a literal.
%pip install -q unitycatalog-ai[databricks] -r /Workspace/Users/<your-username>/uc_python_udf_custom_deps/requirements.txt
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cell 4
# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Edit these values before running the notebook. All supporting files
# MAGIC (`udf_functions.py`, `requirements.txt`) are expected in `NOTEBOOK_DIR`.

# COMMAND ----------

# DBTITLE 1,Cell 5
# ============================================================
# CONFIGURATION — Edit these before running
# ============================================================

import os

CATALOG = "fins_genai"           # UC catalog to register UDFs in — EDIT
SCHEMA  = "classic_ml"           # UC schema to register UDFs in  — EDIT

# All supporting files (udf_functions.py, requirements.txt) live alongside this
# notebook in the same workspace directory. Auto-derive that directory so no
# path editing is needed here; hardcode it instead if this introspection is
# unavailable in your environment.
NOTEBOOK_DIR = "/Workspace" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)

REQUIREMENTS_PATH = f"{NOTEBOOK_DIR}/requirements.txt"
UDF_MODULE_PATH   = f"{NOTEBOOK_DIR}/udf_functions.py"

# Serverless environment version. Use '5' for serverless/pro warehouses,
# 'None' for classic clusters.
ENV_VERSION = "5"

print(f"Target:        {CATALOG}.{SCHEMA}")
print(f"Env version:   {ENV_VERSION}")
print(f"Notebook dir:  {NOTEBOOK_DIR}")
print(f"Requirements:  {REQUIREMENTS_PATH}")
print(f"UDF module:    {UDF_MODULE_PATH}")

# COMMAND ----------

# DBTITLE 1,Cell 6
# MAGIC %md
# MAGIC ## Load Shared Dependencies
# MAGIC
# MAGIC Reads `requirements.txt` from the workspace directory alongside this notebook.

# COMMAND ----------

# DBTITLE 1,Cell 7
import json
import os


def load_requirements(path: str) -> list:
    """Read a requirements.txt file and return a list of dependency strings."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"requirements.txt not found at: {path}\n"
            "Make sure the file exists in the notebook directory."
        )
    deps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                deps.append(line)
    print(f"Loaded {len(deps)} dependencies from: {path}")
    print(f"  Dependencies: {deps}")
    return deps


# Load shared dependencies from workspace requirements.txt
SHARED_DEPS      = load_requirements(REQUIREMENTS_PATH)
SHARED_DEPS_JSON = json.dumps(SHARED_DEPS)

print(f"\nDependency JSON for ENVIRONMENT clause:\n{SHARED_DEPS_JSON}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: `udf_dump_json`

# COMMAND ----------

# DBTITLE 1,Cell 15
# Test via direct Python call (bypasses UC Python UDF sandbox)
from udf_functions import udf_dump_json

test_inputs = ["Hello Databricks!", "Spark is great", "", None]
rows = [{"input": v, "output": udf_dump_json(v)} for v in test_inputs]
display(spark.createDataFrame(rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: `udf_sha3_hash`

# COMMAND ----------

# DBTITLE 1,Cell 17
# Test via direct Python call (bypasses UC Python UDF sandbox)
from udf_functions import udf_sha3_hash

test_inputs = ["Hello Databricks!", "Spark is great", None]
rows = [{"input": v, "output": udf_sha3_hash(v)} for v in test_inputs]
display(spark.createDataFrame(rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: `udf_mask_email`

# COMMAND ----------

# DBTITLE 1,Cell 19
# Test via direct Python call (bypasses UC Python UDF sandbox)
from udf_functions import udf_mask_email

test_inputs = ["john.doe@example.com", "alice@databricks.com", "x@test.org", None]
rows = [{"email_raw": v, "email_masked": udf_mask_email(v)} for v in test_inputs]
display(spark.createDataFrame(rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Test on a DataFrame

# COMMAND ----------

# DBTITLE 1,Cell 21
from udf_functions import udf_mask_email, udf_dump_json, udf_sha3_hash

data = [
    ("john.doe@example.com",  "Hello World"),
    ("alice@databricks.com",  "Spark is great"),
    ("bob@company.org",       "Unity Catalog"),
]

# Build results using direct Python calls
rows = [
    {
        "email": email,
        "message": msg,
        "masked_email": udf_mask_email(email),
        "json_output": udf_dump_json(msg),
        "sha3_hash": udf_sha3_hash(msg),
    }
    for email, msg in data
]
display(spark.createDataFrame(rows))

# COMMAND ----------

# DBTITLE 1,Cell 8
# MAGIC %md
# MAGIC ## Build UDF Registry from `udf_functions.py`
# MAGIC
# MAGIC Imports the external module, extracts function bodies using `inspect.getsource()`,
# MAGIC and builds the `UDF_REGISTRY` list used for SQL registration.
# MAGIC
# MAGIC **All UDFs automatically inherit `SHARED_DEPS`.**

# COMMAND ----------

# DBTITLE 1,Cell 9
import sys
import inspect
import textwrap

# Add notebook directory to Python path for imports
if NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOK_DIR)

# Import (or reload) the UDF definitions module
import importlib
try:
    import udf_functions
    importlib.reload(udf_functions)
except ImportError:
    import udf_functions


def get_function_body(func) -> str:
    """Extract the body of a function (without the def line), dedented."""
    source = inspect.getsource(func)
    lines = source.splitlines()
    # Skip the def line (first line)
    body_lines = lines[1:]
    return textwrap.dedent("\n".join(body_lines))


# Build UDF_REGISTRY from the external module's definitions
UDF_REGISTRY = []
for defn in udf_functions.UDF_DEFINITIONS:
    func = defn["func"]
    UDF_REGISTRY.append({
        "name":    f"{CATALOG}.{SCHEMA}.{func.__name__}",
        "params":  defn["params"],
        "returns": defn["returns"],
        "comment": defn["comment"],
        "body":    get_function_body(func),
    })

print(f"UDF registry loaded from udf_functions.py: {len(UDF_REGISTRY)} UDFs")
for u in UDF_REGISTRY:
    print(f"  → {u['name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register All UDFs
# MAGIC
# MAGIC Generates and executes `CREATE OR REPLACE FUNCTION` for each UDF
# MAGIC using the **shared dependency JSON** and the configured environment version.

# COMMAND ----------

# DBTITLE 1,Cell 11
def build_create_function_sql(udf: dict, deps_json: str, env_version: str) -> str:
    """Build a CREATE OR REPLACE FUNCTION SQL statement from a UDF registry entry."""
    # Extract short function name (last dotted segment)
    func_short_name = udf["name"].split(".")[-1]

    # Extract only parameter names, stripping SQL type annotations
    param_names = ", ".join(
        p.strip().split()[0] for p in udf["params"].split(",")
    ) if udf["params"].strip() else ""

    # Indent body 4 spaces to sit inside the def block
    indented_body = "\n".join("    " + line for line in udf["body"].rstrip().splitlines())
    func_def = f"def {func_short_name}({param_names}):\n{indented_body}"

    return f"""CREATE OR REPLACE FUNCTION {udf['name']}({udf['params']})
RETURNS {udf['returns']}
COMMENT "{udf.get('comment', '')}"
LANGUAGE PYTHON
ENVIRONMENT (
  dependencies = '{deps_json}',
  environment_version = '{env_version}'
)
AS {"$" + "$"}
{func_def}
{"$" + "$"}"""


def register_all_udfs(registry: list, deps_json: str, env_version: str, dry_run: bool = False):
    """
    Register all UDFs in the registry.

    Args:
        registry:    List of UDF definition dicts.
        deps_json:   JSON string of dependencies.
        env_version: ENVIRONMENT version string ('None', '3', etc.).
        dry_run:     If True, print SQL without executing.
    """
    results = []
    for udf in registry:
        sql = build_create_function_sql(udf, deps_json, env_version)
        print(f"\n{'='*60}")
        print(f"Registering: {udf['name']}")
        print(f"{'='*60}")
        if dry_run:
            print("[DRY RUN] SQL:\n")
            print(sql)
            results.append({"name": udf["name"], "status": "dry_run"})
        else:
            try:
                spark.sql(sql)
                print(f"Success: {udf['name']}")
                results.append({"name": udf["name"], "status": "success"})
            except Exception as e:
                print(f"Failed:  {udf['name']}\n   Error: {e}")
                results.append({"name": udf["name"], "status": "failed", "error": str(e)})
    return results


# ---- Preview the SQL first (dry run) ----
print(">>> DRY RUN — Previewing generated SQL <<<\n")
register_all_udfs(UDF_REGISTRY, SHARED_DEPS_JSON, ENV_VERSION, dry_run=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Registration
# MAGIC
# MAGIC Run this cell to actually register the UDFs in Unity Catalog.

# COMMAND ----------

print(">>> REGISTERING UDFs in Unity Catalog <<<\n")
results = register_all_udfs(UDF_REGISTRY, SHARED_DEPS_JSON, ENV_VERSION, dry_run=False)

print("\n\n--- Registration Summary ---")
for r in results:
    status_icon = "✅" if r["status"] == "success" else "❌"
    print(f"  {status_icon}  {r['name']}  [{r['status']}]")

# COMMAND ----------

# DBTITLE 1,Cell 22
# MAGIC %md
# MAGIC ## (Optional) Inspect Loaded Files
# MAGIC
# MAGIC Verify the contents of `udf_functions.py` and `requirements.txt` from the workspace directory.

# COMMAND ----------

# DBTITLE 1,Cell 23
print("=" * 60)
print(f"requirements.txt  ({REQUIREMENTS_PATH})")
print("=" * 60)
with open(REQUIREMENTS_PATH) as f:
    print(f.read())

print("\n" + "=" * 60)
print(f"udf_functions.py  ({UDF_MODULE_PATH})")
print("=" * 60)
with open(UDF_MODULE_PATH) as f:
    print(f.read())

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Optional) Clean Up Test UDFs

# COMMAND ----------

# Uncomment and run to drop the test UDFs
# for udf in UDF_REGISTRY:
#     spark.sql(f"DROP FUNCTION IF EXISTS {udf['name']}")
#     print(f"Dropped: {udf['name']}")

print("Cleanup skipped. Uncomment the block above to drop test UDFs.")