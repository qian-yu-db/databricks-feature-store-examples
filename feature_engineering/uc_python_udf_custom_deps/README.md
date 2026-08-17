# UC Python UDFs with Custom Dependencies

Demonstrates a **modular, Python-driven pattern** for registering
[Unity Catalog Python UDFs](https://docs.databricks.com/aws/en/udf/unity-catalog-python)
that carry **custom pip dependencies** via the `ENVIRONMENT` clause of
`CREATE OR REPLACE FUNCTION`.

Rather than embedding UDF bodies and dependency lists inline in SQL, this example
keeps them in ordinary Python/text files that are the single source of truth:

- **`udf_functions.py`** — UDF bodies as normal Python functions, plus a
  `UDF_DEFINITIONS` registry mapping each function to its SQL signature.
- **`requirements.txt`** — shared pip dependencies installed once and reused by
  every UDF.
- **`modular_udf_dependency_test.py`** — the notebook: it imports the module,
  extracts each function body with `inspect.getsource()`, and generates +
  executes a `CREATE OR REPLACE FUNCTION … ENVIRONMENT (…)` statement per UDF.

Because all UDFs share the same `requirements.txt`, the environment is installed
once and reused across them.

> This example is about **UDF authoring and dependency packaging**, not about
> on-demand features bound to a model. For the latter — computing derived features
> with `FeatureFunction` at train and serve time — see
> [`../on_demand_feature/`](../on_demand_feature).

## The example UDFs

These are intentionally generic (not tied to any ML feature) so the focus stays
on the dependency mechanism:

| Function | Dependency | Behavior |
|----------|------------|----------|
| `udf_dump_json` | `simplejson` | Serializes a string to a JSON object. |
| `udf_sha3_hash` | `pycryptodome` | Returns the SHA3-256 hex digest of the input. |
| `udf_mask_email` | none | Masks the local part of an email address. |

## Requirements

- DBR 16.2+ (classic) **or** Serverless notebooks/jobs **or** Pro/Serverless SQL Warehouse
- Unity Catalog enabled workspace
- `unitycatalog-ai[databricks]` (installed by the notebook)

## How to run

1. Upload all three files (`modular_udf_dependency_test.py`, `udf_functions.py`,
   `requirements.txt`) into the **same workspace folder**.
2. Open `modular_udf_dependency_test.py` and edit the two config spots:
   - The `%pip install … -r /Workspace/Users/<your-username>/uc_python_udf_custom_deps/requirements.txt`
     path in the install cell (the `%pip` magic runs before Python, so it must be a
     literal path — point it at your uploaded `requirements.txt`).
   - `CATALOG` and `SCHEMA` in the configuration cell.

   `NOTEBOOK_DIR` is auto-derived from the notebook's own location, so no editing
   is needed there; hardcode it if that introspection is unavailable in your
   environment.
3. Run the cells top to bottom. The notebook first tests each UDF via direct
   Python calls, then does a **dry run** printing the generated SQL, then executes
   the registration in Unity Catalog. A commented cleanup cell at the end drops
   the test UDFs.

## Adding your own UDF

1. Define the function in `udf_functions.py` (the function name becomes the UC
   function name).
2. Add an entry to `UDF_DEFINITIONS` with its SQL `params`, `returns`, and a
   `comment`.
3. Add any new pip dependencies to `requirements.txt`.
4. Re-run the notebook — the new UDF is picked up automatically.

## References

- [Unity Catalog Python UDFs](https://docs.databricks.com/aws/en/udf/unity-catalog-python)
- [`CREATE FUNCTION` (SQL) — `ENVIRONMENT` clause](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function)
- [`unitycatalog-ai`](https://github.com/unitycatalog/unitycatalog)
