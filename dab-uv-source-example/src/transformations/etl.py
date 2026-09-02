# ETL Pipeline — source-deployed via DAB (no wheel)
#
# Dependencies from pyproject.toml are available because the pipeline
# environment installs them with `--editable ${workspace.file_path}`.

import dlt
from pyspark.sql import functions as F

try:
    from importlib.metadata import version
    pkg_version = version("dab-uv-source-example")
except Exception:
    pkg_version = "unknown"

print(f"Package version (from pyproject.toml): {pkg_version}")


@dlt.table(
    name="raw_events",
    comment="Example raw events table",
)
def raw_events():
    """Read raw event data — replace with your actual source."""
    return spark.read.format("json").load("/databricks-datasets/structured-streaming/events")


@dlt.table(
    name="cleaned_events",
    comment="Cleaned events with numpy/pandas transformations",
)
def cleaned_events():
    """Demonstrate that pyproject.toml deps are available."""
    df = dlt.read("raw_events")
    return df.select("*", F.current_timestamp().alias("processed_at"))
