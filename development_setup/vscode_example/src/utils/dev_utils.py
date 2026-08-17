import os


def is_running_in_databricks():
    """Detect if code is running in Databricks workspace."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ
