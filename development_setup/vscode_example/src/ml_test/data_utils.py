import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split

try:
    from databricks.connect import DatabricksSession
except ImportError:
    DatabricksSession = None


def get_spark():
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        return SparkSession.builder.getOrCreate()
    if DatabricksSession is None:
        raise RuntimeError("databricks-connect is not installed in this environment")
    return DatabricksSession.builder.getOrCreate()


def load_nyctaxi_pandas(
    limit: int = 20000,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """Pull a sample of `samples.nyctaxi.trips` from the workspace via Databricks
    Connect, do light feature engineering on the Spark side, and return train/test
    pandas splits suitable for scikit-learn.

    Target: fare_amount (regression).
    """
    spark = get_spark()

    df = (
        spark.table("samples.nyctaxi.trips")
        .where(F.col("fare_amount").isNotNull())
        .where(F.col("trip_distance") > 0)
        .where(F.col("fare_amount").between(2.5, 200))
        .withColumn("trip_duration_min",
                    (F.unix_timestamp("tpep_dropoff_datetime")
                     - F.unix_timestamp("tpep_pickup_datetime")) / 60.0)
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .where(F.col("trip_duration_min").between(0.5, 180))
        .select(
            "trip_distance",
            "trip_duration_min",
            "pickup_hour",
            "pickup_zip",
            "dropoff_zip",
            "fare_amount",
        )
        .limit(limit)
    )

    pdf = df.toPandas()

    feature_cols = [
        "trip_distance",
        "trip_duration_min",
        "pickup_hour",
        "pickup_zip",
        "dropoff_zip",
    ]
    X = pdf[feature_cols]
    y = pdf["fare_amount"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
