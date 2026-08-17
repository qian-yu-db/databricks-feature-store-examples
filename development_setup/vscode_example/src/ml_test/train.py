import os

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from ml_test.data_utils import load_nyctaxi_pandas


EXPERIMENT_PATH = "/Users/q.yu@databricks.com/ml_test_nyctaxi"


def main():
    # Log to the Databricks workspace MLflow tracking server
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ:

        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")

    mlflow.set_experiment(EXPERIMENT_PATH)

    X_train, X_test, y_train, y_test = load_nyctaxi_pandas(limit=20000)

    params = {"n_estimators": 100, "max_depth": 8, "random_state": 42, "n_jobs": -1}

    with mlflow.start_run(run_name="rf_nyctaxi_fare") as run:
        mlflow.log_params(params)
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows", len(X_test))

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(5),
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"MAE: {mae:.4f} | R2: {r2:.4f}")
        print(f"Experiment: {EXPERIMENT_PATH}")


if __name__ == "__main__":
    main()
