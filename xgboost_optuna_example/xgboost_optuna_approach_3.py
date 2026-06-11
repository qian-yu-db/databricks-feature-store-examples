# Databricks notebook source
# MAGIC %md
# MAGIC ## XGBoost Spark
# MAGIC
# MAGIC Optuna HPO where each trial trains one distributed `SparkXGBClassifier` across all
# MAGIC 4 GPUs (`num_workers=4`), and trials run **sequentially** on the driver.
# MAGIC
# MAGIC ### Why xgboost.spark, compared to the single-process variants
# MAGIC
# MAGIC (See `xgboost_optuna_with_spark.py` / `..._alternative.py` for the opposite design:
# MAGIC 4 **concurrent** single-GPU trials via `MlflowSparkStudy`.)
# MAGIC
# MAGIC * **No 67 GB pandas problem.** The data stays in Spark end-to-end — no driver-side
# MAGIC   pandas load, no dtype downcasting, no shipping arrays to workers through the
# MAGIC   pickled objective closure, and no staging copies on `/local_disk0`. Spark
# MAGIC   partitions the data and `xgboost.spark` feeds each GPU its shard.
# MAGIC * **Per-GPU memory is solved by sharding, not compression alone.** Each of the 4
# MAGIC   workers only holds ~1/4 of the rows (plus the internal QuantileDMatrix
# MAGIC   compression), so the 24 GB A10G limit is much harder to hit — and the same code
# MAGIC   scales to more data by adding nodes/GPUs, with no notebook changes.
# MAGIC * **No manual GPU assignment.** Spark's scheduler hands each barrier task its GPU —
# MAGIC   no `TaskContext.partitionId() % 4` mapping, no risk of two trials colliding on
# MAGIC   one device, no leftover-VRAM sensitivity from other notebooks on the cluster.
# MAGIC * **Simpler, safer Optuna setup.** Trials run in the driver process, so a plain
# MAGIC   `optuna.create_study(direction="maximize")` works — no `MlflowSparkStudy`, and
# MAGIC   none of its hidden default-minimize direction issue. Early stopping via
# MAGIC   `validation_indicator_col` also works out of the box.
# MAGIC
# MAGIC ### Trade-offs
# MAGIC
# MAGIC * Trials are sequential (1 trial × 4 GPUs) vs. 4 concurrent trials × 1 GPU in the
# MAGIC   other notebooks. Distributed training adds per-trial coordination overhead, so
# MAGIC   for pure HPO throughput on data that *fits* one GPU, 4 parallel single-GPU
# MAGIC   trials usually finish the study faster. Prefer this version when one trial's
# MAGIC   data can't fit a single GPU or when scaling beyond one node.
# MAGIC * Each trial re-runs the read → index → assemble lineage (nothing is cached), and
# MAGIC   `tree_method="hist"` on sharded data approximates split finding per worker —
# MAGIC   results can differ slightly from single-GPU training.

# COMMAND ----------

# DBTITLE 1,Cell 2
import optuna
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when, rand
import mlflow

# ── Re-read data & rebuild preprocessing (Python was restarted by %pip) ──────
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col

# COMMAND ----------

spark.conf.set("spark.task.resource.gpu.amount", "1")
spark.conf.set("spark.executor.resource.gpu.amount", "4")

spark_df = spark.read.parquet('/Volumes/fins_genai/classic_ml/data_examples/smaller_parquet')

# COMMAND ----------

# DBTITLE 1,Optuna HPO for SparkXGBClassifier
# All columns are already numeric (Long). Use cat_01 as label, rest as features.
label_col = "cat_01"
exclude_cols = [label_col, "_rescued_data"]
feature_cols = [c for c in spark_df.columns if c not in exclude_cols]

# StringIndexer ensures label is contiguous 0-based (required by XGBoost multiclass)
label_indexer = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="keep")
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

prep_pipeline = Pipeline(stages=[label_indexer, assembler])
prep_model = prep_pipeline.fit(spark_df)
prepared_df = prep_model.transform(spark_df)

num_classes = prepared_df.select("label").distinct().count()
train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)

# ── Add validation indicator column for pruning callback ─────────────────────
# 20% of training data used as validation for early-stopping / pruning
train_with_val = train_df.withColumn(
    "is_val", when(rand(seed=123) < 0.2, True).otherwise(False)
)
print(f"Num classes: {num_classes}")

# ── Optuna objective ─────────────────────────────────────────────────────────
mlflow.set_experiment('/Users/q.yu@databricks.com/xgb_gpu_oom_debug')

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
    }

    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        num_workers=4,
        device="cuda",
        tree_method="hist",
        num_class=num_classes,
        eval_metric="mlogloss",
        validation_indicator_col="is_val",
        early_stopping_rounds=10,
        max_bin=128,
        verbosity=0,
        **params,
    )

    with mlflow.start_run(nested=True):
        model = xgb.fit(train_with_val)
        predictions = model.transform(test_df)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        mlflow.log_params(params)
        mlflow.log_metric("test_accuracy", accuracy)

    return accuracy

# ── Run study ────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="optuna_xgb_spark_hpo"):
    study = optuna.create_study(direction="maximize", study_name="xgb_spark_gpu_tuning")
    study.optimize(objective, n_trials=10, show_progress_bar=True)

print(f"\n{'='*60}")
print(f"Best trial accuracy: {study.best_value:.4f}")
print(f"Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# DBTITLE 1,Retrain best model and evaluate
# Train final model with Optuna's best hyperparameters
best = study.best_params

xgb_best = SparkXGBClassifier(
    features_col="features",
    label_col="label",
    num_workers=4,
    device="cuda",
    tree_method="hist",
    num_class=num_classes,
    eval_metric="mlogloss",
    verbosity=1,
    **best,
)

with mlflow.start_run(run_name="optuna_best_model"):
    best_model = xgb_best.fit(train_df)
    mlflow.log_params(best)

    predictions = best_model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    mlflow.log_metric("test_accuracy", accuracy)
    print(f"\nFinal model test accuracy (best Optuna params): {accuracy:.4f}")

display(predictions.select("cat_01", "label", "prediction", "probability").limit(20))