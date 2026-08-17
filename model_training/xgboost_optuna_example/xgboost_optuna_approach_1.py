# Databricks notebook source
# MAGIC %md
# MAGIC ## Steps
# MAGIC
# MAGIC * Read parquet from a volume with Pandas (Synthetic Data)
# MAGIC * Downcast dtypes and keep train/test as in-memory numpy arrays on the driver
# MAGIC * Set up optuna with spark and xgboost for parameter tuning across 4 GPUs
# MAGIC
# MAGIC ### How the CPU memory design works
# MAGIC
# MAGIC The dataset stays **in memory** end-to-end and reaches the Spark tasks through the
# MAGIC pickled objective closure:
# MAGIC
# MAGIC * The 67 GB int64 dataframe is losslessly downcast to int8/int16 (~8x smaller), then
# MAGIC   materialized as numpy `X_train`/`X_test` that the `objective` function closes over.
# MAGIC * `MlflowSparkStudy.optimize()` cloudpickles the objective **including those arrays**.
# MAGIC   PySpark broadcasts large pickled commands to executors, so each of the 4 Spark
# MAGIC   python workers unpickles its **own full copy** of the train/test arrays.
# MAGIC * Net CPU footprint ≈ driver copies (`df` + `X` + splits) + the broadcast blob +
# MAGIC   4 worker copies — roughly 5–6x the downcast data size. This fits because the
# MAGIC   g5.24xlarge has 384 GB RAM; it is the part to redesign first if the data grows
# MAGIC   (see the streaming variant in `xgboost_optuna_with_spark.py`, which stages
# MAGIC   parquet on `/local_disk0` and keeps worker RAM ~flat).
# MAGIC * The upside of paying that one-time pickle/broadcast/unpickle cost: every trial
# MAGIC   builds its `QuantileDMatrix` straight from worker RAM with zero disk I/O.

# COMMAND ----------

import pandas as pd

df = pd.read_parquet(
    '/Volumes/fins_genai/classic_ml/data_examples/smaller_parquet',
    engine='pyarrow',
    memory_map=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Show in memory usage is 67GB

# COMMAND ----------

df.info(memory_usage='deep')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Downcast Data Type to Reduce Data Size to reduce CPU memory pressure

# COMMAND ----------

# DBTITLE 1,Prepare features and label
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Lossless downcast int64 → int8/int16 based on value range (saves ~8x CPU memory)
for col in df.columns:
    col_min, col_max = df[col].min(), df[col].max()
    if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
        df[col] = df[col].astype(np.int8)
    elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
        df[col] = df[col].astype(np.int16)
    elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
        df[col] = df[col].astype(np.int32)

print(f"Memory after downcast: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# All columns are numeric — pick one as the target
target_col = 'cat_01'
label_enc = LabelEncoder()
y = label_enc.fit_transform(df[target_col].values)
print(f"Classes: {label_enc.classes_}")

# Use remaining columns as features — keep native dtype (int8/int16)
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].values
print(f"Feature matrix shape: {X.shape}, dtype: {X.dtype}, size: {X.nbytes / 1e9:.2f} GB")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# COMMAND ----------

from optuna.pruners import BasePruner


class NoneValuePruner(BasePruner):
  """Custom Pruner to ignore failed trials with None value."""

  def prune(self, study, trial):
    # If the trial's value is None, prune it
    if trial.value is None:
      return True
      
    else:
      return False

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use QuantileDMatrix and DataIter to Reduce GPU memory pressure and Leverage Optuna Spark Study to parallel parameter tuning runs
# MAGIC
# MAGIC - [XGBoost Document for QuantileDMatrix data iterator](https://xgboost.readthedocs.io/en/latest/python/examples/quantile_data_iterator.html)
# MAGIC - [Optuna MlflowSparkStudy Document](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/optuna)
# MAGIC
# MAGIC How this fits 67 GB of training data into 24 GB of VRAM:
# MAGIC
# MAGIC * `NumpyBatchIter` feeds 10M-row numpy slices to `QuantileDMatrix`, which builds the
# MAGIC   quantized index (`GHistIndex`) batch-by-batch **on CPU** — the raw arrays are never
# MAGIC   copied to the GPU.
# MAGIC * At `xgb.train(device='cuda:N')` time, XGBoost converts that CPU index into the GPU
# MAGIC   `EllpackPage` format (the conversion path exists in `iterative_dmatrix.cu`), so only
# MAGIC   the compressed representation lands in VRAM: ~`n_rows × n_features × log2(max_bin)`
# MAGIC   **bits**, i.e. roughly 1 byte per value at `max_bin=128` instead of 8 — a few GB,
# MAGIC   well under 24 GB per GPU. Trade-off vs. feeding cupy batches: the quantile sketch
# MAGIC   runs on CPU, so 4 concurrent trials contend for cores (consider setting `nthread`).
# MAGIC * `MlflowSparkStudy.optimize()` runs as a single Spark job with `numPartitions=n_jobs`,
# MAGIC   so `TaskContext.partitionId()` is `0..n_jobs-1` and maps each concurrent trial to
# MAGIC   its own A10G.
# MAGIC * `MlflowSparkStudy` doesn't expose `direction`, and Optuna's default is **minimize** —
# MAGIC   which would make `best_value` the *worst* accuracy and steer TPE toward bad params.
# MAGIC   The study is therefore pre-created with `direction='maximize'` (and a fresh study
# MAGIC   name, so an old minimize-direction study isn't reloaded) before `MlflowSparkStudy`
# MAGIC   loads it by name.

# COMMAND ----------

# DBTITLE 1,Distributed Optuna + XGBoost GPU training
import mlflow
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score
from mlflow.pyspark.optuna.study import MlflowSparkStudy
from mlflow.optuna.storage import MlflowStorage


class NumpyBatchIter(xgb.DataIter):
    """Feed data to QuantileDMatrix in batches to avoid GPU OOM."""
    def __init__(self, X, y, batch_size=10_000_000):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self._it = 0
        super().__init__()

    def next(self, input_data):
        start = self._it * self.batch_size
        if start >= len(self.X):
            return 0
        end = min(start + self.batch_size, len(self.X))
        input_data(data=self.X[start:end], label=self.y[start:end])
        self._it += 1
        return 1

    def reset(self):
        self._it = 0


def objective(trial):
    """Optuna objective for XGBoost GPU training."""
    from pyspark import TaskContext

    # Assign each Spark partition to a different GPU (4× A10G available)
    ctx = TaskContext.get()
    gpu_id = ctx.partitionId() % 4 if ctx is not None else 0
    device = f'cuda:{gpu_id}'

    max_bin = trial.suggest_categorical('max_bin', [64, 128])

    # QuantileDMatrix streams data to GPU in batches — only the quantized
    # representation stays in VRAM (fits all 144M rows in 24 GB)
    # QuantileDMatrix accepts NumPy (CPU) data via DataIter: it builds the
    # quantile histogram in batches on CPU, then transfers only the compressed
    # representation (~max_bin indices/row) to GPU for training — far smaller
    # than raw data and avoids the ExtMemQuantileDMatrix CPU-data restriction
    # introduced in XGBoost 3.0.
    train_iter = NumpyBatchIter(X_train, y_train)
    dtrain = xgb.QuantileDMatrix(train_iter, max_bin=max_bin)

    test_iter = NumpyBatchIter(X_test, y_test)
    dtest = xgb.QuantileDMatrix(test_iter, max_bin=max_bin, ref=dtrain)

    params = {
        'device': device,
        'tree_method': 'hist',
        'objective': 'multi:softmax',
        'num_class': len(label_enc.classes_),
        'eval_metric': 'mlogloss',
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'max_bin': max_bin,
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
    }

    n_rounds = trial.suggest_int('n_rounds', 100, 500)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_rounds,
        evals=[(dtest, 'eval')],
        verbose_eval=False,
    )

    preds = model.predict(dtest)
    accuracy = accuracy_score(y_test, preds)
    return accuracy


# Distributed Optuna via Spark + MLflow logging
mlflow.set_experiment('/Users/q.yu@databricks.com/xgb_gpu_oom_debug')

experiment = mlflow.get_experiment_by_name('/Users/q.yu@databricks.com/xgb_gpu_oom_debug')
storage = MlflowStorage(experiment_id=experiment.experiment_id)

# MlflowSparkStudy doesn't expose `direction` and would create the study with the
# Optuna default (minimize) — best_value would then be the WORST accuracy and TPE
# would steer toward bad params. Create the study with direction='maximize' first;
# MlflowSparkStudy loads an existing study by name. Use a NEW study name so the old
# minimize-direction study from previous runs isn't reloaded.
study_name = 'xgb_gpu_oom_debug_v2'
optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction='maximize',
    load_if_exists=True,
)

study = MlflowSparkStudy(
    pruner= NoneValuePruner(),
    study_name=study_name,
    sampler=TPESampler(seed=42),
    storage=storage,
)

# n_jobs=4: one trial per GPU (4× A10G), each with its own 24 GB VRAM
study.optimize(objective, n_trials=10, n_jobs=4)

print(f"\nBest accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# COMMAND ----------

# DBTITLE 1,Train final model with best params
# Train final model with best hyperparameters
best = study.best_params
n_rounds = best.pop('n_rounds')

train_iter = NumpyBatchIter(X_train, y_train)
dtrain = xgb.QuantileDMatrix(train_iter, max_bin=best['max_bin'])
test_iter = NumpyBatchIter(X_test, y_test)
dtest = xgb.QuantileDMatrix(test_iter, max_bin=best['max_bin'], ref=dtrain)

final_params = {
    'device': 'cuda',
    'tree_method': 'hist',
    'objective': 'multi:softmax',
    'num_class': len(label_enc.classes_),
    'eval_metric': 'mlogloss',
    **best
}

final_model = xgb.train(
    final_params,
    dtrain,
    num_boost_round=n_rounds,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    verbose_eval=50,
)

# Final evaluation
preds = final_model.predict(dtest)
print(f"\nFinal test accuracy: {accuracy_score(y_test, preds):.4f}")

# COMMAND ----------

