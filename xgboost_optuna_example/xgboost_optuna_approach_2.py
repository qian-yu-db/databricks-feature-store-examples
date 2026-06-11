# Databricks notebook source
# MAGIC %pip install cupy-cuda12x
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Steps
# MAGIC
# MAGIC * Read parquet from a volume with Pandas (Synthetic Data)
# MAGIC * Downcast dtypes, split train/test, and stage the data on local NVMe (`/local_disk0`)
# MAGIC * Set up Optuna (`MlflowSparkStudy`) with XGBoost for parameter tuning across 4 GPUs
# MAGIC
# MAGIC ### Why the restage to /local_disk0?
# MAGIC
# MAGIC `MlflowSparkStudy.optimize()` ships the objective function to Spark executors with
# MAGIC cloudpickle. Anything the objective closes over gets serialized into the Spark task —
# MAGIC if the objective references in-memory numpy arrays (`X_train`, `y_train`, ...), the
# MAGIC whole ~17–67 GB dataset is pickled into the task and the job dies on serialization. 
# MAGIC
# MAGIC Approach: the driver writes the downcast train/test sets to local NVMe once, and each
# MAGIC Spark task **streams the parquet row groups from disk** inside the objective. The
# MAGIC objective then only captures small things (paths, column names, an int).
# MAGIC This works on a single-node cluster because driver and executors share `/local_disk0`.

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

# DBTITLE 1,Prepare features and label, stage train/test on local NVMe
import gc
import os

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

# All columns are numeric — pick one as the target and encode it to 0..k-1
target_col = 'cat_01'
label_enc = LabelEncoder()
df[target_col] = label_enc.fit_transform(df[target_col]).astype(np.int32)
num_class = len(label_enc.classes_)
print(f"Classes: {label_enc.classes_}")

feature_cols = [c for c in df.columns if c != target_col]

# Split the dataframe itself — do NOT materialize X = df.values: that upcasts
# every column to the widest dtype and doubles CPU memory for nothing.
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[target_col]
)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")
del df
gc.collect()

# Stage on local NVMe so Spark tasks stream from disk instead of receiving the
# data through the pickled objective closure. row_group_size bounds how much one
# DataIter batch loads into memory.
DATA_DIR = '/local_disk0/xgb_optuna_cache'
TRAIN_PATH = f'{DATA_DIR}/train.parquet'
TEST_PATH = f'{DATA_DIR}/test.parquet'
os.makedirs(DATA_DIR, exist_ok=True)

train_df.to_parquet(TRAIN_PATH, engine='pyarrow', index=False, row_group_size=1_000_000)
test_df.to_parquet(TEST_PATH, engine='pyarrow', index=False, row_group_size=1_000_000)
print(f"Staged: {os.path.getsize(TRAIN_PATH) / 1e9:.2f} GB train, "
      f"{os.path.getsize(TEST_PATH) / 1e9:.2f} GB test")

# Free driver RAM before the Spark tasks spin up
del train_df, test_df
gc.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use QuantileDMatrix and DataIter to Reduce GPU memory pressure and Leverage Optuna Spark Study to parallel parameter tuning runs
# MAGIC
# MAGIC - [XGBoost Document for QuantileDMatrix data iterator](https://xgboost.readthedocs.io/en/latest/python/examples/quantile_data_iterator.html)
# MAGIC - [Optuna MlflowSparkStudy Document](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/optuna)
# MAGIC
# MAGIC How this fits 67 GB of training data into 24 GB of VRAM:
# MAGIC
# MAGIC * The `DataIter` reads one parquet row group at a time and moves **only that batch**
# MAGIC   to the GPU as a cupy array. XGBoost's GPU quantile sketch requires on-device input —
# MAGIC   it cannot consume CPU numpy data ("GPU cannot consume CPU input data and vice-versa",
# MAGIC   per the XGBoost external-memory docs) — so the cupy conversion is mandatory, not an optimization.
# MAGIC * `QuantileDMatrix` keeps only the quantized representation in VRAM:
# MAGIC   ~`n_rows × n_features × log2(max_bin)` **bits**, i.e. roughly 1 byte per value at
# MAGIC   `max_bin=128` instead of 8 — a few GB for this dataset, well under 24 GB per GPU.
# MAGIC * `MlflowSparkStudy.optimize()` runs as a single Spark job with `numPartitions=n_jobs`,
# MAGIC   so `TaskContext.partitionId()` is `0..n_jobs-1` and maps each concurrent worker to
# MAGIC   its own A10G.

# COMMAND ----------

# DBTITLE 1,Streaming parquet → GPU DataIter
import cupy as cp  # preinstalled on Databricks ML GPU runtime
import pyarrow.parquet as pq
import xgboost as xgb


class ParquetIter(xgb.DataIter):
    """Stream parquet row groups to one GPU, a batch at a time.

    Peak VRAM during QuantileDMatrix construction is one raw batch plus the
    growing quantized matrix — never the full dataset.
    """

    def __init__(self, path, feature_cols, target_col, gpu_id, batch_rows=1_000_000):
        self.path = path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.gpu_id = gpu_id
        self.batch_rows = batch_rows
        self._batches = None
        super().__init__()

    def reset(self):
        self._batches = pq.ParquetFile(self.path).iter_batches(
            batch_size=self.batch_rows,
            columns=self.feature_cols + [self.target_col],
        )

    def next(self, input_data):
        if self._batches is None:
            self.reset()
        batch = next(self._batches, None)
        if batch is None:
            return 0
        pdf = batch.to_pandas()
        with cp.cuda.Device(self.gpu_id):
            X = cp.asarray(pdf[self.feature_cols].to_numpy()).astype(cp.float32)
            y = cp.asarray(pdf[self.target_col].to_numpy())
        input_data(data=X, label=y)
        return 1

# COMMAND ----------

# DBTITLE 1,Distributed Optuna + XGBoost GPU training
import mlflow
import optuna
from optuna.samplers import TPESampler
from mlflow.pyspark.optuna.study import MlflowSparkStudy
from mlflow.optuna.storage import MlflowStorage

N_GPUS = 4


def objective(trial):
    """Optuna objective for XGBoost GPU training.

    Only closes over small objects (paths, column names, num_class) — the data
    is streamed from /local_disk0 inside the Spark task.
    """
    from pyspark import TaskContext

    # MlflowSparkStudy runs optimize() as one Spark job with numPartitions=n_jobs,
    # so partitionId is 0..n_jobs-1 → one GPU per concurrent task.
    ctx = TaskContext.get()
    gpu_id = ctx.partitionId() % N_GPUS if ctx is not None else 0
    device = f'cuda:{gpu_id}'

    max_bin = trial.suggest_categorical('max_bin', [64, 128])

    train_iter = ParquetIter(TRAIN_PATH, feature_cols, target_col, gpu_id)
    dtrain = xgb.QuantileDMatrix(train_iter, max_bin=max_bin)

    # ref=dtrain so the test set reuses the training bin edges
    test_iter = ParquetIter(TEST_PATH, feature_cols, target_col, gpu_id)
    dtest = xgb.QuantileDMatrix(test_iter, max_bin=max_bin, ref=dtrain)

    params = {
        'device': device,
        'tree_method': 'hist',
        'objective': 'multi:softmax',
        'num_class': num_class,
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
    y_test = pq.read_table(TEST_PATH, columns=[target_col]).to_pandas()[target_col].to_numpy()
    accuracy = float((preds == y_test).mean())

    # Each partition runs several trials in the same worker process — release
    # this trial's VRAM before the next one starts.
    del model, dtrain, dtest, train_iter, test_iter
    gc.collect()
    with cp.cuda.Device(gpu_id):
        cp.get_default_memory_pool().free_all_blocks()

    return accuracy


# Distributed Optuna via Spark + MLflow logging
EXPERIMENT_PATH = '/Users/q.yu@databricks.com/xgb_gpu_oom_debug'
STUDY_NAME = 'xgb_gpu_qdm_v1'

mlflow.set_experiment(EXPERIMENT_PATH)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
storage = MlflowStorage(experiment_id=experiment.experiment_id)

# MlflowSparkStudy doesn't expose `direction` and would create the study with the
# Optuna default (minimize) — best_value would then be the WORST accuracy. Create
# the study with direction='maximize' first; MlflowSparkStudy loads the existing one.
optuna.create_study(
    study_name=STUDY_NAME,
    storage=storage,
    direction='maximize',
    load_if_exists=True,
)

study = MlflowSparkStudy(
    study_name=STUDY_NAME,
    storage=storage,
    sampler=TPESampler(seed=42),
)

# n_jobs=4: one trial per GPU (4× A10G), each with its own 24 GB VRAM
study.optimize(objective, n_trials=10, n_jobs=N_GPUS)

print(f"\nBest accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# COMMAND ----------

# DBTITLE 1,Train final model with best params
# Train final model with best hyperparameters — rebuild the matrices on the
# driver (single node, so the driver sees the GPUs and /local_disk0 too)
best = dict(study.best_params)
n_rounds = best.pop('n_rounds')

final_params = {
    'device': 'cuda:0',
    'tree_method': 'hist',
    'objective': 'multi:softmax',
    'num_class': num_class,
    'eval_metric': 'mlogloss',
    **best
}

train_iter = ParquetIter(TRAIN_PATH, feature_cols, target_col, gpu_id=0)
dtrain = xgb.QuantileDMatrix(train_iter, max_bin=best['max_bin'])
test_iter = ParquetIter(TEST_PATH, feature_cols, target_col, gpu_id=0)
dtest = xgb.QuantileDMatrix(test_iter, max_bin=best['max_bin'], ref=dtrain)

final_model = xgb.train(
    final_params,
    dtrain,
    num_boost_round=n_rounds,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    verbose_eval=50,
)

# Final evaluation
preds = final_model.predict(dtest)
y_test = pq.read_table(TEST_PATH, columns=[target_col]).to_pandas()[target_col].to_numpy()
print(f"\nFinal test accuracy: {(preds == y_test).mean():.4f}")

# COMMAND ----------

