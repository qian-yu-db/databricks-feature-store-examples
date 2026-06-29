# XGBoost + Optuna on a 4-GPU Single Node

This example compares three ways to train XGBoost models with Optuna hyperparameter
tuning on a single `g5.24xlarge` Databricks GPU node.

## Problem Details

- Raw input: roughly 35 parquet files at about 40 MB each.
- Pandas dataframe: all numeric features, roughly 67 GB in memory before downcasting.
- GPU instance: `g5.24xlarge`, single node, 4 A10G GPUs.
- CPU memory: 384 GB total, so CPU memory is not the first bottleneck on this node.
- GPU memory: 24 GB per GPU.
- Workload: XGBoost model training with Optuna hyperparameter tuning.

The main challenges are:

- Train a 67 GB dataset when each A10G has only 24 GB of VRAM.
- Use all 4 GPUs on the node effectively during hyperparameter search.

## Notebooks

| Notebook | Approach |
| --- | --- |
| `xgboost_optuna_approach_1.py` | XGBoost non-Spark + Optuna `MlflowSparkStudy` + in-memory numpy |
| `xgboost_optuna_approach_2.py` | XGBoost non-Spark + Optuna `MlflowSparkStudy` + local NVMe parquet streaming |
| `xgboost_optuna_approach_3.py` | `xgboost.spark` + Optuna without `MlflowSparkStudy` |

These notebooks were tested with 67 GB of synthetic data. In that test, approach 2
ran the fastest.

## Approach 1: Non-Spark XGBoost + SparkStudy + In-Memory Numpy

This approach keeps the downcast train/test arrays in memory and uses
`MlflowSparkStudy` to run 4 concurrent Optuna trials, one trial per GPU.

Key implementation details:

- Downcast numeric data to reduce CPU memory pressure.
  - Integer columns are downcast losslessly based on min/max ranges.
  - This has limited impact on GPU training memory, but improves CPU-side data
    transformation and wrangling speed.
- Use `QuantileDMatrix` and `DataIter` to reduce GPU memory pressure.
  - The iterator feeds 10M-row numpy slices to `QuantileDMatrix`.
  - XGBoost builds the quantized index batch by batch on CPU, so raw arrays are
    not copied wholesale to the GPU.
  - At `xgb.train(device="cuda:N")` time, XGBoost converts the CPU quantized index
    into the GPU `EllpackPage` format. Only the compressed representation lands in
    VRAM: roughly `n_rows * n_features * log2(max_bin)` bits, or about 1 byte per
    value at `max_bin=128`, which keeps this dataset well under 24 GB per GPU.
  - The quantile sketch runs on CPU, so 4 concurrent trials contend for CPU cores.
    Consider setting `nthread` to roughly `total_cores / 4`.
- `MlflowSparkStudy.optimize()` runs as one Spark job with `numPartitions=n_jobs`.
  - `TaskContext.partitionId()` is `0..n_jobs-1`.
  - The notebook maps each concurrent trial to its own A10G.

Best fit:

- Maximum HPO throughput on this exact box when the data comfortably fits in RAM
  several times over.

Trade-offs:

- Simplest data path: downcast once, arrays reach Spark workers through the pickled
  objective closure, and each trial quantizes directly from RAM with zero disk I/O.
- Highest CPU memory cost: roughly 5-6 copies of the downcast dataset across driver
  copies, broadcast payload, and 4 worker copies.
- Works here because the node has 384 GB RAM, but has only about 2-3x growth room
  before CPU OOM becomes likely.
- The CPU quantile sketch can become a shared bottleneck when 4 trials run together.

## Approach 2: Non-Spark XGBoost + SparkStudy + NVMe Parquet Streaming

This approach stages downcast train/test data to `/local_disk0` once, then each
Optuna trial streams parquet row groups from local NVMe to its assigned GPU.

Key implementation details:

- Downcast numeric data to reduce CPU memory pressure.
  - Integer columns are downcast losslessly based on min/max ranges.
  - This has limited impact on GPU training memory, but improves CPU-side data
    transformation and wrangling speed.
- Use `QuantileDMatrix` and `DataIter` to reduce GPU memory pressure.
  - The `DataIter` reads one parquet row group at a time.
  - Each batch is moved to the assigned GPU as a cupy array because XGBoost's GPU
    quantile sketch requires on-device input.
  - `QuantileDMatrix` keeps only the quantized representation in VRAM: roughly
    `n_rows * n_features * log2(max_bin)` bits, or about 1 byte per value at
    `max_bin=128`, instead of 8 bytes per float64 value.
- Uses local NVMe (`/local_disk0`) as the staging layer.
- `MlflowSparkStudy.optimize()` runs as one Spark job with `numPartitions=n_jobs`.
  - `TaskContext.partitionId()` is `0..n_jobs-1`.
  - The notebook maps each concurrent worker to its own A10G.

Best fit:

- Same trial parallelism as approach 1, but with much lower CPU-memory pressure.
- A better base when the dataset grows beyond what RAM duplication can tolerate.

Trade-offs:

- Worker CPU memory stays nearly flat, about 1-2 GB regardless of total dataset size.
- Pays NVMe re-reads per trial, typically about two passes per matrix.
- Requires re-staging after each cluster restart because `/local_disk0` is ephemeral.
- Per-trial VRAM is still the binding constraint, so it is sensitive to leftover GPU
  allocations from other notebooks.

## Approach 3: XGBoost Spark + Optuna Without SparkStudy

This approach uses Spark dataframes natively with the `xgboost.spark` API. Optuna runs
in the driver process, and each trial trains on all 4 GPUs at once.

Key implementation details:

- Use Spark dataframe input directly with `SparkXGBClassifier`.
- No pandas load, no manual downcasting, and no closure or local staging workaround.
- GPU memory pressure is reduced by sharding data across workers.
- Optuna setup is simpler because the objective runs in the driver process.
- Trials run sequentially, but each trial uses all 4 GPUs.

Best fit:

- When one trial's data cannot fit on a single GPU.
- When the notebook should look more like a production path that survives data growth,
  cluster restarts, or multi-node scaling.

Trade-offs:

- Data never leaves Spark, and Spark assigns GPUs to workers. This avoids manual
  `partitionId -> GPU` mapping and reduces GPU collision risk.
- Per-GPU memory is solved by sharding: each worker holds roughly one quarter of the
  rows, plus the internal `QuantileDMatrix` compression.
- This is the only approach here that scales to multi-node without major code changes.
- Lowest HPO throughput: trials run one at a time, and each trial includes distributed
  training coordination overhead.
- Sharded histogram training can differ slightly from single-GPU results.
- The `xgboost.spark` API may require more code refactoring than the non-Spark variants.

## GPU Memory Factors

For XGBoost GPU training, the largest drivers of memory usage are:

1. `n_samples * n_features`: fixed by the dataset.
2. `max_bin`: threshold at 256, and it also scales histogram size. Prefer `max_bin <= 128`.
3. `max_depth`: histogram count scales with the number of nodes processed per level.
4. `subsample`: can reduce gradient buffer size.
5. `n_estimators`: trees are built sequentially, so this has minimal additional memory impact.

In practice, the top two factors usually matter most: dataset size and `max_bin`.

## References

- [XGBoost QuantileDMatrix data iterator](https://xgboost.readthedocs.io/en/latest/python/examples/quantile_data_iterator.html)
- [Databricks Optuna `MlflowSparkStudy`](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/optuna)
