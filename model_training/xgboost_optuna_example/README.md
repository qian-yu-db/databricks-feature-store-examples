# Training & Tuning on Large Datasets with Classic GPU Compute

**Theme:** how to train and hyperparameter-tune a model when the training dataset is
much larger than a single GPU's memory, using **classic (provisioned) GPU compute** —
a multi-GPU Databricks node running the ML Runtime.

The example uses XGBoost + [Optuna](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/optuna),
but the patterns generalize to any GPU trainer that can quantize or shard its input.
It works through three ways to spend a fixed multi-GPU budget on a hyperparameter search
and the trade-offs between them.

## The problem

Two constraints define this class of workload:

1. **The dataset does not fit in one GPU's VRAM.** Raw data measured in tens of GB has
   to be trained on GPUs with far less memory each — so it must be quantized, streamed,
   or sharded rather than loaded whole.
2. **You have N GPUs on one classic node and want to use them all.** A hyperparameter
   search is embarrassingly parallel, so the design question is *how* the trials map onto
   the GPUs: N concurrent trials (one per GPU) versus one trial at a time across all N GPUs.

> **Classic vs. serverless GPU.** These examples target *classic* GPU compute — a
> provisioned cluster/node with a specific instance type and the Databricks Runtime for ML,
> where you control GPU count and VRAM. Serverless GPU (AI Runtime) is a different, managed
> path; the memory and parallelism techniques here still apply, but the cluster-topology
> details do not map one-to-one.

### Example setup

The numbers below come from one concrete run and are illustrative — treat them as a worked
example, not requirements:

| | Example value |
| --- | --- |
| Raw input | ~35 parquet files, ~40 MB each |
| In-memory pandas frame | all-numeric, ~67 GB before downcasting |
| GPU node | `g5.24xlarge`, single node, **4× A10G** |
| GPU memory | 24 GB per GPU |
| CPU memory | 384 GB total (not the first bottleneck on this node) |

The core tension: a ~67 GB dataset trained on GPUs with only 24 GB VRAM each, while keeping
all 4 GPUs busy during the search. In this test, **approach 2 ran the fastest.**

## The three patterns

| Notebook | Pattern | Trial parallelism |
| --- | --- | --- |
| `xgboost_optuna_approach_1.py` | Non-Spark XGBoost + Optuna `MlflowSparkStudy`, data held **in memory** | N concurrent trials, one per GPU |
| `xgboost_optuna_approach_2.py` | Non-Spark XGBoost + Optuna `MlflowSparkStudy`, data **streamed from local NVMe** | N concurrent trials, one per GPU |
| `xgboost_optuna_approach_3.py` | `xgboost.spark` (distributed) + Optuna in the driver | 1 trial at a time, across all N GPUs |

Approaches 1 and 2 maximize search *throughput* (more trials in flight); approach 3
maximizes per-trial *capacity* and scalability. The rest of this README covers each in
detail.

## Approach 1: In-memory numpy + concurrent per-GPU trials

Keeps the downcast train/test arrays in memory and uses `MlflowSparkStudy` to run N
concurrent Optuna trials, one trial per GPU.

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
  - The quantile sketch runs on CPU, so concurrent trials contend for CPU cores.
    Consider setting `nthread` to roughly `total_cores / n_jobs`.
- `MlflowSparkStudy.optimize()` runs as one Spark job with `numPartitions=n_jobs`.
  - `TaskContext.partitionId()` is `0..n_jobs-1`.
  - The notebook maps each concurrent trial to its own GPU.

**Best fit:** maximum HPO throughput when the data comfortably fits in RAM several times over.

Trade-offs:

- Simplest data path: downcast once, arrays reach Spark workers through the pickled
  objective closure, and each trial quantizes directly from RAM with zero disk I/O.
- Highest CPU memory cost: roughly 5-6 copies of the downcast dataset across driver
  copies, broadcast payload, and worker copies.
- Works when the node has ample RAM, but has limited growth room (about 2-3x here) before
  CPU OOM becomes likely.
- The CPU quantile sketch can become a shared bottleneck when many trials run together.

## Approach 2: NVMe parquet streaming + concurrent per-GPU trials

Stages downcast train/test data to `/local_disk0` once, then each Optuna trial streams
parquet row groups from local NVMe to its assigned GPU. Same trial parallelism as
approach 1, with much lower CPU-memory pressure.

Key implementation details:

- Downcast numeric data to reduce CPU memory pressure (as in approach 1).
- Use `QuantileDMatrix` and `DataIter` to reduce GPU memory pressure.
  - The `DataIter` reads one parquet row group at a time.
  - Each batch is moved to the assigned GPU as a cupy array because XGBoost's GPU
    quantile sketch requires on-device input.
  - `QuantileDMatrix` keeps only the quantized representation in VRAM: roughly
    `n_rows * n_features * log2(max_bin)` bits, or about 1 byte per value at
    `max_bin=128`, instead of 8 bytes per float64 value.
- Uses local NVMe (`/local_disk0`) as the staging layer.
- `MlflowSparkStudy.optimize()` runs as one Spark job with `numPartitions=n_jobs`, mapping
  each concurrent worker to its own GPU (`TaskContext.partitionId()` is `0..n_jobs-1`).

**Best fit:** the same trial parallelism as approach 1, but a better base when the dataset
grows beyond what RAM duplication can tolerate.

Trade-offs:

- Worker CPU memory stays nearly flat, about 1-2 GB regardless of total dataset size.
- Pays NVMe re-reads per trial, typically about two passes per matrix.
- Requires re-staging after each cluster restart because `/local_disk0` is ephemeral.
- Per-trial VRAM is still the binding constraint, so it is sensitive to leftover GPU
  allocations from other notebooks.

## Approach 3: Distributed `xgboost.spark` + sequential full-node trials

Uses Spark dataframes natively with the `xgboost.spark` API. Optuna runs in the driver
process, and each trial trains on all N GPUs at once.

Key implementation details:

- Use Spark dataframe input directly with `SparkXGBClassifier`.
- No pandas load, no manual downcasting, and no closure or local staging workaround.
- GPU memory pressure is reduced by sharding data across workers.
- Optuna setup is simpler because the objective runs in the driver process.
- Trials run sequentially, but each trial uses all N GPUs.

**Best fit:** when a single trial's data cannot fit on one GPU, or when the notebook should
resemble a production path that survives data growth, cluster restarts, and multi-node scaling.

Trade-offs:

- Data never leaves Spark, and Spark assigns GPUs to workers. This avoids manual
  `partitionId -> GPU` mapping and reduces GPU collision risk.
- Per-GPU memory is solved by sharding: each worker holds roughly `1/N` of the rows, plus
  the internal `QuantileDMatrix` compression.
- The only approach here that scales to multi-node without major code changes.
- Lowest HPO throughput: trials run one at a time, and each includes distributed training
  coordination overhead.
- Sharded histogram training can differ slightly from single-GPU results.
- The `xgboost.spark` API may require more code refactoring than the non-Spark variants.

## Choosing an approach

| If… | Use |
| --- | --- |
| Data fits in RAM several times over, and you want the most trials per hour | Approach 1 (in-memory) |
| Data is too large to duplicate in RAM, but one trial still fits on one GPU | Approach 2 (NVMe streaming) — the fastest in the example run |
| A single trial's data cannot fit on one GPU, or you need a path that scales to multiple nodes | Approach 3 (distributed Spark) |

Approaches 1 and 2 trade per-trial capacity for search throughput; approach 3 does the
reverse. Start from the memory constraint (does one trial fit on one GPU?), then optimize
for throughput.

## GPU memory factors

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
