# Do you support multi-GPU?

Yes. EOLE supports two distributed modes: **data parallelism** and **tensor parallelism**.

## Configuration

First you need to make sure you `export CUDA_VISIBLE_DEVICES=0,1,2,3`.

If you want to use GPU id 1 and 3 of your OS, you will need to `export CUDA_VISIBLE_DEVICES=1,3`

Both `world_size` and `gpu_ranks` need to be set. E.g.:

```yaml
training:
  world_size: 4
  gpu_ranks: [0, 1, 2, 3]
```

## Data Parallelism

The default mode (`parallel_mode: data_parallel`) replicates the model across GPUs and splits the data. This is well-suited for training.

## Tensor Parallelism

Tensor parallelism (`parallel_mode: tensor_parallel`) splits the model tensors across GPUs. This is useful when a single model is too large to fit on one GPU, and is supported for both training and inference.

```yaml
training:
  world_size: 4
  gpu_ranks: [0, 1, 2, 3]
  parallel_mode: tensor_parallel
```

## Multi-node Training

For multi-node distributed training, configure `master_ip` and `master_port`:

* Node 1: `world_size: 4`, `gpu_ranks: [0, 1]`
* Node 2: `world_size: 4`, `gpu_ranks: [2, 3]`

With `accum_count: 2` to accumulate over 2 batches before updating parameters, which reduces inter-node communication.

If you use a regular network card (1 Gbps) then we suggest using a higher `accum_count` to minimize inter-node communication.

