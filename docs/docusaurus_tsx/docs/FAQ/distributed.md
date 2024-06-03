# Do you support multi-gpu?

First you need to make sure you `export CUDA_VISIBLE_DEVICES=0,1,2,3`.

If you want to use GPU id 1 and 3 of your OS, you will need to `export CUDA_VISIBLE_DEVICES=1,3`

Both `-world_size` and `-gpu_ranks` need to be set. E.g. `-world_size 4 -gpu_ranks 0 1 2 3` will use 4 GPU on this node only.

**Warning - Deprecated**

Multi-node distributed training has not been properly re-implemented since OpenNMT-py 2.0.

If you want to use 2 nodes with 2 GPU each, you need to set `-master_ip` and `-master_port`, and

* `-world_size 4 -gpu_ranks 0 1`: on the first node
* `-world_size 4 -gpu_ranks 2 3`: on the second node
* `-accum_count 2`: This will accumulate over 2 batches before updating parameters.

If you use a regular network card (1 Gbps) then we suggest to use a higher `-accum_count` to minimize the inter-node communication.

**Note:**

In the legacy version, when training on several GPUs, you couldn't have them in 'Exclusive' compute mode (`nvidia-smi -c 3`).

The multi-gpu setup relied on a Producer/Consumer setup. This setup means there will be `2<n_gpu> + 1` processes spawned, with 2 processes per GPU, one for model training and one (Consumer) that hosts a `Queue` of batches that will be processed next. The additional process is the Producer, creating batches and sending them to the Consumers. This setup is beneficial for both wall time and memory, since it loads data shards 'in advance', and does not require to load it for each GPU process.

The new codebase allows GPUs to be in exclusive mode, because batches are moved to the device later in the process. Hence, there is no 'producer' process on each GPU.