# How do I train the Transformer model?

The transformer model is very sensitive to hyperparameters. To run it
effectively you need to set different options that mimic the [Google](https://arxiv.org/abs/1706.03762) setup. We have confirmed the following configuration can replicate their WMT results.

Please have a look at the Example of WMT17 EN-DE.

```yaml
<data configuration>
...

# General opts
save_model: mybasemodel
save_checkpoint_steps: 10000
valid_steps: 10000
train_steps: 200000

# Batching
bucket_size: 262144
world_size: 4
gpu_ranks: [0, 1, 2, 3]
num_workers: 2
batch_type: "tokens"
batch_size: 4096
valid_batch_size: 2048
accum_count: [4]
accum_steps: [0]

# Optimization
model_dtype: "fp16"
optim: "adam"
learning_rate: 2
warmup_steps: 8000
decay_method: "noam"
adam_beta2: 0.998
max_grad_norm: 0
label_smoothing: 0.1
param_init: 0
param_init_glorot: true
normalization: "tokens"

# Model
encoder_type: transformer
decoder_type: transformer
position_encoding: true
enc_layers: 6
dec_layers: 6
heads: 8
hidden_size: 512
word_vec_size: 512
transformer_ff: 2048
dropout_steps: [0]
dropout: [0.1]
attention_dropout: [0.1]
```

Here are what the most important parameters mean:

* `param_init_glorot` & `param_init 0`: correct initialization of parameters;
* `position_encoding`: add sinusoidal position encoding to each embedding;
* `optim adam`, `decay_method noam`, `warmup_steps 8000`: use special learning rate;
* `batch_type tokens`, `normalization tokens`: batch and normalize based on number of tokens and not sentences;
* `accum_count 4`: compute gradients based on four batches;
* `label_smoothing 0.1`: use label smoothing loss.