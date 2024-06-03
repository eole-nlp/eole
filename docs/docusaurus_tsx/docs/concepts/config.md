---
sidebar_position: 1
description: Crash course on Eole configuration.
---

# Configuration

One of the core principles around Eole is the structured configuration logic via Pydantic models. This allows for centralized validation of numerous parameters, as well as proper nesting of various objects and scopes.
It can be a lot at first, but that's a necessary evil for proper structure and modularity.

Here is an example configuration to train a GPT-style language model:

```yaml
# General data/vocab/run related settings
seed: 42
save_data: test_save_data
src_vocab_size: 60000
tgt_vocab_size: 60000
share_vocab: true
src_vocab: my_vocab.txt
report_every: 100 # report stats every 100 steps

# datasets
data:
    # training sets can be numerous, and named anything
    corpus_1:
        path_src: my_training_set.txt
    # single validation set, always named "valid"
    valid:
        path_src: my_validation_set.txt

# default transforms, in application order
transforms: [onmt_tokenize, filtertoolong]
# transforms configuration
transforms_configs:
  onmt_tokenize:
    src_subword_type: bpe
    src_subword_model: my_subwords_model.bpe
    src_onmttok_kwargs: {"mode": "aggressive", "joiner_annotate": True, "preserve_placeholders":
    True, "case_markup": True, "soft_case_regions": True, "preserve_segmented_tokens":
    True}
  filtertoolong:
    src_seq_length: 512
    tgt_seq_length: 512

# model architecture configuration
model:
    architecture: "transformer_lm"
    layers: 6
    heads: 8
    hidden_size: 512
    transformer_ff: 2048
    embeddings:
        word_vec_size: 512
        position_encoding: true

# training routine configuration
training:
    # Train on a single GPU
    world_size: 1
    gpu_ranks: [0]
    # Batching
    batch_size: 2048
    batch_type: tokens
    # Optimizer
    model_dtype: "fp32"
    optim: "adam"
    learning_rate: 2
    warmup_steps: 8000
    decay_method: "noam"
    adam_beta2: 0.998
    # Hyperparams
    dropout_steps: [0]
    dropout: [0.1]
    attention_dropout: [0.1]
    max_grad_norm: 0
    label_smoothing: 0.1
    param_init: 0
    param_init_glorot: true
    normalization: "tokens"
    # Where to save the checkpoints (creates a directory)
    model_path: my_model
    # Steps intervals
    save_checkpoint_steps: 10
    train_steps: 50
    valid_steps: 500
```