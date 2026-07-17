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
    compute_dtype: "fp32"
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

## Hugging Face Streaming Datasets

Datasets can stream directly from Hugging Face with `hf://` paths. The last path component is the dataset column to read.

```yaml
data:
    corpus_1:
        path_src: hf://eole-nlp/europarl-v10.de-en/de
        path_tgt: hf://eole-nlp/europarl-v10.de-en/en
        path_sco: hf://eole-nlp/europarl-v10.de-en/sco
```

Supported URI forms are:

- `hf://owner/dataset/field` for the default `train` split.
- `hf://owner/dataset/split/field` for an explicit split.
- `hf://owner/dataset/config/field` for a dataset config with the default `train` split.
- `hf://owner/dataset/config/split/field` for an explicit config and split.

When using HF streaming, `path_tgt` and `path_sco` must point to the same dataset, config, and split as `path_src`; only the final field name may differ.

Use `additional_fields` to copy extra HF columns into each example before transforms run. This is useful when a transform prompt needs metadata such as WMT24++ `domain`, `document_id`, or `segment_id`.

```yaml
data:
    gsm8k:
        path_src: hf://skrishna/gsm8k_only_answer/train/text
        path_tgt: hf://skrishna/gsm8k_only_answer/train/label
        transforms: [chat, huggingface_tokenize]
    wmt24pp-de:
        path_src: hf://google/wmt24pp/en-de_DE/source
        path_tgt: hf://google/wmt24pp/en-de_DE/target
        additional_fields: [domain, document_id, segment_id]
        transforms: [chat, huggingface_tokenize]
    estimator:
        path_src: hf://eole-nlp/estimator_chatml/1720_da/prompt
        path_sco: hf://eole-nlp/estimator_chatml/1720_da/sco
        transforms: [huggingface_tokenize]
```

Compact URIs treat common split names such as `train`, `valid`, `validation`, and `test` as splits. If a dataset config is literally named like a split, use the explicit four-part form: `hf://owner/dataset/config/split/field`.

Missing configured `additional_fields` fail fast when the HF stream is read. `additional_fields` is currently supported only for HF streaming corpora.

## Dataset-Level Transform Overrides

Some transforms can opt in to dataset-level overrides via a corpus-local `transforms_configs` block. This is currently supported by the `chat` transform, which lets each corpus use a different prompt while sharing the same transform pipeline.

```yaml
transforms: [chat, huggingface_tokenize]

transforms_configs:
    chat:
        add_generation_prompt: true
        messages:
          - role: user
            content: "{src}"
    huggingface_tokenize:
        path: /Users/david/Development/Models/eole/eurollm-1.7B/tokenizer.json
        max_length: 128

data:
    wmt24pp-de:
        path_src: hf://google/wmt24pp/en-de_DE/source
        path_tgt: hf://google/wmt24pp/en-de_DE/target
        additional_fields: [domain, document_id]
        transforms: [chat, huggingface_tokenize]
        transforms_configs:
            chat:
                messages:
                  - role: system
                    content: "You are a professional translator."
                  - role: user
                    content: "Translate from English into German.\nDomain: {domain}\nDocument: {document_id}\n\nEnglish: {src}\nGerman:"

    wmt24pp-fr:
        path_src: hf://google/wmt24pp/en-fr_FR/source
        path_tgt: hf://google/wmt24pp/en-fr_FR/target
        transforms: [chat, huggingface_tokenize]
        transforms_configs:
            chat:
                messages:
                  - role: system
                    content: "You are a careful localization translator for Canadian French."
                  - role: user
                    content: "Translate from English into French for Canada. Preserve names, numbers, and formatting.\n\nEnglish: {src}\nFrench:"
```

Dataset-level overrides are intentionally scoped. Apply-time prompt settings such as `chat.messages` can vary by corpus. Warm-up settings such as `huggingface_tokenize.path`, `huggingface_tokenize.max_length`, BPE models, or SentencePiece models are global for the transform instance and cannot be overridden per dataset yet. Unsupported dataset-level transform overrides fail during config validation instead of being silently ignored.

If every corpus using `chat` supplies its own `transforms_configs.chat.messages`, the global `transforms_configs.chat.messages` may be empty. Any chat corpus without a dataset-level `messages` override needs global messages as a fallback. Overrides for transforms that are not enabled in the corpus `transforms` list fail during config validation.
