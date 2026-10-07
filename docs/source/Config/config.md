# Configuration structure

Eole validates configuration through Pydantic models. Unknown fields are rejected.
Use a YAML file with `eole train -c train.yaml` or `eole predict -c predict.yaml`.
The command runner expands environment variables in YAML paths.

A training file has top-level corpus/vocabulary/logging settings and nested
`model`, `training`, and `transforms_configs` sections. For fine-tuning,
`training.train_from` supplies the saved architecture and tokenization settings.
A prediction file places `model_path`, `src`, device, precision, and decoding
settings at the top level; compatible settings are loaded from the checkpoint.

```yaml
model_path: /path/to/converted/model
src: input.txt
output: output.txt
gpu_ranks: [0]
compute_dtype: bf16
beam_size: 1
top_k: 1
```

Top-level run fields can be overridden by their CLI flags. Nested fields should
be edited in YAML; dotted CLI overrides such as `--training.compute_dtype` are
not supported. Server YAML (`models:`) and recipe orchestration YAML are separate
schemas, not prediction configurations.

See the configuration pages below for individual fields, and the website recipe
index for complete model/data examples. GPU kernel availability is independent
of schema validation.
