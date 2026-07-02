---
sidebar_position: 2
description: Recap of command line utilities and how to call them.
---

# Command Line

`eole` is the single command line utility used to run various tools. These tools are categorized into several groups:

## Categories of Tools

### Main Entrypoints
- **`build_vocab`**
- **`train`**
- **`predict`**

### Model Conversion Tools
- **`convert`** 
  - Flavors: `HF` (universal Hugging Face converter), `T5`, `COMET`, `onmt_config` (legacy OpenNMT-py config migration)

### Model Management Tools
- **`model`**
  - Subcommands: `lora`, `release`, `extract_embeddings`, `average`

### Miscellaneous Tools (Mostly Legacy)
- **`tools`**
  - Subcommands: `LM_scoring`, `oracle_comet`, `run_mmlu`, `spm_to_vocab`, `mbr_bleu`, `embeddings_to_torch`, `oracle_bleu`, `hellaswag`

## Usage

The main entrypoints are typically used with a `yaml` configuration file. Most parameters can also be overridden via corresponding command line flags if needed.

### Examples

```sh
eole build_vocab -c your_config.yaml
eole train -c your_config.yaml
eole predict -c your_config.yaml
```

## Additional Tools

All other tools have specific arguments that can be inspected via the command helpstring `-h`.

### Example

```sh
eole tool_name -h
```

## Native EOLE COMET Scoring

EOLE supports native scoring with pre-converted [Unbabel COMET](https://github.com/Unbabel/COMET)
metrics through `EOLE-COMET`, `EOLE-COMET-KIWI`, and `EOLE-XCOMET`. These hosted
EOLE models can be used directly for validation scoring or with
`eole predict --with_score`. See the
[EOLE-COMET Hugging Face collection](https://huggingface.co/collections/eole-nlp/eole-comet)
for all published COMET-family EOLE conversions.

```sh
eole predict \
  --model_path eole-nlp/wmt22-comet-da-eole-fp16 \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/scores.txt \
  --with_score
```

Use the hosted defaults in your training config with:

- `valid_metrics: ["EOLE-COMET"]`, `valid_metrics: ["EOLE-COMET-KIWI"]`, or `valid_metrics: ["EOLE-XCOMET"]`

Set `comet_model` only when you want to override the default hosted EOLE model.
Validation metrics configured with `valid_metrics` use `comet_compute_dtype` and
default to fp16. Direct `eole predict` scoring uses the regular inference dtype
knob; pass `--compute_dtype fp32` if you need fp32 CLI scoring.

Raw Unbabel COMET repos still need conversion before EOLE can load them:

```sh
eole convert COMET --model Unbabel/wmt22-comet-da --output /path/to/converted/wmt22-comet-da
eole convert COMET --model Unbabel/wmt22-cometkiwi-da --output /path/to/converted/wmt22-cometkiwi-da
eole convert COMET --model Unbabel/XCOMET-XL --token "$HF_TOKEN" --output /path/to/converted/XCOMET-XL
```

See `recipes/scoring/comet_native/` for hosted model IDs, direct scoring examples,
conversion options, and parity checks.

## Native EOLE MetricX Scoring

EOLE supports native scoring with pre-converted [Google MetricX](https://github.com/google-research/metricx)
models through `EOLE-METRICX` and `EOLE-METRICX-QE`. Both scorers default to the
hosted fp32 EOLE model `eole-nlp/metricx-24-hybrid-large-v2p6-eole`. See the
[EOLE-MetricX Hugging Face collection](https://huggingface.co/collections/eole-nlp/eole-metricx)
for all published MetricX EOLE conversions.

```sh
eole predict \
  --model_path eole-nlp/metricx-24-hybrid-large-v2p6-eole \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/metricx-scores.txt \
  --with_score \
  --compute_dtype fp32
```

Omit `--ref` to score in QE mode. MetricX scores are lower-is-better raw error
scores. Validation metrics configured with `valid_metrics` use
`metricx_compute_dtype` and default to fp32 for stability. Direct `eole predict`
scoring uses `--compute_dtype`.

Use the hosted defaults in your training config with:

- `valid_metrics: ["EOLE-METRICX"]` for reference-based scoring
- `valid_metrics: ["EOLE-METRICX-QE"]` for reference-free scoring

Set `metricx_model` only when you want to override the default hosted EOLE model.
Raw Google MetricX repos still need conversion before EOLE can load them:

```sh
eole convert MetricX --model google/metricx-24-hybrid-large-v2p6 --dtype fp32 --output /path/to/converted/metricx-24-hybrid-large-v2p6-eole
```

See `recipes/scoring/metricx_native/` for hosted model IDs, direct scoring
examples, conversion options, and MetricX-23/24 input mode details.
