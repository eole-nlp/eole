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

EOLE supports native scoring with converted [Unbabel COMET](https://github.com/Unbabel/COMET)
metrics through `EOLE-COMET`, `EOLE-COMET-KIWI`, and `EOLE-XCOMET`. These models
use EOLE's generic encoder-only `transformer_encoder_scorer` architecture with
the COMET-specific `scoring_type: comet` specialization.

```sh
eole convert COMET --model Unbabel/wmt22-comet-da --output /path/to/converted/wmt22-comet-da
eole convert COMET --model Unbabel/wmt22-cometkiwi-da --output /path/to/converted/wmt22-cometkiwi-da
eole convert COMET --model Unbabel/XCOMET-XL --token "$HF_TOKEN" --output /path/to/converted/XCOMET-XL
```

Then use the converted model directory in your training config with:

- `valid_metrics: ["EOLE-COMET"]`, `valid_metrics: ["EOLE-COMET-KIWI"]`, or `valid_metrics: ["EOLE-XCOMET"]`
- `comet_model: /path/to/converted/model`

`eole predict --with_score` emits scalar scores for converted scorer models. For
XCOMET, explainable span parity can be validated with the
`recipes/scoring/comet_native/parity_check.py` harness.
