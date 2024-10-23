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
- **`translate`**

### Model Conversion Tools
- **`convert`** 
  - Flavors: `xgen`, `redpajama`, `llama_legacy`, `falcon`, `mpt`, `HF`, `T5`

### Model Management Tools
- **`model`**
  - Subcommands: `lora`, `release`, `extract_embeddings`, `average`

### Miscellaneous Tools (Mostly Legacy)
- **`tools`**
  - Subcommands: `LM_scoring`, `oracle_comet`, `run_mmlu`, `spm_to_vocab`, `mbr_bleu`, `embeddings_to_torch`, `oracle_bleu`

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