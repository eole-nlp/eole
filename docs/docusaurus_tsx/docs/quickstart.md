---
sidebar_position: 2
---

# Quickstart

Complete the [installation](index.md#installation) first. Commands below run from
the repository root. For a clone without an installed console command, replace
`eole` with `python -m eole.bin.main`.

## Train a small translation model

Start with the [English–German Transformer tutorial](../../../recipes/wmt17/README.md).
It builds a vocabulary, trains on CPU, translates held-out sentences, and scores
the result. The same tutorial shows how to train on the WMT corpus on a GPU.

## Generate with a supported Hugging Face model

The [direct HF recipe](recipes/hf/README.md) uses a small Qwen model without a
separate conversion command:

```bash
eole predict -c recipes/hf/predict.yaml
```

Its YAML specifies GPU 0 through `gpu_ranks: [0]`, BF16 compute, PyTorch attention,
and an input fixture. Supported model IDs are resolved at inference time;
this still downloads weights. For a reusable converted artifact:

```bash
export EOLE_MODEL_DIR="$PWD/models"
eole convert HF --model_dir Qwen/Qwen3.5-0.8B \
  --output "$EOLE_MODEL_DIR/qwen3.5-0.8B"
eole serve -c recipes/server/serve.example.yaml --host 127.0.0.1
```

See the [server recipe](../../../recipes/server/README.md) for API requests. Model support
is architecture-specific; the HF converter does not support every Hub model.
For knowledge evaluation, use [MMLU](../../../recipes/mmlu/README.md); that harness
generates an answer token rather than ranking option log probabilities.

## Fine-tune a pretrained model

Start with the [Qwen LoRA recipe](../../../recipes/finetuning/README.md) for a short single-GPU run with bundled data.
The [REINFORCE recipe](../../../recipes/rl/README.md) describes scorer-reward fine-tuning with your own model and data.

For instruction datasets, prompt masking can be configured as a fragment in a
complete training YAML:

```yaml
transforms: [insert_mask_before_placeholder, sentencepiece, filtertoolong]
transforms_configs:
  insert_mask_before_placeholder:
    response_patterns: ["Response : ｟newline｠"]
training:
  zero_out_prompt_loss: true
```

Choose a response pattern and tokenizer matching your dataset and checkpoint.
The transform inserts a loss-mask marker; `training.zero_out_prompt_loss` makes
the loss ignore the prompt portion.
