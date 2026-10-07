---
sidebar_position: 2
---

# Quickstart

Complete the [installation](index.md#installation) first. Commands below run from
the repository root. For a clone without an installed console command, replace
`eole` with `python -m eole.bin.main`.

## Train a small translation model

Download and extract the toy data, keeping your working directory at the root:

```bash
wget https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
tar xf toy-ende.tar.gz
```

Create `toy_en_de.yaml` with the complete configuration below. Parallel text
contains one tokenized sentence per line, with corresponding source and target
lines aligned. This short run demonstrates the workflow, not translation quality.

```yaml
src_vocab: toy-ende/run/example.vocab.src
tgt_vocab: toy-ende/run/example.vocab.tgt
save_data: toy-ende/run/samples
overwrite: false
data:
  corpus_1:
    path_src: toy-ende/src-train.txt
    path_tgt: toy-ende/tgt-train.txt
  valid:
    path_src: toy-ende/src-val.txt
    path_tgt: toy-ende/tgt-val.txt
model:
  architecture: transformer
training:
  world_size: 1
  gpu_ranks: [0]
  model_path: toy-ende/run/model
  save_checkpoint_steps: 500
  train_steps: 1000
  valid_steps: 500
  bucket_size: 1000
```

```bash
eole build_vocab -c toy_en_de.yaml --n_sample 10000
eole train -c toy_en_de.yaml
eole predict --model_path ./toy-ende/run/model --src toy-ende/src-test.txt \
  --output toy-ende/pred_1000.txt --gpu_ranks 0
```

`build_vocab` samples up to 10,000 lines per corpus. The checkpoint root selects
the latest saved step; use a step subdirectory to select a specific checkpoint.
See [WMT17](recipes/wmt17/README.md) for a larger training workflow.

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

See the [server recipe](recipes/server/README.md) for API requests. Model support
is architecture-specific; the HF converter does not support every Hub model.
For knowledge evaluation, use [MMLU](recipes/mmlu/README.md); that harness
generates an answer token rather than ranking option log probabilities.

## Fine-tune a pretrained model

The [Llama2 recipe](recipes/llama2/README.md) documents NF4/LoRA fine-tuning.
The [REINFORCE recipe](recipes/rl/README.md) describes scorer-reward fine-tuning.
Supply the required model and data before launching either template.

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
