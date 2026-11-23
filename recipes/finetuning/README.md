# Fine-tune a Qwen chat model with LoRA

Run a short single-GPU fine-tune of [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
in BF16. The bundled examples teach a simple catalog-label format and exercise
the training, checkpoint, merge, and inference steps.

Use a BF16-capable NVIDIA GPU and an [installed Eole environment](../../README.md#installation).
Run the commands from the repository root.

## Prepare the model and an output directory

```bash
export QWEN_LORA_MODEL="$PWD/models/qwen3-0.6B"
export EOLE_LORA_DIR="$PWD/runs/qwen3-lora"
mkdir -p "$PWD/models" "$EOLE_LORA_DIR"
eole convert HF --model_dir Qwen/Qwen3-0.6B --output "$QWEN_LORA_MODEL"
```

Skip conversion if you already have the converted Eole checkpoint. Point
`QWEN_LORA_MODEL` at that directory. Use a fresh `EOLE_LORA_DIR` for each run.

## Train the adapter

```bash
EOLE_TORCH_COMPILE=0 eole train -c recipes/finetuning/train.yaml
```

[train.yaml](train.yaml) adds rank-8 LoRA to the attention query and value
projections and trains for 12 optimizer steps. The base weights are frozen.
The adapter is saved under `$EOLE_LORA_DIR/adapter/step_12`.

[train-prompts.txt](train-prompts.txt) and [train-answers.txt](train-answers.txt)
contain one matching prompt and answer per line. Eole tokenizes the pair and
adds the end-of-turn token to the answer. The Qwen role markers delimit turns,
and Eole's tokenizer converts
`｟newline｠` to a line break. This example computes next-token loss on the
full conversation. To adapt it to your task, supply your own preformatted
training and validation examples and choose an appropriate training schedule.
The bundled run checks that training and checkpoint creation work.

## Generate from the fine-tuned model

Merge the adapter with the same base checkpoint, then run the bundled held-out
prompt:

```bash
eole model lora --action merge --base_model "$QWEN_LORA_MODEL" \
  --lora_weights "$EOLE_LORA_DIR/adapter/step_12" --output "$EOLE_LORA_DIR/merged"
EOLE_TORCH_COMPILE=0 eole predict -c recipes/finetuning/predict.yaml
cat "$EOLE_LORA_DIR/prediction.txt"
```

The 12-step train, merge, and predict sequence was smoke-tested on an RTX 4090.
The merged model is a full checkpoint. Keep the base model directory available
for this example because the saved tokenizer configuration points to its
`tokenizer.json`.
