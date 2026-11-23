# Run Gemma 3 text generation

Use [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it) for a small
instruction-tuned text model. Run from the repository root with
[Eole installed](../../README.md#installation) and a BF16-capable NVIDIA GPU.
Accept the model terms on Hugging Face and sign in with `hf auth login` before
downloading the checkpoint.

```bash
export EOLE_MODEL_DIR="$PWD/models"
mkdir -p "$EOLE_MODEL_DIR"
eole convert HF --model_dir google/gemma-3-1b-it \
  --output "$EOLE_MODEL_DIR/gemma-3-1b-it"
EOLE_TORCH_COMPILE=0 eole predict -c recipes/gemma3/predict.yaml
cat /tmp/eole-gemma3-output.txt
```

The [bundled prompt](prompt.txt) uses Gemma's turn markers. Replace the input
with `--src` and set your destination with `--output` when testing your own
prompts. For ordinary chat messages and automatic template application,
follow the [server recipe](../server/README.md).
