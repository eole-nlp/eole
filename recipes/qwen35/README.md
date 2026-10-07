# Qwen3.5: text, vision, streaming, and serving

Run from the repository root. Follow the [installation guide](../../README.md#installation)
for Eole CUDA kernels, FlashAttention, and `fla-core` in your CUDA-enabled environment.
The inference scripts use `Qwen/Qwen3.5-4B`; the server example uses a separate
27B INT4 checkpoint. These are different model paths, not interchangeable steps.

## Convert and run the 4B examples

```bash
export EOLE_MODEL_DIR="$PWD/models"
mkdir -p "$EOLE_MODEL_DIR"
eole convert HF --model_dir Qwen/Qwen3.5-4B \
  --output "$EOLE_MODEL_DIR/qwen3.5-4B"
```

The scripts select FlashAttention; install `flash-attn --no-build-isolation`, or
change `self_attn_backend="flash"` to `"pytorch"` to use the fallback backend.

```bash
python recipes/qwen35/test_inference.py   # image questions with bundled fixtures
python recipes/qwen35/test_inference2.py  # text generation
python recipes/qwen35/test_stream.py      # live text streaming
```

The first script prints one response per image in `eole/tests/data/images`.
The second prints a generated passage. Adjust GPU ranks, generation length, and
model paths in the scripts to fit your hardware.

## Serve a model

For the small-model starting point, use the [server recipe](../server/README.md).
`serve.yaml` in this directory is an advanced Qwen3.5-27B INT4 example: it
requires a converted checkpoint at `$EOLE_MODEL_DIR/qwen3.5-27B-int4`, quantization
kernels, and FlashAttention. Its large context/cache settings must be adjusted
to available VRAM; it is not a conversion or download recipe.

```bash
eole serve -c recipes/qwen35/serve.yaml --host 127.0.0.1
```

For Qwen3.8 MTP and a reproducible baseline comparison, use the
[Qwen3.8 recipe](../qwen38/README.md).
