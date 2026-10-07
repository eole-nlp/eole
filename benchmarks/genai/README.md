# Gemma3 generation benchmark examples

Run scripts from the repository root in a CUDA-enabled environment. These are
backend-specific experiments, not a controlled current performance ranking.
The [historical results](results.md) include Eole runs from January 2026, but do
not fully specify GPU, package versions, and equivalent decoding settings for
all engines. They have not been reproduced on the current checkout.

## Eole

```bash
export EOLE_MODEL_DIR=/path/to/models
eole convert HF --model_dir google/gemma-3-1b-it \
  --output "$EOLE_MODEL_DIR/gemma-3-1b-it" --token "$HF_TOKEN"
python benchmarks/genai/generate-eole.py
```

Authenticate for the selected checkpoint if required. The script uses four
prompts, BF16, GPU 0, FlashAttention, and a 2048-token output cap. Install
`flash-attn --no-build-isolation`. It reports Eole inference timing; model load
and compilation time should be reported separately in any new measurements.

## Other engines

`generate-hf.py`, `generate-vllm.py`, and `generate-ct2.py` require Transformers,
vLLM, and CTranslate2 respectively, in compatible environments. The first two
load `google/gemma-3-1b-it` directly. CTranslate2 requires a separate converted
model at `$EOLE_MODEL_DIR/gemma-3-1b-it-ct2`. Its script now expands that variable.
These scripts have different generation and timing settings; inspect and align
them before comparing measurements. Their external package APIs were not
runtime-tested in this audit.

For a YAML-driven baseline/MTP benchmark with recorded hardware, warmup,
acceptance, and output differences, see [Qwen3.8](../../recipes/qwen38/README.md).
For current compilation options, see [the compilation guide](../../TORCHCOMPILE_README.md).
