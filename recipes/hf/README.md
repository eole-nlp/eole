# Run a supported Hugging Face model directly

Run from the repository root in a CUDA-enabled PyTorch environment:

```bash
pip install -e .
eole predict -c recipes/hf/predict.yaml
```

This YAML points `model_path` at `Qwen/Qwen3.5-0.8B` rather than a local Eole
checkpoint. Eole resolves supported HF architecture/configuration, downloads
the required files, and loads weights without a separate `eole convert HF`
step. First run needs network access and model-storage space. The response is
saved to `/tmp/eole-hf-output.txt`.

The bundled prompt is already formatted for Qwen chat, including Eole's newline
marker and an empty thinking block. Change the prompt format when switching
model families; `eole predict` does not turn arbitrary plain text into a chat
conversation automatically. Use the [server recipe](../server/README.md) when
you want message-based input and chat-template application.

Support is limited to architectures and quantization layouts implemented by
Eole, not arbitrary Transformers models. Gated models require HF authentication
(`hf_token` in the inference configuration, or your normal HF login). Avoid
committing credentials. For reusable/offline Eole artifacts or custom conversion
options, use `eole convert HF` as shown in the server recipe.

The YAML is schema-checked; actual model download and GPU generation remain
hardware/network-dependent. For native Qwen3.8 MTP use the dedicated
[MTP recipe](../qwen38/README.md), which checks saved MTP tensors explicitly.
