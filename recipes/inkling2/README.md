# Inkling 2 Mini text inference

This recipe runs the text decoder from the
[official Thinking Machines checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
The checkpoint uses NVFP4 and MXFP8 for the routed experts and BF16 for most
other weights. The 24 physical layers provide 48 visits per token. The central
12 layers share their weights across three passes, with separate attention and
convolution state for each visit.

Run the following commands from the Eole checkout. The recipe opens the
original Hugging Face safetensors and tokenizer files directly, so a model
conversion step is unnecessary.

```bash
hf download thinkingmachines/Inkling-2-Mini \
  --revision c9d028302cd0f035fdf8e7e91f089a4ce94d2195 \
  --local-dir /path/to/Inkling-2-Mini \
  config.json generation_config.json model.safetensors.index.json \
  model-00001-of-00001.safetensors tokenizer.json chat_template.jinja

PYTHONPATH=. python recipes/inkling2/predict.py \
  --model /path/to/Inkling-2-Mini --decode graph --max-new-tokens 128 \
  "What is the capital of France? Reply with only the city name."
```

The example prints the generated text with the model's message markers, followed
by timings on stderr. The answer block contains `Paris`. The checkpoint takes
about 19 GiB on disk. The RTX 4090 smoke test used about 18.4 GiB of PyTorch GPU
allocations with a short prompt. Allow additional GPU memory for longer inputs.

## Choose a decode mode

All three modes use the same checkpoint, prompt prefill, and greedy token
selection. The flag controls how Eole computes each token after prefill.

| `--decode` | How it runs |
|---|---|
| `reference` (default) | Eager PyTorch with a growing cache. Selected expert weights are dequantized to BF16 before the matrix multiplications. |
| `packed` | Eager execution with a growing cache and expert kernels that multiply directly from the packed weights. |
| `graph` | Packed expert kernels, fused single-token operations, and CUDA graph replay with preallocated KV and convolution state. |

`packed` and `graph` require an NVIDIA GPU with compute capability 8.9 or newer.
`graph` sizes the cache from the prompt and `--max-new-tokens`, warms up the first
two decode steps, and captures on the third. First use of a kernel shape can
trigger Triton compilation. Floating-point reduction order can affect greedy
output across modes.

The recipe uses a dedicated text decoder and the Inkling chat template. Use
`predict.py` above for Inkling and the Qwen recipes for Qwen models.
