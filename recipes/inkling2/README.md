# Inkling 2 Mini text inference

This recipe runs the text decoder from the
[official Thinking Machines checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
The checkpoint uses NVFP4 and MXFP8 for the routed experts and BF16 for most
other weights. The 24 physical layers provide 48 visits per token. The central
12 layers share their weights across three passes, with separate attention and
convolution state for each visit.

Download the checkpoint and tokenizer files from the Eole checkout.

```bash
hf download thinkingmachines/Inkling-2-Mini \
  --revision c9d028302cd0f035fdf8e7e91f089a4ce94d2195 \
  --local-dir /path/to/Inkling-2-Mini \
  config.json generation_config.json model.safetensors.index.json \
  model-00001-of-00001.safetensors tokenizer.json chat_template.jinja

PYTHONPATH=. python recipes/inkling2/predict.py \
  --model /path/to/Inkling-2-Mini --max-new-tokens 128 \
  "What is the capital of France? Reply with only the city name."
```

The example prints the generated text with the model's message markers, followed
by timings on stderr. The answer block contains `Paris`. The checkpoint takes
about 19 GiB on disk. The RTX 4090 smoke test used about 18.4 GiB of PyTorch GPU
allocations with a short prompt. Allow additional GPU memory for longer inputs.

This is a reference for single-request greedy text generation. Expert weights
stay packed in memory and Eole dequantizes the selected weights to BF16 for the
matrix multiplications. Prefill runs in chunks and retains only the live window
for local-attention layers. The global-attention cache grows with the input.

On NVIDIA SM89 or newer, add `--decode graph` for packed experts, fused
single-token operations, and CUDA graph replay. The first two decode steps warm the kernels, and the third
captures the graph. Triton compiles kernels on first use and reuses them for
matching shapes. Each request has its own cache, sized from the prompt and
`--max-new-tokens`. Use `--decode packed` to run the same expert kernels in eager
mode. Floating-point reduction order can change greedy tokens across decoding
modes. The reference decoder remains the default for numerical comparisons.
