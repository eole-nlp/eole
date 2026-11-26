# Inkling 2 Mini text generation

Eole runs the text decoder from the
[official Thinking Machines checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
It uses the regular Eole transformer decoder and inference engine. The central
12 of its 24 physical layers are visited three times. Each of the 48 visits
has its own attention and convolution state.

## Download and run

Run these commands from the Eole checkout. The checkpoint takes about 19 GiB
on disk and uses a mix of NVFP4 and MXFP8 experts with BF16 dense weights.

```bash
hf download thinkingmachines/Inkling-2-Mini \
  --revision c9d028302cd0f035fdf8e7e91f089a4ce94d2195 \
  --local-dir /path/to/Inkling-2-Mini \
  config.json generation_config.json model.safetensors.index.json \
  model-00001-of-00001.safetensors tokenizer.json tokenizer_config.json \
  special_tokens_map.json chat_template.jinja

EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 PYTHONPATH=. \
  python recipes/inkling2/predict.py --model /path/to/Inkling-2-Mini \
    --max-new-tokens 128 "What is the capital of France? Reply with only the city name."
```

The script renders the chat template, passes the prompt to
`InferenceEnginePY.infer_list()`, and reports the Eole timings. Use `--gpu 1`
to select the second visible GPU. If you already have a templated prompt, pass
it directly to `infer_list()` from a script using the same `PredictConfig`.

On NVIDIA GPUs with compute capability 8.9 or newer, BF16 inference with
`EOLE_TORCH_COMPILE=1` and mode `0` or `2` selects packed expert kernels and CUDA
graph replay for decode. The graph warms on the first two decode steps and
captures on the third. New Triton kernel shapes may add first-use latency.
With `EOLE_TORCH_COMPILE=0` the same decoder uses eager PyTorch and dequantizes
only the selected experts. Reduction order can change greedy output across modes.

This recipe supports one unpadded text sequence on one device with
`batch_size=1` and `beam_size=1`. It uses native KV cache precision. Vision,
speculative decoding, persistent prefix caching, and continuous batching
are outside this release's support. Longer inputs need additional GPU memory.

## Convert to an Eole checkpoint

Use conversion if you want to store the Eole-formatted weights and vocabulary
on disk. The converter preserves the source tensor dtypes and packed expert weights.

```bash
eole convert HF --model_dir /path/to/Inkling-2-Mini --output /path/to/Inkling-2-Mini-eole
```

Use the converted directory as `--model` in the command above.
