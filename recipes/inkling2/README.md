# Inkling 2 Mini text and image inference

Eole runs [Thinking Machines’ Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
through the shared transformer decoder and vision encoder. The decoder visits
the central 12 of its 24 physical layers three times. Each of the 48 visits
maintains its own attention and convolution state.

## Download and run

Run these commands from the Eole checkout. The target checkpoint takes about
19 GiB on disk and uses NVFP4 and MXFP8 experts with BF16 dense weights.
The optional DSpark drafter adds about 607 MiB.

```bash
hf download thinkingmachines/Inkling-2-Mini \
  --revision c9d028302cd0f035fdf8e7e91f089a4ce94d2195 \
  --local-dir /path/to/Inkling-2-Mini \
  config.json generation_config.json model.safetensors.index.json \
  model-00001-of-00001.safetensors tokenizer.json tokenizer_config.json \
  special_tokens_map.json chat_template.jinja mtp.safetensors

EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 PYTHONPATH=. \
  python recipes/inkling2/predict.py --model /path/to/Inkling-2-Mini \
    --max-new-tokens 128 "What is the capital of France? Reply with only the city name."
```

Add `--image /path/to/photo.jpg` for an image prompt. Repeat `--image` to
attach multiple images in order.

```bash
EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 PYTHONPATH=. \
  python recipes/inkling2/predict.py --model /path/to/Inkling-2-Mini \
    --image /path/to/photo.jpg "Describe the photo in one sentence."
```

The script renders the chat template, calls `InferenceEnginePY.infer_list()`,
and reports Eole’s timings. Use `--gpu 1` to select the second visible GPU.
For scripting, use the same `PredictConfig`. Set `data_type="image"` and pass
a dictionary with `text` and `images` to `infer_list()` for an image prompt.
The keys of `images` correspond to placeholders such as `{image0}` in the
rendered prompt.

## Try DSpark speculative decoding

Add `--dspark` to either example, or set `dspark=True` in `PredictConfig`.
Eole loads the bundled `mtp.safetensors` and verifies up to seven proposals
per block. Use `--draft-tokens 3` to cap the proposal count at three.
DSpark supports unconstrained greedy decoding for a single sequence.
With compile mode `0` or `2` on NVIDIA SM89 or newer and the Eole CUDA
extension, the target verifier captures full blocks. The first two
blocks warm up the kernels and capture occurs on the third. Shortened
blocks at the output limit execute eagerly. DSpark throughput depends
on draft acceptance and the cost of running several tokens through the
target's routed experts.

## Convert to an Eole checkpoint

Conversion preserves the source tensor dtypes and packed expert weights. It
converts the vision encoder and projector along with the text decoder and
copies the separate `mtp.safetensors` when present.

```bash
eole convert HF --model_dir /path/to/Inkling-2-Mini --output /path/to/Inkling-2-Mini-eole
```

Use the converted directory as `--model` in the examples. If you have an
earlier text-only conversion, convert the original checkpoint into a new
directory to get the vision encoder.

## Runtime details

This recipe runs one unpadded sequence on one device with `batch_size=1`,
`beam_size=1`, and native KV cache precision. Continuous batching and
persistent prefix caching are unavailable. Longer inputs need additional
GPU memory.

On NVIDIA GPUs with compute capability 8.9 or newer, BF16 inference with
the Eole CUDA extension groups prompt tokens by expert on the GPU and
reads NVFP4 and MXFP8 weights directly during prefill. It uses 512-token
chunks by default. The PyTorch fallback dequantizes selected expert
weights and uses 128-token chunks. Set `prefill_chunk_size` in
`PredictConfig` to override the default.

For plain decode, `EOLE_TORCH_COMPILE=1` with mode `0` or `2` enables packed
expert kernels and CUDA graph replay on those GPUs. The graph warms on the
first two decode steps and captures on the third. New Triton kernel shapes
can add first-use latency. Different reduction orders across execution
paths can change greedy output.


On Blackwell, add `--fused-relative-attention` to try the fused KV update
and split relative attention during single-token graph decode. The backend
uses FP32 accumulation across splits and can change greedy output. Fixed
caches above 8,192 positions and multi-token verification use the existing
attention path.


For the optional native Blackwell kernels, install a vLLM build that supports
`vllm.models.inkling2` in a compatible environment, then pass
`--kernel-backend vllm --fused-relative-attention`. This adapter uses vLLM's
NVFP4/MXFP8 expert activation quantization and dense projection kernels inside
Eole's existing decoder. The default backend is `eole`. Numerical differences
can change greedy continuations. The vLLM adapter serves single-token decode,
with the existing Eole kernels handling prefill and speculative blocks.

The Orwell prompt with BF16 vocabulary weights measured 149.4 tok/s on the first
request and 152.0 tok/s on the warmed request on an RTX 5090, with PyTorch 2.15,
Triton 3.9 and vLLM `0.34.1rc1.dev285+g6beda2a3a`. Use the same model and
2,048-token output budget to reproduce the comparison.
