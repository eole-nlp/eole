# EOLE

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://eole-nlp.github.io/eole)

Eole is an open language modeling toolkit built on PyTorch, originally spun off
from OpenNMT-py. Train, fine-tune, evaluate, and serve encoder, decoder, and
encoder-decoder models in a compact, modular codebase built for experimentation.

Use it for language generation, machine translation, neural translation scoring,
vision and OCR, and speech recognition. Bring supported Hugging Face checkpoints
or train your own architecture.

## Latest: 0.6.2

This release fixes Qwen RoPE conversion and GGUF MTP/tokenizer handling, adds
compressed-tensors INT4 support and the out-of-tree vLLM GGUF backend, and
provides a LiveCodeBench evaluation recipe. Start with the
[Qwen3.8/MTP recipe](recipes/qwen38/README.md) using the recommended RedHatAI INT4
checkpoint, or the [LiveCodeBench recipe](recipes/livecodebench/README.md).
See the [changelog](CHANGELOG.md#062) for details, conversion migration guidance,
and current limitations.

## Choose your workflow

| I want to… | Start here |
|---|---|
| Run a supported HF model without a separate conversion step | [Direct HF inference](recipes/hf/README.md) |
| Run a local LLM with a chat API | [Model server](recipes/server/README.md) |
| Try Qwen3.8-27B with MTP speculative decoding | [Qwen3.8 and MTP](recipes/qwen38/README.md) |
| Use Claude Code with a locally served Qwen model | [Claude Code endpoint](recipes/claude-code/README.md) |
| Try Qwen3.5 text and image inputs | [Qwen3.5](recipes/qwen35/README.md) |
| Translate through a web interface | [EuroLLM](recipes/eurollm/README.md) |
| Train a translation model | [WMT17](recipes/wmt17/README.md) |
| Translate with a pretrained multilingual model | [NLLB](recipes/nllb/README.md) |
| Generate text with Mistral | [Mistral](recipes/mistral/README.md) |
| Fine-tune an LLM | [Llama2 LoRA](recipes/llama2/README.md) |
| Fine-tune with a scorer reward | [REINFORCE](recipes/rl/README.md) |
| Score translations or use neural metrics during training | [COMET, KIWI, XCOMET, and MetricX](recipes/scoring/README.md) |
| Extract text from images and documents | [HunyuanOCR](recipes/hunyuanocr/README.md) or [DeepSeek-OCR](recipes/deepseekocr/README.md) |
| Transcribe audio | [Whisper](recipes/whisper/README.md) |
| Train a language model from scratch | [WikiText-103](recipes/wiki_103/README.md) or [FineWeb](recipes/fineweb10B/README.md) |
| Evaluate model quality or inference speed | [LiveCodeBench coding](recipes/livecodebench/README.md), [MMLU](recipes/mmlu/README.md), [model validator](recipes/model-validator/README.md), or [benchmarks](https://github.com/eole-nlp/eole/blob/main/benchmarks/genai/README.md) |

Browse the [full recipe index](recipes/README.md) for more workflows.

## Quickstart: serve a small chat model

Run from the repository root in a Python environment with a compatible CUDA
PyTorch installation and an NVIDIA GPU. See [installation](#installation) for
requirements and optional kernels.

```bash
git clone https://github.com/eole-nlp/eole
cd eole
python -m pip install -e . --no-build-isolation
export EOLE_MODEL_DIR="$PWD/models"
mkdir -p "$EOLE_MODEL_DIR"
eole convert HF --model_dir Qwen/Qwen3.5-0.8B \
  --output "$EOLE_MODEL_DIR/qwen3.5-0.8B"
eole serve -c recipes/server/serve.example.yaml --host 127.0.0.1
```

In a second terminal:

```bash
curl --fail-with-body http://127.0.0.1:5000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-0.8B","messages":[{"role":"user","content":"Explain machine translation in one sentence."}],"max_tokens":128,"temperature":0}'
```

The response contains the generated text in `choices[0].message.content`.
Interactive API documentation is available at `http://127.0.0.1:5000/docs`.
The [server recipe](recipes/server/README.md) explains streaming and both API formats.

## Features

- **Training and fine-tuning:** Transformer encoder, decoder, and encoder-decoder
  architectures, RNN encoder-decoder models, LoRA, quantized fine-tuning, dynamic
  data transforms, tensor parallelism, and mixed or pure BF16 training.
- **Efficient inference:** FlashAttention KV caching, CUDA and Triton kernels,
  fused projections, quantized dense and MoE inference, chunked prefill, prefix
  caching, and optional `torch.compile` / CUDA graphs. Availability depends on
  model, hardware, installed kernels, and configuration.
- **Multi-token prediction:** Auxiliary MTP heads for decoder-only training and
  native speculative decoding for compatible checkpoints, including Qwen3.8-27B. Current inference
  support is text-only, single-sequence, deterministic greedy decoding;
  unsupported configurations fall back to ordinary decoding.
- **Serving:** FastAPI server with native inference, OpenAI-style chat completions,
  and Anthropic-style Messages endpoints, including tool-use handling.
- **Evaluation:** BLEU, chrF, TER, perplexity, native COMET/KIWI/XCOMET and MetricX
  scorers, custom scorer modules, and scorer-based early stopping.
- **Reinforcement learning:** On-policy REINFORCE with registered scorer rewards,
  batch-mean baselines, and an optional frozen reference-model KL penalty.
  DPO, GRPO, and PPO are planned, not implemented.

## Supported Hugging Face model families

| Family / task | Examples and notes |
|---|---|
| Qwen | Qwen, Qwen2, Qwen3 (including MoE); Qwen3.5 text/vision; Qwen3.8-27B with MTP |
| Gemma | Gemma3 text/image and Gemma4 support |
| Mistral | Mistral, Mixtral, Mathstral, and supported multimodal / Ministral variants |
| Llama and Phi | Llama3.x and Phi2/3 |
| Translation | EuroLLM, Hunyuan-MT, NLLB, and Tower recipes |
| OCR and vision | HunyuanOCR, DeepSeek-OCR, Pixtral, and supported vision-language checkpoints |
| Speech | Whisper |
| Translation scoring | COMET, COMET-KIWI, XCOMET, MetricX, and MetricX-QE |

Support is architecture-specific; this table does not imply every checkpoint,
quantization format, or modality in a family is supported. Consult the relevant
recipe and converter for the exact configuration. Supported HF checkpoints can
also be used [directly for inference](recipes/hf/README.md); conversion creates a reusable Eole artifact.

## Performance and reproducibility

See the [inference benchmarks](https://github.com/eole-nlp/eole/blob/main/benchmarks/genai/README.md) for example scripts and historical results (with incomplete environment metadata), and the [compilation guide](https://github.com/eole-nlp/eole/blob/main/TORCHCOMPILE_README.md) for
`torch.compile` configuration. Separate compilation/warmup from repeated warm
runs. Throughput depends on model, precision, prompt length, generation length,
and batch size; benchmark results are not a universal ranking of engines.

The [MTP recipe](recipes/qwen38/README.md) includes a baseline comparison and
acceptance diagnostics. GPU speedup and production BF16/quantized token parity
must be measured on the chosen checkpoint.

## Installation

### From source

- Python >= 3.11 (current CI uses Python 3.12).
- PyTorch >= 2.10 and < 2.13, with a CUDA build compatible with your GPU and driver.
- To compile CUDA extensions: the CUDA toolkit (`nvcc`), a compatible C++ compiler,
  and the build dependencies below. The CUDA runtime bundled with PyTorch alone
  does not provide `nvcc`.

Install CUDA-enabled PyTorch first in your chosen environment, then run from the
repository root:

```bash
python -m pip install "setuptools<69" wheel packaging ninja psutil
python -c 'import torch; print(torch.__version__, torch.version.cuda); print("GPU available:", torch.cuda.is_available())'
nvcc --version
MAX_JOBS=2 python -m pip install -e . --no-build-isolation
```

`setup.py` builds Eole's `eole._ops` CUDA extension, including normalization,
rotary embeddings, activations, and Marlin quantization kernels. It only enables
the extension when PyTorch is installed and `torch.cuda.is_available()` is true.
`--no-build-isolation` lets the build use that PyTorch installation. `MAX_JOBS`
limits compilation memory use; increase it if your machine has sufficient RAM.

### Build kernels for running directly from a clone

If your environment already contains Eole's Python dependencies and you prefer
to run scripts without installing the package, build the extension in place:

```bash
# Run from the repository root, with CUDA-enabled PyTorch and build tools above.
MAX_JOBS=2 python setup.py build_ext --inplace
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m eole.bin.main --help
```

This builds `eole/_ops*.so`; it does not install Python dependencies or the
`eole` console command. Use `python -m eole.bin.main` in place of `eole` in recipes.
Rebuild after changing PyTorch, CUDA, Python, or extension sources. The current
build targets the visible GPU's compute capability; it is not a universal wheel
for every GPU architecture. A successful command on a CPU-only environment may
skip the extension entirely. Check it in the GPU environment:

```bash
python -c 'import torch; from eole import _ops; assert torch.cuda.is_available(); print("Eole CUDA extension loaded")'
```

### FlashAttention and hybrid-attention kernels

Install these separately in the same Python environment. FlashAttention is
optional for the small-model quickstart, which uses `self_attn_backend: pytorch`.
For recipes selecting `self_attn_backend: flash`, follow the
[FlashAttention installation requirements](https://github.com/Dao-AILab/flash-attention#installation-and-features):

```bash
MAX_JOBS=2 python -m pip install flash-attn --no-build-isolation
python -c 'from flash_attn import flash_attn_func, flash_attn_with_kvcache; print("FlashAttention interfaces loaded")'
```

Use a release or wheel compatible with your GPU, Python, PyTorch, and CUDA
versions. The Qwen3.8 measurements used FlashAttention 2.8.3; installing a newer
package alone does not reproduce that environment.

For Qwen3.5/Qwen3.8 fast gated-delta computation, install
[FLA kernels (`fla-core`)](https://pypi.org/project/fla-core/):

```bash
python -m pip install fla-core
python -c 'from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule; print("FLA gated-delta interfaces loaded")'
```

Eole also supports the separate CUDA convolution package:

```bash
MAX_JOBS=2 python -m pip install causal-conv1d --no-build-isolation
```

When `causal-conv1d` is absent, Eole can use FLA's convolution implementation;
without either, it has a PyTorch fallback. FLA uses Triton kernels compiled at
runtime. Import checks confirm availability, not kernel execution or backend
selection; inspect the recipe's backend diagnostics during a GPU run. INT4
Marlin recipes require Eole's compiled CUDA extension.

Install other optional task dependencies as needed:

```bash
python -m pip install -r requirements.opt.txt
```

Use `HF_TOKEN` with `--token "$HF_TOKEN"` for checkpoints requiring Hugging Face
authentication. See individual recipes for model-specific requirements.

### Docker

Requires Docker, a compatible NVIDIA driver, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Use a [published image](https://github.com/eole-nlp/eole/pkgs/container/eole) or
build an image from your checkout. The [Docker guide](docs/docker.md) covers GPU
prerequisites, local builds, CUDA kernels, model/data mounts, YAML inference and
training, and serving an API endpoint.

Release images contain that release's features; use a current checkout image for
newer features such as MTP. Eole's CUDA extension must be built with GPU access,
which a normal Docker image build does not provide.

## Documentation and contributing

- [Full documentation](https://eole-nlp.github.io/eole)
- [Recipes](recipes/README.md) and [release changelog](https://github.com/eole-nlp/eole/blob/main/CHANGELOG.md)
- [Training precision](https://github.com/eole-nlp/eole/blob/main/docs/training-precision.md)
- [Contributing](https://github.com/eole-nlp/eole/blob/main/CONTRIBUTING.md)

Use [Discussions](https://github.com/eole-nlp/eole/discussions) for questions and
feature proposals, and [Issues](https://github.com/eole-nlp/eole/issues) for bugs.
