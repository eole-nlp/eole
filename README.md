# EOLE

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://eole-nlp.github.io/eole)

Eole is an open language modeling toolkit built on PyTorch, originally spun off
from OpenNMT-py. Train, fine-tune, evaluate, and serve encoder, decoder, and
encoder-decoder models in a compact, modular codebase built for experimentation.

Use it for language generation, machine translation, neural translation scoring,
vision and OCR, and speech recognition. Bring supported Hugging Face checkpoints
or train your own architecture.

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
| Evaluate model quality or inference speed | [MMLU](recipes/mmlu/README.md), [model validator](recipes/model-validator/README.md), or [benchmarks](https://github.com/eole-nlp/eole/blob/main/benchmarks/genai/README.md) |

Browse the [full recipe index](recipes/README.md) for more workflows.

## Quickstart: serve a small chat model

Run from the repository root in a Python environment with a compatible CUDA
PyTorch installation and an NVIDIA GPU. See [installation](#installation) for
requirements and optional kernels.

```bash
git clone https://github.com/eole-nlp/eole
cd eole
pip install -e .
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
- PyTorch >= 2.10 and < 2.13, with a build compatible with your GPU and driver.

```bash
pip install -e .
```

Install optional task dependencies as needed:

```bash
pip install -r requirements.opt.txt
```

FlashAttention is optional; the quickstart uses the PyTorch attention backend:

```bash
pip install flash-attn --no-build-isolation
```

Quantization and fast hybrid-attention kernels have additional dependencies;
follow the corresponding recipe. Use `HF_TOKEN` with `--token "$HF_TOKEN"` for
checkpoints requiring Hugging Face authentication. If installation runs out of
memory, reduce build parallelism; `--no-cache-dir` can reduce pip cache usage.

### Docker

[Published images](https://github.com/eole-nlp/eole/pkgs/container/eole) provide a
versioned environment:

```bash
docker run --rm -it --gpus all \
  ghcr.io/eole-nlp/eole:0.6.0-torch2.11.0-ubuntu24.04-cuda13.0
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
This image is the **0.6.0 release**; newer source features such as MTP inference
and REINFORCE require a checkout containing those changes. See [docker/](https://github.com/eole-nlp/eole/tree/main/docker)
for building an image from your checkout. Mount model storage and forward port
5000 when serving from a container.

## Documentation and contributing

- [Full documentation](https://eole-nlp.github.io/eole)
- [Recipes](recipes/README.md) and [release changelog](https://github.com/eole-nlp/eole/blob/main/CHANGELOG.md)
- [Training precision](https://github.com/eole-nlp/eole/blob/main/docs/training-precision.md)
- [Contributing](https://github.com/eole-nlp/eole/blob/main/CONTRIBUTING.md)

Use [Discussions](https://github.com/eole-nlp/eole/discussions) for questions and
feature proposals, and [Issues](https://github.com/eole-nlp/eole/issues) for bugs.
