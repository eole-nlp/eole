# Recipes

Start with [direct HF inference](hf/README.md) for one CLI prompt, the
[server quickstart](server/README.md) for an API, or the
[browser workspace](ui/README.md) for an interactive session. Choose a task
below for the model, commands, and expected output.

## Inference, workspace, and serving

| Workflow | Recipe |
|---|---|
| Direct inference from a supported Hugging Face model ID | [HF inference](hf/README.md) |
| Browser Chat, translation, OCR, RAG, and benchmarks | [Local workspace](ui/README.md) |
| Deploy the workspace on a rented GPU with authentication | [RunPod deployment](runpod/README.md) |
| FastAPI server, native inference, OpenAI-style chat, Anthropic-style Messages | [Server](server/README.md) |
| Qwen3.8-27B text inference, MTP verification, and baseline comparison | [Qwen3.8 / MTP](qwen38/README.md) |
| Use Codex with local Qwen | [Codex](codex/README.md) |
| Run concurrent Responses requests and benchmark tool loops | [Continuous serving](serving/README.md) |
| Use Claude Code with local Qwen | [Claude Code](claude-code/README.md) |
| Qwen3.5 text, vision, streaming, and serving | [Qwen3.5](qwen35/README.md) |
| Qwen3.8-27B with a separate speculative drafter | [DFlash2](qwen38/dflash2/README.md) |
| Small Gemma text model | [Gemma 3](gemma3/README.md) |
| Mistral and Mixtral inference | [Mistral](mistral/README.md), [Mixtral](mixtral/README.md) |

## Translation, documents, and speech

| Workflow | Recipe |
|---|---|
| Default translation workspace for text, documents, subtitles and glossaries | [Hy-MT2-7B](translation/README.md) |
| Serve the original Hunyuan-MT-7B checkpoint through the API | [Hunyuan-MT-7B](hunyuan/README.md) |
| NLLB conversion and translation | [NLLB](nllb/README.md) |
| Ask questions about local documents and inspect cited passages | [Knowledge (RAG)](rag/README.md) |
| Image understanding | [Pixtral](pixtral/README.md), [Qwen3.5](qwen35/README.md) |
| OCR and document extraction | [HunyuanOCR](hunyuanocr/README.md), [DeepSeek-OCR](deepseekocr/README.md) |
| Speech recognition and timestamps | [Whisper](whisper/README.md) |

## Training and fine-tuning

| Workflow | Recipe |
|---|---|
| Fine-tune a chat model with LoRA on one GPU | [Qwen LoRA](finetuning/README.md) |
| Train a Transformer on CPU, then scale to WMT on GPU | [English–German translation tutorial](wmt17/README.md) |
| Train a language model from scratch | [WikiText-103](wiki_103/README.md), [FineWeb](fineweb10B/README.md) |
| Train on streamed synthetic parallel text | [NewsPalm](NewsPalm-synthetic/README.md) |
| Scorer-reward fine-tuning (REINFORCE) | [RL](rl/README.md) |

## Evaluation and performance

| Workflow | Recipe |
|---|---|
| Native COMET, KIWI, XCOMET, MetricX; training validation and custom scorers | [Scoring overview](scoring/README.md) |
| Smoke-test conversion, generation, and MMLU | [Model validator](model-validator/README.md) |
| Python code generation graded by executable tests | [LiveCodeBench](livecodebench/README.md) |
| Knowledge benchmark with a small Qwen model | [MMLU](mmlu/README.md) |
| GPT-2 inference and HellaSwag reference results | [GPT-2](gpt2/README.md) |
| Reproduce inference measurements | [Generation benchmarks](https://github.com/eole-nlp/eole/blob/main/benchmarks/genai/README.md) |

## Contributing a recipe

Use a single `README.md` so the documentation build can include the recipe.
Specify the working directory, exact checkpoint, dependencies, expected output,
and limitations. Keep scripts configurable, avoid machine-specific paths, and
state what was actually tested. Link the recipe here and from the main README
when it is a useful starting point. Respect model and dataset licenses.
