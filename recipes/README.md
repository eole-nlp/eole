# Recipes

Choose a task below. Each recipe documents its working directory, model/data
requirements, commands, and expected output. Model weights and datasets are
separate downloads; check their licenses before use.

See the [README recipe audit](../docs/recipe-audit.md) for what was checked,
fixed, and actually executed on the current checkout.

## Generation and serving

| Workflow | Recipe |
|---|---|
| Direct inference from a supported Hugging Face model ID | [HF inference](hf/README.md) |
| FastAPI server, native inference, OpenAI-style chat, Anthropic-style Messages | [Server](server/README.md) |
| Qwen3.8-27B text inference, MTP verification, and baseline comparison | [Qwen3.8 / MTP](qwen38/README.md) |
| Claude Code CLI backed by local Qwen3.8 | [Claude Code](claude-code/README.md) |
| Qwen3.5 text, vision, streaming, and serving | [Qwen3.5](qwen35/README.md) |
| Llama inference and fine-tuning | [Llama2](llama2/README.md), [Llama3](llama3/README.md) |
| Mistral and Mixtral inference | [Mistral](mistral/README.md), [Mixtral](mixtral/README.md) |

## Translation and training

| Workflow | Recipe |
|---|---|
| EuroLLM translator web interface | [EuroLLM](eurollm/README.md) |
| Train an encoder-decoder translation model | [WMT17 English–German](wmt17/README.md) |
| NLLB conversion and translation | [NLLB](nllb/README.md) |
| Translation with Tower and Llama | [Tower / Mistral](wmt22_with_TowerInstruct-Mistral/README.md), [Tower / Llama2](wmt22_with_TowerInstruct-llama2/README.md), [Llama3.1](wmt22_with_llama3.1/README.md) |
| Language-model training | [GPT-2](gpt2/README.md), [WikiText-103](wiki_103/README.md), [FineWeb](fineweb10B/README.md) |
| Synthetic data | [NewsPalm](NewsPalm-synthetic/README.md) |
| Scorer-reward fine-tuning (REINFORCE) | [RL](rl/README.md) |

## Multimodal models

| Workflow | Recipe |
|---|---|
| Image understanding | [Pixtral](pixtral/README.md), [Qwen3.5](qwen35/README.md) |
| OCR and document extraction | [HunyuanOCR](hunyuanocr/README.md), [DeepSeek-OCR](deepseekocr/README.md) |
| Speech recognition and timestamps | [Whisper](whisper/README.md) |

## Evaluation and performance

| Workflow | Recipe |
|---|---|
| Native COMET, KIWI, XCOMET, MetricX; training validation and custom scorers | [Scoring overview](scoring/README.md) |
| Smoke-test conversion, generation, and MMLU | [Model validator](model-validator/README.md) |
| Python code generation graded by executable tests | [LiveCodeBench](livecodebench/README.md) |
| Knowledge benchmark | [MMLU](mmlu/README.md) |
| Reproduce inference measurements | [Generation benchmarks](https://github.com/eole-nlp/eole/blob/main/benchmarks/genai/README.md) |

## Contributing a recipe

Use a single `README.md` so the documentation build can include the recipe.
Specify the working directory, exact checkpoint, dependencies, expected output,
and limitations. Keep scripts configurable, avoid machine-specific paths, and
state what was actually tested. Link the recipe here and from the main README
when it is a useful starting point. Respect model and dataset licenses.
