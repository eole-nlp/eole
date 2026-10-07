# README recipe audit — 2026-10-07

Scope: every recipe in the main README workflow table (19 recipe directories),
the nested scoring examples, the Llama3 YAML referenced by MMLU, and the linked
`benchmarks/genai` examples. This is a source/configuration audit, not a claim
that every model and full training run was executed on the current checkout.

| Entry | Findings / changes | Validation and remaining limits |
|---|---|---|
| Direct HF inference | Current small-Qwen example and explicit input/output paths | PredictConfig checked; generation not rerun |
| Model server | Small-Qwen conversion and YAML paths agree | Server model config checked; small-model generation not rerun |
| Qwen3.8 / MTP | Exact Frozenlock INT4 conversion documented; native-head check retained | Existing RTX 5090 baseline/MTP runs; conversion not rerun; greedy output parity unresolved |
| Claude Code | Explicit model ID, output budget, and writable CLI config directory | Live text/tool/API streaming checks and separate Read/Write/Bash sessions passed previously |
| Qwen3.5 | Conversion path matches the bundled text/image scripts; large INT4 server is a separate example | Image runner config and fixtures, server config checked; generation not rerun |
| EuroLLM | Explicit GPU/BF16/context defaults and Gradio dependencies | Server model config checked; translator UI and generation not rerun |
| WMT17 | Fixed YAML name and preparation typo; prediction now consumes raw test input with saved transforms and GPU ranks | Five training YAMLs checked; full corpus preparation/training not rerun; old timings/results labelled historical |
| NLLB | Correctly labelled pretrained translation; separate HF/onmt conversion directories | Both inference configs checked; translation not rerun |
| Mistral | New BF16 prediction YAML matches converted base model; AWQ example identified as separate | Both inference configs checked; shell syntax checked; generation not rerun |
| Llama2 | NF4 dependency, access requirements, hardware and inherited training tokenizer explained | Training and both inference YAMLs checked; fine-tuning and two-GPU run not rerun |
| REINFORCE | Explicit model/data template and implemented/planned algorithm status | Training YAML checked; RL training not rerun |
| Scoring | Input placeholders made schema-valid; training fragments identified; legacy custom-estimator caveats clarified | Native/custom inference and training schemas checked; fragments checked within complete configs; scorer parity not rerun |
| HunyuanOCR | Root-relative script command; removed unnecessary manual config edit | Script PredictConfig and bundled image paths checked; GPU OCR not rerun |
| DeepSeek-OCR | Root-relative commands; PDF helper takes input/output arguments and YAML instead of local hard-coded paths; batch size one | Script/YAML schemas, fixture paths, PDF CLI help and syntax checked; PDF dependencies and GPU OCR not executed |
| Whisper | Default model path agrees with conversion; evaluation directory command corrected | Prediction/fine-tuning YAMLs and evaluation fragment checked; ASR/finetuning not rerun |
| WikiText-103 | Current HF CLI and pinned dataset namespace; consistent training/inference paths; raw-text inference; sample-output path enabled | Training/prediction schemas, shell syntax, tiny Parquet line-separation fixture checked; full download/training not rerun |
| FineWeb 10BT | Added complete preparation/training commands; integer split size, Parquet-only inputs, exact-boundary split fix and file close | Training schema and tiny Parquet split/rerun fixtures checked; full download/training not rerun |
| MMLU | Referenced config now uses checkpoint directory instead of legacy model.pt; required data layout and generated-answer method explained | Referenced PredictConfig and bundled CSV layout checked; full evaluation not rerun |
| Model validator | Quantization applies only to flagged entries; removed architecture name used as invalid HF ID; accurate smoke-test scope | Shell syntax and mock CLI dispatch for plain/NF4 entries passed; real model sweep not rerun |
| Generation benchmarks | Historical timing claims given explicit limitations; CT2 expands model-storage variable | Python syntax checked; external engines and historical measurements not rerun |

Validation completed: 39 configuration schema checks, including six fragments
merged with required parent fields; four server model configurations; three
image runner configurations and fixture paths; helper fixture checks; modified
shell/Python syntax and formatting; local Markdown links and whitespace.
The CI entry point `python eole/tests/test_recipes.py recipes` also passes.
The Qwen3.8 validation-runner YAML uses its own shared schema, rather than being
interpreted as a prediction config.

Schema validation does not check whether external weights/datasets are present,
whether optional GPU kernels run, or whether model outputs are correct.

GPU validation available so far concerns Qwen3.8 and its API/Claude Code route.
The MTP output discrepancy remains a separate investigation; this recipe audit
does not resolve or reclassify it.
