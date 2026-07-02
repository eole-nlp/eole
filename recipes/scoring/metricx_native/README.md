# Native EOLE MetricX Scoring

This recipe shows how to use native EOLE scoring with pre-converted
[Google MetricX](https://github.com/google-research/metricx) models hosted on
Hugging Face:

- `EOLE-METRICX` (reference-based)
- `EOLE-METRICX-QE` (reference-free / QE)

Converted MetricX models are represented as EOLE
`transformer_encoder_decoder_scorer` models with template-driven scorer inputs.
Raw Google MetricX Hugging Face repos are not EOLE model layouts; use the hosted
EOLE conversions below, or convert raw upstream checkpoints yourself.

MetricX raw scores are lower-is-better scores clamped to the model's configured
range, normally `0..25`. EOLE preserves the raw MetricX score and does not invert
or normalize it.

## 1) Use hosted MetricX models

The native EOLE MetricX scorers default to a hosted pre-converted fp32 EOLE
model:

See the [EOLE-MetricX Hugging Face collection](https://huggingface.co/collections/eole-nlp/eole-metricx)
for all published MetricX EOLE conversions.

| Model | Use case | Input modes |
| --- | --- | --- |
| `eole-nlp/metricx-24-hybrid-large-v2p6-eole` | Default, practical speed | `reference`, `qe` |
| `eole-nlp/metricx-24-hybrid-xl-v2p6-eole` | Larger/intermediate MetricX-24 model | `reference`, `qe` |
| `eole-nlp/metricx-23-large-v2p0-eole` | MetricX-23 reference-based model | `reference` |

By default, `EOLE-METRICX` and `EOLE-METRICX-QE` both use
`eole-nlp/metricx-24-hybrid-large-v2p6-eole`. MetricX-24 Hybrid supports both
reference and QE modes.

The hosted MetricX EOLE artifacts are fp32 because MetricX uses mT5-style
encoder-decoder scorer models, which can be numerically sensitive under reduced
precision. Use fp32 for stable validation and parity reporting. Use bf16 or fp16
only as an explicit opt-in after validating score drift for your use case.

For public repos, no local conversion is required. For private repos,
authenticate with `hf auth login` or set `HF_TOKEN`. For direct `eole predict`,
`--hf_token` can also be used.

## 2) Use MetricX metrics during training

Use the default hosted `EOLE-METRICX` model in reference mode with:

```yaml
valid_metrics: ["EOLE-METRICX"]
metricx_batch_size: 8
metricx_device: cuda
metricx_compute_dtype: fp32
```

`EOLE-METRICX` forces reference mode and requires references in the validation
corpus.

Use the same hosted default model in reference-free QE mode with:

```yaml
valid_metrics: ["EOLE-METRICX-QE"]
metricx_batch_size: 8
metricx_device: cuda
metricx_compute_dtype: fp32
```

`EOLE-METRICX-QE` forces reference-free mode and scores from source+hypothesis
only. The configured converted model must declare QE support in
`supported_input_modes`.

Set `metricx_model` only when you want to override the default hosted EOLE
model, for example to use the hosted XL conversion or a local conversion:

```yaml
valid_metrics: ["EOLE-METRICX"]
metricx_model: eole-nlp/metricx-24-hybrid-xl-v2p6-eole
metricx_batch_size: 4
metricx_device: cuda
metricx_compute_dtype: fp32
```

`metricx_compute_dtype` accepts `fp32`, `fp16`, or `bf16`. MetricX validation
defaults to `fp32` for stability. For CUDA speed/memory, opt in explicitly with
`metricx_compute_dtype: bf16` after validating score drift.

EOLE line-oriented files may represent embedded newlines as the internal newline
sentinel (`｟newline｠`). By default, EOLE-METRICX restores this sentinel to a real
newline before tokenization so MetricX scores the logical text rather than the
file serialization. Set this to `false` only when you intentionally want to score
the sentinel characters literally, for example in literal-file parity tests:

```yaml
metricx_replace_newline_sentinel: false
```

MetricX validation metrics are lower-is-better and can be used for early
stopping:

```yaml
valid_metrics: ["EOLE-METRICX"]
early_stopping: 5
early_stopping_criteria: EOLE-METRICX
```

To run on CPU:

```yaml
valid_metrics: ["EOLE-METRICX"]
metricx_batch_size: 1
metricx_device: cpu
metricx_compute_dtype: fp32
```

On Apple Silicon / MPS, also use `metricx_device: mps` and keep fp32 unless you
have validated a reduced-precision path for your model:

```yaml
valid_metrics: ["EOLE-METRICX"]
metricx_batch_size: 1
metricx_device: mps
metricx_compute_dtype: fp32
```

## 3) Direct CLI scoring with hosted models

For `transformer_encoder_decoder_scorer` models, use:

- `--src`: source sentences
- `--tgt`: MT hypotheses
- `--ref`: references (optional; when provided, reference mode is used)

Direct `eole predict` scoring uses `--compute_dtype`. The training-validation
setting `metricx_compute_dtype` is only read by validation metrics configured
with `valid_metrics`.

Reference-based scoring with the default hosted model:

```bash
eole predict \
  --model_path eole-nlp/metricx-24-hybrid-large-v2p6-eole \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/metricx-scores.txt \
  --with_score \
  --batch_size 8 \
  --compute_dtype fp32
```

Reference-free / QE scoring omits `--ref`:

```bash
eole predict \
  --model_path eole-nlp/metricx-24-hybrid-large-v2p6-eole \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --output /path/to/metricx-qe-scores.txt \
  --with_score \
  --batch_size 8 \
  --compute_dtype fp32
```

Use the hosted XL conversion when you want the larger MetricX-24 model:

```bash
eole predict \
  --model_path eole-nlp/metricx-24-hybrid-xl-v2p6-eole \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/metricx-xl-scores.txt \
  --with_score \
  --batch_size 4 \
  --compute_dtype fp32
```

For literal-sentinel parity on files containing `｟newline｠`, pass
`--metricx_replace_newline_sentinel false` to `eole predict`. For semantic
parity with native MetricX, keep the EOLE default and normalize the native input
to real newlines.

By default, `eole predict --with_score` writes one segment score per input line:

```text
0.18036818504333496
0.5929612517356873
```

To emit a single system-level score, use `--score_level system`:

```bash
eole predict \
  --model_path eole-nlp/metricx-24-hybrid-large-v2p6-eole \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/metricx-system-score.txt \
  --with_score \
  --score_level system \
  --compute_dtype fp32
```

`score_level: system` writes one numeric line containing the arithmetic mean of
the segment scores:

```text
0.3866647183895111
```

## 4) Convert your own MetricX models

Convert raw Google MetricX checkpoints when you need a model that is not already
published as a pre-converted EOLE repo, or when you need a local/offline artifact.

```bash
export EOLE_MODEL_DIR=~/Development/Models/eole

eole convert MetricX \
  --model google/metricx-24-hybrid-large-v2p6 \
  --dtype fp32 \
  --output "$EOLE_MODEL_DIR/metricx/metricx-24-hybrid-large-v2p6-eole"
```

MetricX conversion downloads the MetricX checkpoint and the matching
`google/mt5-{large,xl,xxl}` tokenizer assets. The converted model directory
contains the EOLE config, safetensors weights, vocabulary, and bundled
SentencePiece tokenizer.

MetricX-23 can also be converted:

```bash
eole convert MetricX \
  --model google/metricx-23-large-v2p0 \
  --dtype fp32 \
  --output "$EOLE_MODEL_DIR/metricx/metricx-23-large-v2p0-eole"
```

Google also publishes bfloat16 MetricX-24 variants. To preserve bf16 weights in
the converted EOLE artifact, convert with `--dtype bf16`:

```bash
eole convert MetricX \
  --model google/metricx-24-hybrid-large-v2p6-bfloat16 \
  --dtype bf16 \
  --output "$EOLE_MODEL_DIR/metricx/metricx-24-hybrid-large-v2p6-bfloat16-eole"
```

For gated or private models, pass a Hugging Face token or authenticate with
`hf auth login`:

```bash
eole convert MetricX \
  --model google/metricx-24-hybrid-xxl-v2p6 \
  --token "$HF_TOKEN" \
  --dtype fp32 \
  --output "$EOLE_MODEL_DIR/metricx/metricx-24-hybrid-xxl-v2p6-eole"
```

Use a local conversion by setting `metricx_model` in training config or by
passing the local directory to `eole predict --model_path`.

## 5) MetricX-23 vs MetricX-24

MetricX version-specific input formatting is stored in the converted model config
as `input_templates`.

MetricX-24 reference template:

```text
source: {src} candidate: {tgt} reference: {ref}
```

MetricX-24 QE template:

```text
source: {src} candidate: {tgt}
```

MetricX-23 reference template:

```text
candidate: {tgt} reference: {ref}
```

MetricX-23 QE template:

```text
candidate: {tgt} source: {src}
```

EOLE selects reference mode when `--ref` is provided and QE mode when `--ref` is
absent. The selected mode is validated against the converted model's
`supported_input_modes`.

## 6) Interpreting MetricX scores

MetricX scores are lower-is-better. EOLE reports the raw MetricX score, so lower
validation scores indicate better translations.

When combining MetricX with higher-is-better metrics such as COMET, normalize and
invert MetricX in the blending layer rather than changing the raw MetricX scorer.
For example, a unified metric can map MetricX `0..25` to a higher-is-better
`0..1` range with:

```text
normalized_metricx = 1 - clamp(metricx, 0, 25) / 25
```

## 7) Parity notes

There is not currently a reusable MetricX parity harness in this recipe, unlike
the COMET parity harness in `recipes/scoring/comet_native/`.

The hosted MetricX EOLE artifacts are fp32 reference conversions. Reduced
precision conversions should be validated separately before use in parity or
reporting workflows.
