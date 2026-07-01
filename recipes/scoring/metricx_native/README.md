# Native EOLE MetricX Scoring

This recipe shows how to use native EOLE scoring with converted
[Google MetricX](https://github.com/google-research/metricx) models:

- `EOLE-METRICX` (reference-based)
- `EOLE-METRICX-QE` (reference-free / QE)

Converted MetricX models are represented as EOLE
`transformer_encoder_decoder_scorer` models with template-driven scorer inputs.
Direct inference from Google Hugging Face model IDs is not supported here;
convert the MetricX checkpoint first.

MetricX raw scores are lower-is-better scores clamped to the model's configured
range, normally `0..25`. EOLE preserves the raw MetricX score and does not invert
or normalize it.

## 1) Convert MetricX models to EOLE artifacts

```bash
export EOLE_MODEL_DIR=~/Development/Models/eole

eole convert MetricX \
  --model google/metricx-24-hybrid-large-v2p6 \
  --output "$EOLE_MODEL_DIR/metricx/google--metricx-24-hybrid-large-v2p6"
```

MetricX conversion downloads the MetricX checkpoint and the matching
`google/mt5-{large,xl,xxl}` tokenizer assets. The converted model directory
contains the EOLE config, safetensors weights, vocabulary, and bundled
SentencePiece tokenizer.

MetricX-23 can also be converted:

```bash
eole convert MetricX \
  --model google/metricx-23-large-v2p0 \
  --output "$EOLE_MODEL_DIR/metricx/google--metricx-23-large-v2p0"
```

Google also publishes a bfloat16 MetricX-24 variant. To preserve bf16 weights in
the converted EOLE artifact, convert with `--dtype bf16`:

```bash
eole convert MetricX \
  --model google/metricx-24-hybrid-large-v2p6-bfloat16 \
  --dtype bf16 \
  --output "$EOLE_MODEL_DIR/metricx/google--metricx-24-hybrid-large-v2p6-bfloat16"
```

You can choose the conversion precision independently of scoring runtime
precision. Validation metrics configured with `valid_metrics` use
`metricx_compute_dtype`; direct `eole predict` scoring uses the regular
inference dtype knob, `--compute_dtype`. MetricX validation defaults to `fp32`
for stability. For CUDA speed/memory, opt in explicitly with
`metricx_compute_dtype: bf16` for validation scoring or `--compute_dtype bf16`
for CLI scoring.

For gated or private models, pass a Hugging Face token:

```bash
eole convert MetricX \
  --model google/metricx-24-hybrid-large-v2p6 \
  --token "$HF_TOKEN" \
  --output "$EOLE_MODEL_DIR/metricx/google--metricx-24-hybrid-large-v2p6"
```

## 2) Use `EOLE-METRICX` in a training config

Set in your train YAML:

```yaml
valid_metrics: ["EOLE-METRICX"]
metricx_model: "${EOLE_MODEL_DIR}/metricx/google--metricx-24-hybrid-large-v2p6"
metricx_batch_size: 8
metricx_device: cuda
metricx_compute_dtype: fp32
```

`metricx_compute_dtype` accepts `fp32`, `fp16`, or `bf16`. Use `fp32` for stable
validation and parity reporting. Use `bf16` explicitly on CUDA when speed/memory
matters.

EOLE line-oriented files may represent embedded newlines as the internal
newline sentinel (`｟newline｠`). By default, EOLE-METRICX restores this sentinel
to a real newline before tokenization so MetricX scores the logical text rather
than the file serialization. Set this to `false` only when you intentionally want
to score the sentinel characters literally, for example in literal-file parity
tests:

```yaml
metricx_replace_newline_sentinel: false
```

CUDA bf16 opt-in example:

```yaml
valid_metrics: ["EOLE-METRICX"]
metricx_model: "${EOLE_MODEL_DIR}/metricx/google--metricx-24-hybrid-large-v2p6-bfloat16"
metricx_batch_size: 8
metricx_device: cuda
metricx_compute_dtype: bf16
```

`EOLE-METRICX` forces reference mode and requires references in the validation
corpus.

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
metricx_model: "${EOLE_MODEL_DIR}/metricx/google--metricx-24-hybrid-large-v2p6"
metricx_batch_size: 1
metricx_device: cpu
metricx_compute_dtype: fp32
```

The CPU validation scorer loads MetricX in `fp32` to avoid reduced-precision CPU
numerical issues.

On Apple Silicon / MPS, also use `metricx_device: mps`. The native MetricX
validation scorer loads MetricX in `fp32` on MPS because the `fp16` path can
produce `nan` scores for this mT5-style scorer. MPS autocast to `fp16` showed the
same issue in local smoke tests. The bf16 MetricX variant is finite on MPS, but
with lower precision than fp32.

```yaml
valid_metrics: ["EOLE-METRICX"]
metricx_model: "${EOLE_MODEL_DIR}/metricx/google--metricx-24-hybrid-large-v2p6"
metricx_batch_size: 1
metricx_device: mps
metricx_compute_dtype: fp32
```

## 3) Use `EOLE-METRICX-QE` in a training config

Set in your train YAML:

```yaml
valid_metrics: ["EOLE-METRICX-QE"]
metricx_model: "${EOLE_MODEL_DIR}/metricx/google--metricx-24-hybrid-large-v2p6"
metricx_batch_size: 8
metricx_device: cuda
metricx_compute_dtype: fp32
```

`EOLE-METRICX-QE` forces reference-free mode and scores from source+hypothesis
only. The configured converted model must declare QE support in
`supported_input_modes`.

```yaml
valid_metrics: ["EOLE-METRICX-QE"]
early_stopping: 5
early_stopping_criteria: EOLE-METRICX-QE
```

MetricX-24 hybrid converted artifacts support both reference and QE modes. Some
MetricX-23 artifacts may support only the mode they were converted for.

## 4) Direct CLI scoring with `eole predict`

For `transformer_encoder_decoder_scorer` models, use:

- `--src`: source sentences
- `--tgt`: MT hypotheses
- `--ref`: references (optional; when provided, reference mode is used)

Direct `eole predict` scoring uses `--compute_dtype`. The training-validation
setting `metricx_compute_dtype` is only read by validation metrics configured
with `valid_metrics`.

Reference-based scoring:

```bash
eole predict \
  --model_path "$EOLE_MODEL_DIR/metricx/google--metricx-24-hybrid-large-v2p6" \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/metricx-scores.txt \
  --with_score \
  --batch_size 8 \
  --compute_dtype fp32
```

Reference-free / QE scoring:

```bash
eole predict \
  --model_path "$EOLE_MODEL_DIR/metricx/google--metricx-24-hybrid-large-v2p6" \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --output /path/to/metricx-qe-scores.txt \
  --with_score \
  --batch_size 8 \
  --compute_dtype fp32
```

CUDA bf16 CLI opt-in example:

```bash
eole predict \
  --model_path "$EOLE_MODEL_DIR/metricx/google--metricx-24-hybrid-large-v2p6-bfloat16" \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/metricx-bf16-scores.txt \
  --with_score \
  --batch_size 8 \
  --compute_dtype bf16
```

For literal-sentinel parity on files containing `｟newline｠`, pass
`--metricx_replace_newline_sentinel false` to `eole predict`. For semantic
parity with native MetricX, keep the EOLE default and normalize the native input
to real newlines.

On CPU or MPS, prefer `fp32`:

```bash
eole predict \
  --model_path "$EOLE_MODEL_DIR/metricx/google--metricx-24-hybrid-large-v2p6" \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/metricx-cpu-scores.txt \
  --with_score \
  --batch_size 1 \
  --compute_dtype fp32
```

MPS example:

```bash
eole predict \
  --model_path "$EOLE_MODEL_DIR/metricx/google--metricx-24-hybrid-large-v2p6" \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/metricx-mps-scores.txt \
  --with_score \
  --batch_size 1 \
  --compute_dtype fp32
```

By default, `eole predict --with_score` writes one segment score per input line:

```text
0.18036818504333496
0.5929612517356873
```

To emit a single system-level score, use `--score_level system`:

```bash
eole predict \
  --model_path "$EOLE_MODEL_DIR/metricx/google--metricx-24-hybrid-large-v2p6" \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/metricx-system-score.txt \
  --with_score \
  --score_level system
```

`score_level: system` writes one numeric line containing the arithmetic mean of
the segment scores:

```text
0.3866647183895111
```

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

Direct forward parity against Hugging Face MetricX has been checked manually for
converted `fp16` EOLE artifacts. Expected differences against an upstream `fp32`
forward pass are small, around `2e-4` mean absolute error on smoke examples.
