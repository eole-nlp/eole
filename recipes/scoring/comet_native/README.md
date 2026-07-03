# Native EOLE COMET Scoring

This recipe shows how to use native EOLE scoring with pre-converted
[Unbabel COMET](https://github.com/Unbabel/COMET) models hosted on Hugging Face:

- `EOLE-COMET` (reference-based)
- `EOLE-COMET-KIWI` (reference-free)
- `EOLE-XCOMET` (reference-optional, xCOMET scalar score with explainable spans validated internally)

Converted COMET models are represented as EOLE `transformer_encoder_scorer`
models with the COMET-specific `scoring_type: comet` specialization. Raw
Unbabel COMET Hugging Face repos are not EOLE model layouts; use the hosted
EOLE conversions below, or convert raw upstream checkpoints yourself.

## 1) Use hosted COMET models

The native EOLE COMET scorers default to hosted pre-converted EOLE models:

See the [EOLE-COMET Hugging Face collection](https://huggingface.co/collections/eole-nlp/eole-comet)
for all published COMET, COMET-KIWI, and XCOMET EOLE conversions.

| Scorer | Default model | Reference required |
| --- | --- | --- |
| `EOLE-COMET` | `eole-nlp/wmt22-comet-da-eole-fp16` | Yes |
| `EOLE-COMET-KIWI` | `eole-nlp/wmt23-cometkiwi-da-xl-eole-fp16` | No |
| `EOLE-XCOMET` | `eole-nlp/xcomet-xl-eole-fp16` | Optional |

For public repos, no local conversion is required. For gated/private repos,
authenticate with `hf auth login` or set `HF_TOKEN`. For direct `eole predict`,
`--hf_token` can also be used.

## 2) Use COMET metrics during training

Use the default hosted `EOLE-COMET` model with:

```yaml
valid_metrics: ["EOLE-COMET"]
comet_batch_size: 64
```

The reference-based scorer requires references in the validation corpus.
Validation metrics configured with `valid_metrics` use `comet_compute_dtype` and
default to fp16, matching Unbabel COMET's behavior of loading fp32 weights and
then moving the model to half precision. Set `comet_compute_dtype: fp32` if you
need fp32 validation scoring.

Use the default hosted `EOLE-COMET-KIWI` model with:

```yaml
valid_metrics: ["EOLE-COMET-KIWI"]
comet_batch_size: 64
```

`EOLE-COMET-KIWI` is reference-free and scores from source+hypothesis only.

Use the default hosted `EOLE-XCOMET` model with:

```yaml
valid_metrics: ["EOLE-XCOMET"]
comet_batch_size: 64
```

`EOLE-XCOMET` is reference-optional and reports the scalar xCOMET score as a
validation metric. When references are present, it uses source+hypothesis+reference
scoring; without references, it falls back to source+hypothesis scoring. Native
prediction can compute xCOMET span metadata internally; the validation metric
interface currently emits only the scalar score.

Set `comet_model` only when you want to override the default hosted model, for
example to pin a different hosted EOLE conversion or use a local conversion:

```yaml
valid_metrics: ["EOLE-COMET-KIWI"]
comet_model: eole-nlp/wmt22-cometkiwi-da-eole-fp16
comet_batch_size: 64
```

All native EOLE COMET-family validation metrics are higher-is-better and can be
used for early stopping:

```yaml
valid_metrics: ["EOLE-COMET-KIWI"]
early_stopping: 5
early_stopping_criteria: EOLE-COMET-KIWI
```

## 3) Direct CLI scoring with hosted models

For `transformer_encoder_scorer` models, use:

- `--src`: source sentences
- `--tgt`: MT hypotheses
- `--ref`: references (required only for reference-based models; optional for XCOMET)

Reference-based COMET scoring:

```bash
eole predict \
  --model_path eole-nlp/wmt22-comet-da-eole-fp16 \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/scores.txt \
  --with_score
```

Reference-free COMET-KIWI scoring:

```bash
eole predict \
  --model_path eole-nlp/wmt23-cometkiwi-da-xl-eole-fp16 \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --output /path/to/scores.txt \
  --with_score
```

XCOMET score-only CLI output with references:

```bash
eole predict \
  --model_path eole-nlp/xcomet-xl-eole-fp16 \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/xcomet-scores.txt \
  --with_score
```

Omit `--ref` to run XCOMET in source+hypothesis mode:

```bash
eole predict \
  --model_path eole-nlp/xcomet-xl-eole-fp16 \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --output /path/to/xcomet-qe-scores.txt \
  --with_score
```

Direct `eole predict` scoring uses the regular inference dtype knob, not
`comet_compute_dtype`. Pass `--compute_dtype fp32` if you need fp32 CLI scoring.

By default, `eole predict --with_score` writes one segment score per input line:

```text
0.812345
0.734567
0.901234
```

To emit a single system-level score, use `--score_level system`:

```bash
eole predict \
  --model_path eole-nlp/wmt22-comet-da-eole-fp16 \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/system-score.txt \
  --with_score \
  --score_level system
```

`score_level: system` writes one numeric line containing the arithmetic mean of
the segment scores, similar to Unbabel COMET's `--only_system` mode but using
EOLE's score-file format:

```text
0.816049
```

## 4) Convert your own COMET models

Convert raw Unbabel COMET checkpoints when you need a model that is not already
published as a pre-converted EOLE repo, or when you need a local/offline artifact.

```bash
export EOLE_MODEL_DIR=~/Development/Models/eole

eole convert COMET \
  --model Unbabel/wmt22-comet-da \
  --output "$EOLE_MODEL_DIR/comet/Unbabel--wmt22-comet-da"

eole convert COMET \
  --model Unbabel/wmt22-cometkiwi-da \
  --output "$EOLE_MODEL_DIR/comet/Unbabel--wmt22-cometkiwi-da"
```

COMET conversion defaults to fp32 weights. To write fp16 weights for publication
or smaller local artifacts, pass `--dtype fp16`:

```bash
eole convert COMET \
  --model Unbabel/wmt22-comet-da \
  --dtype fp16 \
  --output "$EOLE_MODEL_DIR/comet/Unbabel--wmt22-comet-da-fp16"
```

For gated models, pass a token or authenticate with `hf auth login`:

```bash
eole convert COMET \
  --model Unbabel/wmt23-cometkiwi-da-xl \
  --token "$HF_TOKEN" \
  --output "$EOLE_MODEL_DIR/comet/Unbabel--wmt23-cometkiwi-da-xl"

eole convert COMET \
  --model Unbabel/XCOMET-XL \
  --token "$HF_TOKEN" \
  --output "$EOLE_MODEL_DIR/comet/Unbabel--XCOMET-XL"
```

Use a local conversion by setting `comet_model` in training config or by passing
the local directory to `eole predict --model_path`.

## 5) Parity check vs Unbabel COMET

Use the reusable parity harness to compare EOLE native scores against
[Unbabel COMET](https://github.com/Unbabel/COMET):

```bash
python recipes/scoring/comet_native/parity_check.py \
  --src /path/to/src.txt \
  --mt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --hf-model Unbabel/wmt22-comet-da \
  --eole-model "$EOLE_MODEL_DIR/comet/Unbabel--wmt22-comet-da" \
  --output /tmp/parity_wmt22_da.json
```

For KIWI / reference-free:

```bash
python recipes/scoring/comet_native/parity_check.py \
  --src /path/to/src.txt \
  --mt /path/to/mt.txt \
  --hf-model Unbabel/wmt23-cometkiwi-da-xl \
  --eole-model "$EOLE_MODEL_DIR/comet/Unbabel--wmt23-cometkiwi-da-xl" \
  --output /tmp/parity_wmt23_xl.json
```

For XCOMET, the parity harness compares scalar scores, MQM scores, and decoded
error spans against [Unbabel COMET](https://github.com/Unbabel/COMET):

```bash
python recipes/scoring/comet_native/parity_check.py \
  --src /path/to/src.txt \
  --mt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --hf-model Unbabel/XCOMET-XL \
  --eole-model "$EOLE_MODEL_DIR/comet/Unbabel--XCOMET-XL" \
  --output /tmp/parity_xcomet_xl.json
```
