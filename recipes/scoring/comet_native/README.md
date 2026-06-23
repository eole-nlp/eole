# Native EOLE COMET Scoring

This recipe shows how to use native EOLE scoring with converted
[Unbabel COMET](https://github.com/Unbabel/COMET) models:

- `EOLE-COMET` (reference-based)
- `EOLE-COMET-KIWI` (reference-free)
- `EOLE-XCOMET` (reference-based, xCOMET scalar score with explainable spans validated internally)

Converted COMET models are represented as EOLE `transformer_encoder_scorer`
models with the COMET-specific `scoring_type: comet` specialization. Direct
inference from Unbabel Hugging Face model IDs is not supported here; convert the
COMET checkpoint first.

## 1) Convert COMET models to EOLE artifacts

```bash
export EOLE_MODEL_DIR=~/Development/Models/eole

eole convert COMET --model Unbabel/wmt22-comet-da --output "$EOLE_MODEL_DIR/comet/Unbabel--wmt22-comet-da"
eole convert COMET --model Unbabel/wmt22-cometkiwi-da --output "$EOLE_MODEL_DIR/comet/Unbabel--wmt22-cometkiwi-da"
```

For gated models, pass a token:

```bash
eole convert COMET --model Unbabel/wmt23-cometkiwi-da-xl --token "$HF_TOKEN" --output "$EOLE_MODEL_DIR/comet/Unbabel--wmt23-cometkiwi-da-xl"
eole convert COMET --model Unbabel/XCOMET-XL --token "$HF_TOKEN" --output "$EOLE_MODEL_DIR/comet/Unbabel--XCOMET-XL"
```

## 2) Use `EOLE-COMET` in a training config

Set in your train YAML:

```yaml
valid_metrics: ["EOLE-COMET"]
comet_model: "${EOLE_MODEL_DIR}/comet/Unbabel--wmt22-comet-da"
comet_batch_size: 64
```

The reference-based scorer requires references in the validation corpus.

## 3) Use `EOLE-COMET-KIWI` in a training config

Set in your train YAML:

```yaml
valid_metrics: ["EOLE-COMET-KIWI"]
comet_model: "${EOLE_MODEL_DIR}/comet/Unbabel--wmt22-cometkiwi-da"
comet_batch_size: 64
```

`EOLE-COMET-KIWI` is reference-free and scores from source+hypothesis only.

## 4) Use `EOLE-XCOMET` in a training config

Set in your train YAML:

```yaml
valid_metrics: ["EOLE-XCOMET"]
comet_model: "${EOLE_MODEL_DIR}/comet/Unbabel--XCOMET-XL"
comet_batch_size: 64
```

`EOLE-XCOMET` is reference-based and reports the scalar xCOMET score as a
validation metric. Native prediction can compute xCOMET span metadata internally;
the validation metric interface currently emits only the scalar score.

## 5) Parity check vs Unbabel COMET

Use the reusable parity harness to compare EOLE native scores against
[Unbabel COMET](https://github.com/Unbabel/COMET):

```bash
python recipes/scoring/comet_native/parity_check.py \
  --src /path/to/src.txt \
  --mt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --hf-model Unbabel/wmt22-comet-da \
  --eole-model "${EOLE_MODEL_DIR}/comet/Unbabel--wmt22-comet-da" \
  --output /tmp/parity_wmt22_da.json
```

For KIWI / reference-free:

```bash
python recipes/scoring/comet_native/parity_check.py \
  --src /path/to/src.txt \
  --mt /path/to/mt.txt \
  --hf-model Unbabel/wmt23-cometkiwi-da-xl \
  --eole-model "${EOLE_MODEL_DIR}/comet/Unbabel--wmt23-cometkiwi-da-xl" \
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
  --eole-model "${EOLE_MODEL_DIR}/comet/Unbabel--XCOMET-XL" \
  --output /tmp/parity_xcomet_xl.json
```

## 6) Direct CLI scoring with `eole predict`

For `transformer_encoder_scorer` models, use:

- `--src`: source sentences
- `--tgt`: MT hypotheses
- `--ref`: references (required only for reference-based models)

Reference-based (`EOLE-COMET`):

```bash
eole predict \
  --model_path "$EOLE_MODEL_DIR/comet/Unbabel--wmt22-comet-da" \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/scores.txt \
  --with_score
```

Reference-free (`EOLE-COMET-KIWI`):

```bash
eole predict \
  --model_path "$EOLE_MODEL_DIR/comet/Unbabel--wmt22-cometkiwi-da" \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --output /path/to/scores.txt \
  --with_score
```

XCOMET (`EOLE-XCOMET`) score-only CLI output:

```bash
eole predict \
  --model_path "$EOLE_MODEL_DIR/comet/Unbabel--XCOMET-XL" \
  --src /path/to/src.txt \
  --tgt /path/to/mt.txt \
  --ref /path/to/ref.txt \
  --output /path/to/xcomet-scores.txt \
  --with_score
```

By default, `eole predict --with_score` writes one segment score per input line:

```text
0.812345
0.734567
0.901234
```

To emit a single system-level score, use `--score_level system`:

```bash
eole predict \
  --model_path "$EOLE_MODEL_DIR/comet/Unbabel--wmt22-comet-da" \
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
