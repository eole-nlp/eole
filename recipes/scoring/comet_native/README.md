# Native EOLE COMET Scoring

This recipe shows how to use native EOLE scoring with converted
[Unbabel COMET](https://github.com/Unbabel/COMET) models:

- `EOLE-COMET` (reference-based)
- `EOLE-COMET-KIWI` (reference-free)

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

## 4) Parity check vs Unbabel COMET

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

## 5) Direct CLI scoring with `eole predict`

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
