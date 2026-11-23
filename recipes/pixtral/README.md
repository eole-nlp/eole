# Pixtral

Only the mistral-community/pixtral-12b version is supported right now. The official mistralai/Pixtral-12B-2409 config is rather incomplete.

## Convert the model

Run from the repository root in an Eole environment with CUDA and FlashAttention.

```bash
export EOLE_MODEL_DIR="$PWD/models"
mkdir -p "$EOLE_MODEL_DIR/mistralai"
eole convert HF --model_dir mistral-community/pixtral-12b \
  --output "$EOLE_MODEL_DIR/mistralai/pixtral-12b"
```

## Run the test script

```bash
python recipes/pixtral/test_inference.py
```

There are several examples in the test script (taken from pixtral blog posts). A single one is activated by default, but you can uncomment the others to test the various cases.

## Finetuning

Finetuning is untested for now. Feel free to try it out and fix any arising issues.