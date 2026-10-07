# Model conversion smoke tests

This runner converts selected Hugging Face models, generates a short continuation,
and evaluates MMLU. It does not compare outputs or logits against Transformers
and is not a numerical parity validator.

Run from the repository root:

```bash
export EOLE_MODEL_DIR=/path/to/models
# Set HF_TOKEN if any selected checkpoint requires authentication.
bash recipes/model-validator/run.sh > model-validator.log 2>&1
```

Before running, edit the `models` array in `run.sh` to select a small set that
fits your hardware and disk capacity. The default list is a historical catalogue,
not a current compatibility guarantee. Each entry must be an HF repository ID;
append `|quant` to enable runtime bitsandbytes NF4 quantization for that model.
Install `bitsandbytes` for those entries. Conversion still stores the original
weights, so quantization does not reduce conversion storage requirements.

Generation and MMLU use GPU 0. Results go into `outputs/<HF-repository-ID>/`, and
failures are recorded in `error_log.txt`. Existing conversion directories can be
written again, so use dedicated model storage. This checks execution, not output
quality; see [MMLU](../mmlu/README.md) for its evaluation method.
