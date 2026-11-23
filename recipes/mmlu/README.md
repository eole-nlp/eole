# Evaluation with MMLU dataset

## How to run

Install Eole and convert the model named in your inference YAML first. This
harness requires `pandas` and the MMLU CSV layout `data/dev/*_dev.csv` and
`data/test/*_test.csv`; the repository includes those splits. Use `--data_dir`
to select a different data root. It generates a one-token answer for each
five-shot prompt; it does not rank answer-option log probabilities.


From the repository root, run the supplied Qwen3.5-0.8B example. Reuse the
converted checkpoint if you already followed the [server quickstart](../server/README.md).

```bash
export EOLE_MODEL_DIR="$PWD/models"
mkdir -p "$EOLE_MODEL_DIR"
eole convert HF --model_dir Qwen/Qwen3.5-0.8B \
  --output "$EOLE_MODEL_DIR/qwen3.5-0.8B"
EOLE_TORCH_COMPILE=0 eole tools run_mmlu -c recipes/mmlu/predict-qwen35.yaml \
  --data_dir recipes/mmlu/data
```

The full run writes `mmlu_results.json` alongside the converted checkpoint.
Use your own prediction YAML with `-c` to evaluate another model. Treat this
one-token completion result as a reproducibility check for this harness.

## Historical results

The older OpenNMT-py runs are available in the
[historical results table](https://github.com/eole-nlp/eole/blob/282528e25acbb2dacdd0ee1a8e319d88f06a0798/recipes/mmlu/README.md#results).
They document earlier checkpoints and evaluation settings.
