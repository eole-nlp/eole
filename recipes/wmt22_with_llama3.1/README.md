# Llama3

---
**NOTE**
To make your life easier, run these commands from the recipe directory (here `recipes/wmt22_with_llama3.1`).
---

## Retrieve and convert model

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

### Download and convert model

```
eole convert HF --model_dir meta-llama/Meta-Llama-3.1-8B --output $EOLE_MODEL_DIR/llama3.1-8b --token $HF_TOKEN
```


## Inference

### Build the prompt for translation of newstest2022-src.en

```
python promptize_llama3.py
```

### Run inference

```
eole predict -c llama-instruct-inference.yaml -src newstest2022-src-prompt.en -output newstest2022-hyp.de
```

Then you can score newstest2022-hyp.de against newstest2022-ref.de with a scorer (sacrebleu or comet) or just use cometkiwi for reference-less score.
