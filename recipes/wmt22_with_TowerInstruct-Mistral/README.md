# TowerInstruct (Mistral)

---
**NOTE**
To make your life easier, run these commands from the recipe directory (here `recipes/wmt22_with_TowerInstruct-Mistral`).
---

## Retrieve and convert model

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

### Download and convert model

```
eole convert HF --model_dir Unbabel/TowerInstruct-Mistral-7B-v0.2 --output $EOLE_MODEL_DIR/TowerInstruct-Mistral-7b-v0.2 --token $HF_TOKEN
```


## Inference

### Build the prompt for translation of newstest2022-src.en

```
python promptize_mistral.py
```

### Run inference

```
eole predict -c tower-inference.yaml -src newstest2022-src-prompt.en -output newstest2022-hyp.de
```

Then you can score newstest2022-hyp.de against newstest2022-ref.de with a scorer (sacrebleu or comet) or just use cometkiwi for reference-less score.
