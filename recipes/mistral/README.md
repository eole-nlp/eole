# Mistral

---
**NOTE**
To make your life easier, run these commands from the recipe directory (here `recipes/mistral`).
---

## Retrieve and convert model

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

### Download and convert model

```
eole convert HF --model_dir mistralai/Mistral-7B-v0.3 --output ${EOLE_MODEL_DIR}/mistral-7b-v0.3 --token $HF_TOKEN
```


## Inference

### Write test prompt to text file

```
echo -e "What are some nice places to visit in France?" | sed ':a;N;$!ba;s/\n/｟newline｠/g' > test_prompt.txt
```

### Run inference

```
eole predict -c mistral-7b-awq-gemm-inference.yaml -src test_prompt.txt -output test_output.txt
```