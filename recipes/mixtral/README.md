# Mixtral

---
**NOTE**
To make your life easier, run these commands from the recipe directory (here `recipes/mixtral`).
---

## Retrieve and convert model

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

### Download and convert model

```
eole convert HF --model_dir TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ --output ${EOLE_MODEL_DIR}/mixtral-8x7b-instruct-v0.1-awq --token $HF_TOKEN
```


## Inference

### Write test prompt to text file

```
echo -e "What are some nice places to visit in France?" | sed ':a;N;$!ba;s/\n/｟newline｠/g' > test_prompt.txt
```

### Run inference

```
eole predict -c mixtral-inference-awq.yaml -src test_prompt.txt -output test_output.txt
```