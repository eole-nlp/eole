# Llama3

---
**NOTE**
To make your life easier, run these commands from the recipe directory (here `recipes/llama3`).
---

## Retrieve and convert model

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

### Download and convert model

```
eole convert HF --model_dir meta-llama/Meta-Llama-3-8B-Instruct --output $EOLE_MODEL_DIR/llama3-8b-instruct --token $HF_TOKEN
```


## Inference

### Write test prompt to text file

```
echo -e "What are some nice places to visit in France?" | sed ':a;N;$!ba;s/\n/｟newline｠/g' > test_prompt.txt
```

### Run inference

```
eole predict -c llama-inference.yaml -src test_prompt.txt -output test_output.txt
```