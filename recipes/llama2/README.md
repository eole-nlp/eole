# Llama2

---
**NOTE**
To make your life easier, run these commands from the recipe directory (here `recipes/llama2`).
---

## Retrieve and convert model

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

### Download and convert model

```
eole convert HF --model_dir meta-llama/Llama-2-7b-chat-hf --output $EOLE_MODEL_DIR/llama2-7b-chat-hf --token $HF_TOKEN
```


## Inference

### Write test prompt to text file

```
echo -e "<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
What are some nice places to visit in France? [/INST]" | sed ':a;N;$!ba;s/\n/｟newline｠/g' > test_prompt.txt
```

### Run inference

**Single GPU**
```
eole predict -c llama-inference.yaml -src test_prompt.txt -output test_output.txt
```

**Dual GPU (tensor parallelism)**
```
eole predict -c llama-inference-tp-2gpu.yaml -src test_prompt.txt -output test_output.txt
```

## Finetuning

### Retrieve data

```
[ ! -d ./data ] && mkdir ./data
# Alpaca
wget -P ./data https://opennmt-models.s3.amazonaws.com/llama/alpaca_clean.txt

# Vicuna
wget -P ./data https://opennmt-models.s3.amazonaws.com/llama/sharegpt.txt

# Open Assisstant
wget -P ./data https://opennmt-models.s3.amazonaws.com/llama/osst1.flattened.txt
```


### Finetune

```
eole train -c llama-finetune.yaml
```

### Merge LoRa weights

```
eole model lora --action merge --base_model ${EOLE_MODEL_DIR}/llama2-7b-chat-hf --lora_weights ./finetune/llama2-7b-chat-hf-finetune --output ./finetune/merged
```

Then you can just update your inference setup to use the newly finetuned & merged model.