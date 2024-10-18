# naive script with commands from the readme
# (useful to make sure the recipe still runs)

eole convert HF --model_dir meta-llama/Llama-2-7b-chat-hf --output $EOLE_MODEL_DIR/llama2-7b-chat-hf --token $HF_TOKEN
echo -e "<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
What are some nice places to visit in France? [/INST]" | sed ':a;N;$!ba;s/\n/｟newline｠/g' > test_prompt.txt
eole predict -c llama-inference.yaml -src test_prompt.txt -output test_output.txt
eole predict -c llama-inference-tp-2gpu.yaml -src test_prompt.txt -output test_output.txt
[ ! -d ./data ] && mkdir ./data
# Alpaca
wget -P ./data https://opennmt-models.s3.amazonaws.com/llama/alpaca_clean.txt
# Vicuna
wget -P ./data https://opennmt-models.s3.amazonaws.com/llama/sharegpt.txt
# Open Assisstant
wget -P ./data https://opennmt-models.s3.amazonaws.com/llama/osst1.flattened.txt
eole train -c llama-finetune.yaml
eole model lora --action merge --base_model ${EOLE_MODEL_DIR}/llama2-7b-chat-hf --lora_weights ./finetune/llama2-7b-chat-hf-finetune --output ./finetune/merged