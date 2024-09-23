# naive script with commands from the readme
# (useful to make sure the recipe still runs)

eole convert HF --model_dir openai-community/gpt2 --output $EOLE_MODEL_DIR/openai_gpt2 --token $HF_TOKEN
echo -e "The European Union was created in" > lm_input.txt
eole predict -c inference.yaml
eole tools eval_hellaswag -c inference.yaml