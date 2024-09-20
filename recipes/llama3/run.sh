# naive script with commands from the readme
# (useful to make sure the recipe still runs)

eole convert HF --model_dir meta-llama/Meta-Llama-3-8B-Instruct --output $EOLE_MODEL_DIR/llama3-8b-instruct --token $HF_TOKEN
echo -e "What are some nice places to visit in France?" | sed ':a;N;$!ba;s/\n/｟newline｠/g' > test_prompt.txt
eole predict -c llama-inference.yaml -src test_prompt.txt -output test_output.txt