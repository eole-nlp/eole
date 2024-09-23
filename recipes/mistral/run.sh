eole convert HF --model_dir mistralai/Mistral-7B-v0.3 --output ${EOLE_MODEL_DIR}/mistral-7b-v0.3 --token $HF_TOKEN
echo -e "What are some nice places to visit in France?" | sed ':a;N;$!ba;s/\n/｟newline｠/g' > test_prompt.txt
eole predict -c mistral-7b-awq-gemm-inference.yaml -src test_prompt.txt -output test_output.txt