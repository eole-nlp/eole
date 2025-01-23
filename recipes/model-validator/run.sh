#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Define the models table
models=(
    # Validated
    "google/gemma-2-2b"
    "mistralai/Ministral-8B-Instruct-2410"
    "mistralai/Mistral-7B-v0.3"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/CodeLlama-7b-hf"
    "microsoft/Phi-3.5-mini-instruct"
    "microsoft/Phi-3-mini-128k-instruct"
    # to work on
    # "mistralai/Mathstral-7B-v0.1" # fp32 !
    # "microsoft/Phi-3.5-MoE-instruct" # convert_HF not set for PhiMoEForCausalLM
    # "microsoft/Phi-3-small-128k-instruct" # tokenizer to be taken from another model
)

# Log file for errors
ERROR_LOG="$SCRIPT_DIR/error_log.txt"
echo "Error log for $(date)" > "$ERROR_LOG"

# Loop through models
for model_path in "${models[@]}"; do
  model_name=$(basename "$model_path")

  echo "================================================="
  echo "Processing model: $model_name"
  echo "Path: $model_path"
  echo "================================================="
  
  # Define directories and paths
  DATA_DIR="$SCRIPT_DIR/outputs/$model_path"
  MODEL_DIR="$EOLE_MODEL_DIR/$model_path"
  mkdir -p "$DATA_DIR"
  
  test_prompt_file="$DATA_DIR/test_prompt.txt"
  test_output_file="$DATA_DIR/test_output.txt"

  # Step 1: Convert the model
  echo "Converting to $MODEL_DIR"
  if ! eole convert HF --model_dir "$model_path" --output "$MODEL_DIR" --token "$HF_TOKEN"; then
    echo "Error: Conversion failed for $model_name" | tee -a "$ERROR_LOG"
    continue
  fi
  
  # Step 2: Prepare the prompt
  echo "Preparing prompt for testing:"
  PROMPT="What are some nice places to visit in France?"
  echo "\"$PROMPT\""
  if ! echo -e "$PROMPT" | sed ':a;N;$!ba;s/\n/｟newline｠/g' > "$test_prompt_file"; then
    echo "Error: Failed to prepare prompt for $model_name" | tee -a "$ERROR_LOG"
    continue
  fi
  echo "Prompt saved to $test_prompt_file"
  
  # Step 3: Run prediction
  echo "Running prediction:"
  if ! eole predict -model_path "$MODEL_DIR" -gpu_ranks 0 -src "$test_prompt_file" -output "$test_output_file"; then
    echo "Error: Prediction failed for $model_name" | tee -a "$ERROR_LOG"
    continue
  fi
  
  # Step 4: Output the prediction
  echo "Prediction for $model_name:"
  echo "-------------------------------------------------"
  if ! cat "$test_output_file"; then
    echo "Error: Failed to read output for $model_name" | tee -a "$ERROR_LOG"
    continue
  fi
  echo "-------------------------------------------------"

  # Step 5: Run MMLU
  echo "MMLU for $model_name:"
  echo "-------------------------------------------------"
  if ! eole tools run_mmlu -model_path "$MODEL_DIR" -gpu_ranks 0 -batch_size 1 -batch_type sents; then
    echo "Error: Failed to run MMLU for $model_name" | tee -a "$ERROR_LOG"
    continue
  fi
  echo "-------------------------------------------------"
  
  
  echo "Files saved to $DATA_DIR"
  echo "================================================="
done

echo "Script finished. Check $ERROR_LOG for any errors."
