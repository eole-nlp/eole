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
    "microsoft/phi-2"
    # Needs quantization to be tested on 24GB GPU
    # "Qwen/Qwen3-30B-A3B|quant"
    # seems ok
    # "Qwen/Qwen3-0.6B"
    # "Qwen/Qwen3-1.7B"
    # "Qwen/Qwen3-4B"
    # "Qwen/Qwen3-8B"
    # "Qwen/Qwen3-14B"
    # "Qwen/Qwen2-0.5B"
    # "Qwen/Qwen2.5-0.5B"
    # "Qwen/Qwen2.5-0.5B-Instruct"
    # "Qwen/Qwen2-1.5B"
    # "Qwen/Qwen2.5-1.5B"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B"
    # "Qwen/Qwen2.5-3B-Instruct"
    # to work on
    # "mistralai/Mathstral-7B-v0.1" # fp32 !
    # "microsoft/Phi-3.5-MoE-instruct" # convert_HF not set for PhiMoEForCausalLM
    # "microsoft/Phi-3-small-128k-instruct" # tokenizer to be taken from another model
)

QUANT_SETTINGS="--quant_type bnb_NF4 --quant_layers gate_up_proj down_proj up_proj linear_values linear_query linear_keys final_linear w_in w_out"

# Log file for errors
ERROR_LOG="$SCRIPT_DIR/error_log.txt"
echo "Error log for $(date)" > "$ERROR_LOG"

# Loop through models
for model_entry in "${models[@]}"; do
  IFS='|' read -r model_path model_flag <<< "$model_entry"
  model_name=$(basename "$model_path")

  # Determine quantization
  quant_args=""
  if [[ "$model_flag" == "quant" ]]; then
    echo "Quantization enabled for $model_name"
    quant_args=$QUANT_SETTINGS
  else
    echo "Quantization disabled for $model_name"
  fi

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
  if ! eole convert HF --model_dir "$model_path" --output "$MODEL_DIR" --token "$HF_TOKEN" --nshards 2; then
    echo "Error: Conversion failed for $model_name" | tee -a "$ERROR_LOG"
    continue
  fi
  
  # Step 2: Prepare the prompt
  echo "Preparing prompt for testing:"
  PROMPT="What are some nice places to visit in France?"
  # special tokens prompt (to check Qwen instruct models for instance)
  # PROMPT="<|im_start|>user\nWhat are some nice places to visit in France?<|im_end|>\n<|im_start|>assistant\n"
  echo "\"$PROMPT\""
  if ! echo -e "$PROMPT" | sed ':a;N;$!ba;s/\n/｟newline｠/g' > "$test_prompt_file"; then
    echo "Error: Failed to prepare prompt for $model_name" | tee -a "$ERROR_LOG"
    continue
  fi
  echo "Prompt saved to $test_prompt_file"
  
  # Step 3: Run prediction
  echo "Running prediction:"
  if ! eole predict -model_path "$MODEL_DIR" -gpu_ranks 0 -src "$test_prompt_file" -output "$test_output_file" $QUANT_SETTINGS; then
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
  if ! eole tools run_mmlu -model_path "$MODEL_DIR" -gpu_ranks 0 -batch_size 1 -batch_type sents $QUANT_SETTINGS; then
    echo "Error: Failed to run MMLU for $model_name" | tee -a "$ERROR_LOG"
    continue
  fi
  echo "-------------------------------------------------"
  
  
  echo "Files saved to $DATA_DIR"
  echo "================================================="
done

echo "Script finished. Check $ERROR_LOG for any errors."
