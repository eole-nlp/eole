#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${EOLE_MODEL_DIR:?Set EOLE_MODEL_DIR to your model storage directory}"
eole convert HF --model_dir mistralai/Mistral-7B-v0.3 --output "$EOLE_MODEL_DIR/mistral-7b-v0.3" --token "${HF_TOKEN:-}"
printf '%s\n' 'What are some nice places to visit in France?' > test_prompt.txt
eole predict -c recipes/mistral/predict.yaml
