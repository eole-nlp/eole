#!/bin/bash
# Convert a HuggingFace Whisper model to eole format.
#
# Usage:
#   ./convert_whisper.sh [MODEL_NAME] [OUTPUT_DIR]
#
# Examples:
#   ./convert_whisper.sh openai/whisper-tiny ./whisper-tiny-eole
#   ./convert_whisper.sh openai/whisper-base ./whisper-base-eole
#   ./convert_whisper.sh openai/whisper-small ./whisper-small-eole
#   ./convert_whisper.sh openai/whisper-medium ./whisper-medium-eole
#   ./convert_whisper.sh openai/whisper-large-v3 ./whisper-large-v3-eole

set -e

MODEL_NAME="${1:-openai/whisper-tiny}"
OUTPUT_DIR="${2:-./whisper-tiny-eole}"

echo "Converting ${MODEL_NAME} to eole format..."
echo "Output directory: ${OUTPUT_DIR}"

eole convert HF --model_dir "${MODEL_NAME}" --output "${OUTPUT_DIR}"

echo "Conversion complete. Model saved to ${OUTPUT_DIR}"
