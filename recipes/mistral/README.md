# Mistral inference

Run from the repository root in your CUDA-enabled Eole environment. This example
uses the unquantized base model `mistralai/Mistral-7B-v0.3`, not an instruction-tuned
or AWQ checkpoint. BF16 weights alone need approximately 14 GB; allow additional
VRAM for caches and working memory.

```bash
export EOLE_MODEL_DIR=/path/to/models
# Set HF_TOKEN if authentication is required.
eole convert HF --model_dir mistralai/Mistral-7B-v0.3 \
  --output "$EOLE_MODEL_DIR/mistral-7b-v0.3" --token "$HF_TOKEN"
printf '%s\n' 'What are some nice places to visit in France?' > test_prompt.txt
eole predict -c recipes/mistral/predict.yaml
```

The continuation is written to `test_output.txt`. The YAML uses the converted
checkpoint's tokenizer, BF16 compute, PyTorch attention, and greedy decoding.

`mistral-7b-awq-gemm-inference.yaml` is a separate legacy example requiring an
already converted `mistral-7b-instruct-v0.2-awq` checkpoint and compatible AWQ
kernels. The conversion above does not produce that checkpoint. For a documented
LoRA fine-tuning workflow, see [Llama2](../llama2/README.md).
