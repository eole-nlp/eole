# NLLB

## Conversion

```bash
eole convert HF --model_dir facebook/nllb-200-1.3B --output ./nllb-1.3b --token $HF_TOKEN --tokenizer onmt
```

## Inference

```bash
eole predict -c inference.yaml
```