# NLLB

## Conversion

### 1. Sentencepiece with OpenNMT Tokenizer

```bash
eole convert HF --model_dir facebook/nllb-200-1.3B --output ./nllb-1.3b --token $HF_TOKEN --tokenizer onmt
```

### 2. HuggingFace Tokenizer

```bash
eole convert HF --model_dir facebook/nllb-200-1.3B --output ./nllb-1.3b --token $HF_TOKEN
```


## Inference

```bash
echo "What is the weather like in Tahiti?" > test.en
```


### 1. Sentencepiece with OpenNMT Tokenizer

```bash
eole predict -c inference-pyonmttok.yaml
```

### 2. HuggingFace Tokenizer

```bash
eole predict -c inference-hf.yaml
```