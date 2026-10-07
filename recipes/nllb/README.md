# NLLB conversion and translation

Run from this recipe directory: `cd recipes/nllb`. This is an English-to-German
inference example with `facebook/nllb-200-1.3B`; for translation training from
scratch see [WMT17](../wmt17/README.md).

Choose one tokenizer route. Use separate output directories so converting one
route does not overwrite the other. Set `HF_TOKEN` if authentication is required.

## Hugging Face tokenizer

```bash
eole convert HF --model_dir facebook/nllb-200-1.3B \
  --output ./nllb-1.3b-hf --token "$HF_TOKEN"
printf '%s\n' 'What is the weather like in Tahiti?' > test.en
eole predict -c inference-hf.yaml
```

## SentencePiece through OpenNMT tokenizer

```bash
eole convert HF --model_dir facebook/nllb-200-1.3B \
  --output ./nllb-1.3b-onmt --token "$HF_TOKEN" --tokenizer onmt
printf '%s\n' 'What is the weather like in Tahiti?' > test.en
eole predict -c inference-pyonmttok.yaml
```

Both configurations use GPU 0 and write German translations to `test.de`.
They add the source `eng_Latn` and target `deu_Latn` language prefixes. Change
these prefixes together when changing the translation direction. Tokenizer
settings are loaded from each converted checkpoint.
