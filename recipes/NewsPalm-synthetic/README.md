# Example of using Hugging Face streaming dataset

## Based on: https://arxiv.org/pdf/2408.06537

Introducing the NewsPaLM MBR and QE Dataset:
LLM-Generated High-Quality Parallel Data Outperforms Traditional
Web-Crawled Data

### Get the vocab and BPE model on HF

https://huggingface.co/eole-nlp/NewsPalmSynthetic-ENDE

copy files: 
* ende.vocab2
* subwords.en_de.bpe


### Optionally you can get the trained model to test it.


* config.json
* vocab.json
* model.00.safetensors

## Train with the yaml config file

```
eole train -c newspalm-synthetic-hfstreaming.yaml
```

## Run inference

```
eole predict -c inference.yaml --src newstest2023-src.en --output newstest2023-hyp.de
```

Then you can score with sacrebleu and/or comet


Scoring with Unbabel/wmt22-comet-da gives: 81.90

You can compare to table 5 lines 2a) to 2d) of the paper https://arxiv.org/pdf/2408.06537
