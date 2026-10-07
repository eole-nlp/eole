# Ensemble models at inference

Provide several converted checkpoint directories to `eole predict`:

```bash
eole predict --model_path ./model1 ./model2 --src input.txt --output output.txt --gpu_ranks 0
```

Models must use compatible architectures, tokenization, and target vocabularies.
Ensembling consumes memory for every model. This is a prediction feature;
MTP speculative decoding does not support ensembling and falls back to ordinary
decoding when its configuration is unsupported.
