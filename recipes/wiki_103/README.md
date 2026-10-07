# Train a language model on WikiText-103

Run from `recipes/wiki_103` in your Eole environment. Preparation requires
`huggingface_hub`, `pandas`, `pyarrow`, `pyonmttok`, GNU coreutils, and OpenSSL.
The [pinned WikiText snapshot](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1)
contains the raw train, validation, and test Parquet splits.

```bash
cd recipes/wiki_103
pip install huggingface_hub pandas pyarrow
bash prepare_wikitext-103_data.sh
eole build_vocab -c wiki_103.yaml --n_sample -1
eole train -c wiki_103.yaml
```

Preparation removes empty lines, shuffles training lines with seed 42, and
learns `data/wikitext-103-raw-v1/subwords.bpe`. The YAML reads `train.txt` and
`validation.txt` from that directory and applies BPE on the fly. A decoder-only
corpus uses `path_src` without `path_tgt`. Checkpoints are written to
`data/wikitext-103-raw-v1/run/model-lm`, with step directories below it.

The default is a six-layer Transformer trained for 100,000 steps on GPU 0.
Adjust batching and precision under `training:` to fit your GPU. The old
perplexity expectation of 20–22 has not been revalidated on the current code.

## Generate text

Use raw text: the saved tokenizer transforms are applied at inference time.

```bash
head -n 10 data/wikitext-103-raw-v1/validation.txt | cut -d ' ' -f 1-15 \
  > data/wikitext-103-raw-v1/lm_input.txt
eole predict -c inference.yaml
```

The inference YAML uses the root checkpoint (latest saved step), nucleus
sampling with `top_p: 0.9`, and ten candidate sequences with three returned.
Generated text is written to `data/wikitext-103-raw-v1/lm_pred.txt`.
