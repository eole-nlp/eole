# Train a language model on FineWeb 10BT

Run from `recipes/fineweb10B` in your Eole environment. This is a GPT-2-sized
training example inspired by [llm.c](https://github.com/karpathy/llm.c/discussions/481),
not a reproduction of its results. The [FineWeb 10BT sample](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/sample/10BT)
is a large download; allow space for both Parquet and generated text.

```bash
cd recipes/fineweb10B
pip install huggingface_hub pandas pyarrow tiktoken tqdm
hf download HuggingFaceFW/fineweb --repo-type dataset --include 'sample/10BT/*.parquet' --local-dir data
python parse_fineweb_10B.py data/sample/10BT --valid_size 100000000
head -n 50000 data/sample/10BT/fineweb10B_valid.txt \
  > data/sample/10BT/fineweb10B_valid.50k.txt
eole train -c fineweb10B.yaml
```

Preparation assigns the first approximately 100 million GPT-2 tokens to
validation, splitting at document boundaries, then writes remaining documents
to training. The YAML uses only the first 50,000 validation lines. Files are
processed in sorted order; this is not a randomized split.

The bundled `vocab.txt` and `merges.txt` provide GPT-2 BPE artifacts, so no
`build_vocab` step is needed. The model is trained from scratch; its sinusoidal
position encoding differs from GPT-2's learned positional embeddings.

The YAML configures GPU 0, FP16 mixed precision, 20,000 training steps, and
checkpoints in `model_fineweb10B_gpt2`. Its large token batches, accumulation,
and prefetch settings are research defaults; reduce them under `training:` to
fit GPU and host memory. Full training and quality have not been rerun during
the documentation audit.
