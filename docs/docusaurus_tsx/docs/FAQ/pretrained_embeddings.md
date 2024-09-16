# How do I use Pretrained embeddings (e.g. GloVe)?

This is handled in the initial steps of the `onmt_train` execution.

Pretrained embeddings can be configured in the main YAML configuration file.

### Example

1. Get GloVe files:

```bash
mkdir "glove_dir"
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d "glove_dir"
```

2. Adapt the configuration:

```yaml
# <your_config>.yaml

<Your data config...>

...

# this means embeddings will be used for both encoder and decoder sides
both_embeddings: glove_dir/glove.6B.100d.txt
# to set src and tgt embeddings separately:
# src_embeddings: ...
# tgt_embeddings: ...

# supported types: GloVe, word2vec
embeddings_type: "GloVe"

# word_vec_size need to match with the pretrained embeddings dimensions
word_vec_size: 100

```

3. Train:

```bash
eole train -config <your_config>.yaml
```

Notes:

- the matched embeddings will be saved at `<save_data>.enc_embeddings.pt` and `<save_data>.dec_embeddings.pt`;
- additional flags `freeze_word_vecs_enc` and `freeze_word_vecs_dec` are available to freeze the embeddings.
