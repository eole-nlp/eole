# Initialize pretrained word embeddings

Eole prepares pretrained embeddings during `eole train` initialization.
Supply the embedding text file and type in a complete training YAML:

```yaml
save_data: run/embeddings
both_embeddings: glove_dir/glove.6B.100d.txt
embeddings_type: GloVe
model:
  architecture: transformer
  embeddings:
    word_vec_size: 100
    freeze_word_vecs_enc: false
    freeze_word_vecs_dec: false
```

Use `src_embeddings` and `tgt_embeddings` for separate files. `GloVe` and
`word2vec` text formats are supported. Embedding dimensions must match the model;
set the freeze fields under `model.embeddings` to keep those weights fixed.
Matched tensors are saved as `<save_data>.enc_embeddings.pt` and
`<save_data>.dec_embeddings.pt`. Download and extract the selected vectors before
training; the placeholder path above is not a bundled asset.
