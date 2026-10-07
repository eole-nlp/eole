# Update a checkpoint vocabulary

Build vocabulary files for the new corpus, then point a complete training YAML
at those files and the existing checkpoint. Existing token embeddings are mapped
to the new vocabulary; newly added tokens receive initialized embeddings.

```yaml
src_vocab: /path/to/new.src.vocab
training:
  train_from: /path/to/checkpoint
  update_vocab: true
  reset_optim: states
```

For a separate target vocabulary, set `tgt_vocab` as well. The config validator
requires `reset_optim: states` or `all` when updating a vocabulary. Keep tokenizer
artifacts and vocabulary IDs consistent, especially for converted HF models.
