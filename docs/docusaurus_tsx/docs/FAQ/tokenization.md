# How can I apply on-the-fly tokenization and subword regularization when training?

This is naturally embedded in the data configuration format introduced in OpenNMT-py 2.0. Each entry of the `data` configuration will have its own `transforms`. `transforms` basically is a `list` of functions that will be applied sequentially to the examples when read from file.

### Example

This example applies sentencepiece tokenization with `pyonmttok`, with `nbest=20` and `alpha=0.1`.

```yaml
# <your_config>.yaml

...

# Tokenization options
src_subword_type: sentencepiece
src_subword_model: examples/subword.spm.model
tgt_subword_type: sentencepiece
tgt_subword_model: examples/subword.spm.model

# Number of candidates for SentencePiece sampling
subword_nbest: 20
# Smoothing parameter for SentencePiece sampling
subword_alpha: 0.1
# Specific arguments for pyonmttok
src_onmttok_kwargs: "{'mode': 'none', 'spacer_annotate': True}"
tgt_onmttok_kwargs: "{'mode': 'none', 'spacer_annotate': True}"

# Corpus opts:
data:
    corpus_1:
        path_src: toy-ende/src-train1.txt
        path_tgt: toy-ende/tgt-train1.txt
        transforms: [onmt_tokenize]
        weight: 1
    valid:
        path_src: toy-ende/src-val.txt
        path_tgt: toy-ende/tgt-val.txt
        transforms: [onmt_tokenize]
...

```

Other tokenization methods and transforms are readily available. See the dedicated docs for more details.