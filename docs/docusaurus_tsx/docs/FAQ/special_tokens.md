# What special tokens are used?

There are 4 main special tokens:
- BOS for "beginning of sentence";
- PAD for "padding";
- EOS for "end of sentence";
- UNK for "unknown".

## Special tokens actually used

Depending on the context, these tokens can take various values:

1. Default behaviour, training from scratch

Some default values are defined as [constants](https://github.com/eole-nlp/eole/blob/ff39275c50d12951963008da11d029940b590713/eole/constants.py#L8) for the project:
```python
class DefaultTokens(object):
    PAD = "<blank>"
    BOS = "<s>"
    EOS = "</s>"
    UNK = "<unk>"
```

2. Retrieving a pretrained model from HF

The special tokens will be retrieved and configured from the `special_tokens_map.json` configuration file from the HF model files.

3. Custom behaviour

In any case, these tokens can be overriden via the ad-hoc configuration settings:
- `bos_token`
- `pad_token`
- `eos_token`
- `unk_token`

## Special tokens behaviour in Eole

When we train a SEQ2SEQ model we use:
SRC: srctok1 srctok2 srctok3 .... srctokn
TGT: BOS tgttok1 tgttok2 ..... tgttokm EOS
But when training a LM
SRC: BOS srctok1 srctok2 srctok3 .... srctokn
TGT: srctok1 srctok2 srctok3 .... srctokn EOS

Having said that, sometimes we need to finetune models (eg: NLLB-200, Llama, ...) with existing vocab
and special tokens are not the same.

ex with NLLB-200
BOS id=0
PAD id=1
EOS id=2
UNK id=3
And the decoder start token is EOS (</s>) which means in fact that the BOS is never used.
At training, TGT needs to start with EOS instead of BOS in the default OpenNMT-py config.

Example of Llama
UNK id=0
BOS id=1
EOS id=2
There was no PAD but to avoid conflicts we forced PAD id=3 (which was token '<0x00>' in the original llama tokenizer)
