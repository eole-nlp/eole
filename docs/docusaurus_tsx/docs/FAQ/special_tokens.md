# What special tokens are used?

In the v2, special tokens were different for SEQ2SEQ and LM:
LM was BOS, PAD, EOS with IDs (0, 1, 2) and the first vocab token started at id=3
SEQ2SEQ was UNK, PAD, BOS, EOS with IDs (0, 1, 2, 3) and first vocab token started at id=4

In v3 we changed this behavior to align things:
    group.add(
        "--default_specials",
        "-default_specilas",
        nargs="+",
        type=str,
        default=[
            DefaultTokens.UNK,
            DefaultTokens.PAD,
            DefaultTokens.BOS,
            DefaultTokens.EOS,
        ])

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
