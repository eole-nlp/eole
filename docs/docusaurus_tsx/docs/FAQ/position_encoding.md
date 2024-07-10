# Position encoding: Absolute vs Relative vs Rotary Embeddings vs Alibi

The basic feature is absolute position encoding stemming from the original Transformer Paper.
However, even with this, we can use SinusoidalInterleaved (default OpenNMT-py) or SinusoidalConcat (default Fairseq imported models)
* `position_encoding_type: 'SinusoidalInterleaved'`
Do not forget to set also `param_init_glorot: true`

If you prefer to use relative position encoding, we support 3 modes:
* "Shaw": https://arxiv.org/abs/1803.02155 - you need to set `position_encoding_type: 'Relative'` and `n_positions: N` where N > 1 (use 16, 20, 32) see paper.
* "Rope" Rotary Embeddings: https://arxiv.org/abs/2104.09864 - you need to set `position_encoding_type: 'Rotary'`
* "Alibi" (used by MPT-7B for example) https://arxiv.org/abs/2108.12409 - you need to set `position_encoding_type: 'Alibi'`

In a nutshell, at the time if this writing (v3.1) absolute position encoding is managed in the Embeddings module, whereas
the relative position encoding is managed directly in the multi-head self-attention module.
