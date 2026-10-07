# Position encodings

For training from scratch, configure position encoding under `model.embeddings`.
Eole propagates relative attention settings to its Transformer configuration.
For converted checkpoints, preserve the converter's architecture-specific values.

```yaml
model:
  architecture: transformer_lm
  embeddings:
    position_encoding: true
    position_encoding_type: SinusoidalInterleaved
training:
  param_init_method: xavier_uniform
```

`SinusoidalConcat` is another absolute sinusoidal layout. Learned positions use
`position_encoding_type: Learned` and require `n_positions`. Shaw-style relative
positions use `Relative` with `n_positions` specifying the maximum relative
distance; rotary positions use `Rotary`, and ALiBi uses `Alibi`. These are different
model architectures, not inference switches to change on an existing checkpoint.
See the [model configuration reference](../reference/Config/models.md) for rotary
scaling, position limits, and validation rules.
