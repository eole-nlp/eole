# Gradient checkpointing

In a complete training YAML, set the layers to recompute during backward:

```yaml
training:
  use_ckpting: [ffn, mha, lora]
```

Supported names are `ffn`, `mha`, and `lora`. Checkpointed modules must participate
in gradient computation. This trades extra computation for lower activation
memory during training; it is not an inference cache option.
