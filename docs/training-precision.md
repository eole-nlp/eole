# Training precision

Set these options inside the `training:` section of a training configuration.
Choose precision based on GPU support, memory use, and validation quality.

## Automatic mixed precision

```yaml
training:
  compute_dtype: bf16  # use fp16 on appropriate hardware
  use_amp: true
  optim: adamw        # adam is also supported
```

Eole uses PyTorch optimizers for AMP. Its implementation also uses GradScaler
with BF16 AMP; keep that behavior when reproducing existing training runs.
Compare validation metrics when changing precision rather than assuming two
configurations will converge identically.

## Pure BF16 with torch-optimi

```yaml
training:
  compute_dtype: bf16
  use_amp: false
  optim: adamw
```

When available, `torch-optimi` is selected for supported optimizers without AMP.
Its low-precision updates use Kahan summation to reduce accumulated rounding
error. This can reduce optimizer memory compared with keeping FP32 master
weights. Training speed and final quality depend on the workload; measure both
before changing a production training configuration.

The legacy Apex-style `fusedadam` implementation was deprecated in Eole 0.2.
See [configuration documentation](https://eole-nlp.github.io/eole) and
[training recipes](../recipes/README.md) for complete examples.
