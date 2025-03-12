# Compute dtype (precision) and storage dtype

Various compute precisions are supported. Below is a quick recap of the current cases.

## How to configure

It's important to note that compute precision does not necessarily reflect model parameters dtype.
With this considered, compute precision can be configured by setting the `compute_dtype` field.
From that, and other optimization settings (or specicic cases), the `storage_dtype` computed field is deduced.
This is different from the specific quantization logic configured via `quant_layers` and `quant_type`. If such quantization is enabled, precision is still taken into account for non quantized components.

**Note**: the `compute_dtype` field can take both `str` and `torch.dtype` input types. An `str` input is validated to the corresponding [`torch.dtype`](https://pytorch.org/docs/stable/tensors.html) via a custom mapping (see `eole.config.common.RunningConfig.compute_dtype`).

## Available modes

### Full precision
`compute_dtype: {fp32, torch.float32}`
Standard float precision.

**Note**: flash attention is not compatible with float32 precision.

### Half precision
`compute_dtype: {fp16, torch.float16}`

In most cases, the main model `storage_dtype` will be `torch.float32`, and some parameters will be automatically casted to `torch.float16` with torch [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html).

**Note**: this means that checkpoints will be stored in `torch.float32` in the `amp` case.

### BFloat16
`compute_dtype: {bf16, torch.bfloat16}`

See [bfloat16 floating-point format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) for specificities.

When using the flag use_amp=True, behavior will be the same as above, ie torch AMP (mixed precision)

When using use_amp=False, we switch to torch-optimi which enables pure BF16 training using Kahan summation. see [Optimi](https://github.com/warner-benjamin/optimi)


### Int8
`compute_dtype: {int8, torch.int8}`

This specific setting is only valid for **CPU prediction**, to enable [Dynamic Quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html).

In that case, `storage_dtype` will initially be `torch.float32`, and the model will then be quantized to `torch.qint8` with [`torch.quantization.quantize_dynamic`](https://pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html).
