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

An exception is when using the `fusedadam` optimizer, which is more efficient using the legacy [`apex`](https://github.com/NVIDIA/apex/blob/master/apex/contrib/optimizers/fused_adam.py) implementation. This relies on the legacy [`FP16_Optimizer`](https://github.com/NVIDIA/apex/blob/master/apex/contrib/optimizers/fp16_optimizer.py) which requires swiching the model to `torch.float16` upstream.

### BFloat16
`compute_dtype: {bf16, torch.bfloat16}`

See [bfloat16 floating-point format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) for specificities.

For now, the logic is the same as the `torch.float16` case with `torch.amp`. This is experimental and has not been extensively tested.
Some specific implementations might be explored, e.g. this [adapted AdamW implementation](https://github.com/arogozhnikov/adamw_bfloat16).


### Int8
`compute_dtype: {int8, torch.int8}`

This specific setting is only valid for **CPU prediction**, to enable [Dynamic Quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html).

In that case, `storage_dtype` will initially be `torch.float32`, and the model will then be quantized to `torch.qint8` with [`torch.quantization.quantize_dynamic`](https://pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html).
