# How to use LoRA and quantization to finetune a big model?

Cf paper: [LoRA](https://arxiv.org/abs/2106.09685)

LoRA is a mechanism that helps to finetune bigger models on a single GPU card by limiting the amount of VRAM needed.
The principle is to make only a few layers trainable (hence reducing the amount of required memory especially for the Adam optimizer).

You need to train_from a model (for instance NLLB-200 3.3B) and use the following options:

* `lora_layers: ['linear_values', 'linear_query']` these are the two layers of the Self-Attention module the paper recommends to make trainable.
* `lora_rank: 2`
* `lora_dropout: 0.1` or any value you can test
* `lora_alpha: 1` or any value you can test
* `lora_embedding: true` makes Embeddings LoRA compatible, hence trainable in the case you use `update_vocab: true` or if you want to finetune Embeddings as well.

Bitsandbytes enables quantization of Linear layers. For more information: https://github.com/TimDettmers/bitsandbytes
Also you can read the blog post here: https://huggingface.co/blog/hf-bitsandbytes-integration

You need to add the following options:

* `quant_layers: ['up_proj', 'down_proj', 'linear_values', 'linear_query']`
* `quant_type: "bnb_NF4"`

You can for instance quantize the layers of the PositionWise Feed-Forward from the Encoder/Decoder and the key/query/values/final from the Multi-head attention.
Choices for quantization are `"bnb_8bit"`, `"bnb_FP4"`, `"bnb_NF4"`, `"awq_gemm"`, `"awq_gemv"`, `"autoround"`, `"gguf"`.


