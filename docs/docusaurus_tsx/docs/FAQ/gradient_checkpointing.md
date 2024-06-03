# How to use gradient checkpointing when dealing with a big model ?

* `use_ckpting: ["ffn", "mha", "lora"]`

Be carefull, the module that you use checkpointing needs to have gradients.