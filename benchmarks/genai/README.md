These scripts show the throughput we can get from various toolkits for the Gemma3 Text 1B LLM.

## Hugging Face / Transformers:
- Results are obtained with Flash Attention.
- Batch size=1 => 31 tokens / sec
- Batch size=4 => 80 tokens / sec

This measure is not 100% accurate because it takes the (very short) time of prefill and do not count for prefill tokens.
Difference would be negligible

## CTranslate2:
- The model needs to be converted first.
- CT2 is a full C++ inference only toolkit.
- Batch size > 1 is not supported for generation (yet)
- Batch size=1 => 230 tokens / sec

## Eole:
- Eole uses full python code plus Cuda kernels for RMSNorm, RotaryEmbeddings, Fused Activation
- Batch size=1 => 112 tokens / sec
- Batch size=4 => 359 tokens / sec

## Eole + cudagraphs:
- Batch size=1 => 381 tokens / sec
- **Batch size=4 => 1016 tokens / sec ===========> Winner !**


## vLLM:
- vLLM has many optimization configurations.
- Basically we can compare:
- Without Cudagraph => 350 tokens / sec
- With Cudagraph => 950 tokens / sec
















