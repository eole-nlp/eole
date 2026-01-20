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

## vLLM:
- vLLM has many optimization configurations.
- Basically we can compare:
- Without Cudagraph => 350 tokens / sec
- With Cudagraph => 900 tokens / sec BUT very long start-up time

Even wihout Cudagraph, the full start-end timing is longer than for Eole.



#### Comparison with and without _eole_ops (cuda kernels)

* With eole_ops:
- python generate-eole.py
- [2026-01-20 20:17:42,660 INFO] Init single process mode
- [2026-01-20 20:17:42,660 INFO] Loading metadata from gemma-3-1b-it
- [2026-01-20 20:17:42,876 INFO] Building model...
- [2026-01-20 20:17:42,944 INFO] Loading data into the model
- [2026-01-20 20:17:43,403 INFO] Missing key in safetensors checkpoint: generator.weight
- [2026-01-20 20:17:43,403 INFO] └─> Sharing from embeddings matrix since `share_decoder_embeddings` flag is enabled.
- [2026-01-20 20:17:47,554 INFO] Initialized tokenizers from HF model: google/gemma-3-1b-it
- [2026-01-20 20:17:47,554 INFO] Transforms applied: ['huggingface_tokenize']
- [2026-01-20 20:17:47,554 INFO] Build and loading model took 4.89 sec.
- [2026-01-20 20:17:55,108 INFO] PRED SCORE: -0.2315, PRED PPL: 1.26 NB SENTENCES: 4
- [2026-01-20 20:17:55,108 INFO] ESTIM SCORE: 1.0000, ESTIM PPL: 0.37 NB SENTENCES: 4
- [2026-01-20 20:17:55,108 INFO] Step 0 time (s): 0.25
- [2026-01-20 20:17:55,108 INFO] Enc/Step 0 tokens / sec: 274.9
- [2026-01-20 20:17:55,108 INFO] Subsequent prediction time including all (s): 7.09
- [2026-01-20 20:17:55,108 INFO] Average prediction time (ms): 1773.5
## [2026-01-20 20:17:55,108 INFO] Tokens per second: 359.2
- [2026-01-20 20:17:55,108 INFO] pred_words_total: 2548.0

* Without eole_ops:
- python generate-eole.py 
- [2026-01-20 20:25:03,140 INFO] Init single process mode
- [2026-01-20 20:25:03,140 INFO] Loading metadata from gemma-3-1b-it
- [2026-01-20 20:25:03,342 INFO] Building model...
- [2026-01-20 20:25:03,413 INFO] Loading data into the model
- [2026-01-20 20:25:03,885 INFO] Missing key in safetensors checkpoint: generator.weight
- [2026-01-20 20:25:03,885 INFO] └─> Sharing from embeddings matrix since `share_decoder_embeddings` flag is enabled.
- [2026-01-20 20:25:08,180 INFO] Initialized tokenizers from HF model: google/gemma-3-1b-it
- [2026-01-20 20:25:08,181 INFO] Transforms applied: ['huggingface_tokenize']
- [2026-01-20 20:25:08,181 INFO] Build and loading model took 5.04 sec.
- [2026-01-20 20:25:30,091 INFO] PRED SCORE: -0.2256, PRED PPL: 1.25 NB SENTENCES: 4
- [2026-01-20 20:25:30,091 INFO] ESTIM SCORE: 1.0000, ESTIM PPL: 0.37 NB SENTENCES: 4
- [2026-01-20 20:25:30,091 INFO] Step 0 time (s): 0.37
- [2026-01-20 20:25:30,091 INFO] Enc/Step 0 tokens / sec: 185.0
- [2026-01-20 20:25:30,091 INFO] Subsequent prediction time including all (s): 21.33
- [2026-01-20 20:25:30,091 INFO] Average prediction time (ms): 5332.8
## [2026-01-20 20:25:30,091 INFO] Tokens per second: 132.5
- [2026-01-20 20:25:30,091 INFO] pred_words_total: 2826.0
















