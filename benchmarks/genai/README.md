These scripts show the throughput we can get from various toolkits for the Gemma3 Text 1B LLM.

## Hugging Face / Transformers:
Results are obtained with Flash Attention.
Batch size=1 => 31 tokens / sec
Batch size=4 => 80 tokens / sec

This measure is not 100% accurate because it takes the (very short) time of prefill and do not count for prefill tokens.
Difference would be negligible

## CTranslate2:
The model needs to be converted first.
CT2 is a full C++ inference only toolkit.
Batch size > 1 is not supported for generation (yet)
Batch size=1 => 230 tokens / sec

## Eole:
Eole uses full python code plus Cuda kernels for RMSNorm, RotaryEmbeddings, Fused Activation
Batch size=1 => 97 tokens / sec
BAtch size=4 => 291 tokens / sec

## vLLM:
vLLM has many optimization configurations.
Basically we can compare:
Without Cudagraph => 350 tokens / sec
With Cudagraph => 900 tokens / sec BUT very long start-up time

Even wihout Cudagraph, the full start-end timing is longer than for Eole.



#### Comparison with and without _eole_ops (cuda kernels)

* With eole_ops:
python generate-eole.py 
[2026-01-17 15:30:20,370 INFO] Init single process mode
[2026-01-17 15:30:20,370 INFO] Loading metadata from /mnt/InternalCrucial4/LLM_work/gemma-3-1b-it
[2026-01-17 15:30:20,575 INFO] Building model...
[2026-01-17 15:30:20,624 INFO] Loading data into the model
[2026-01-17 15:30:21,055 INFO] Missing key in safetensors checkpoint: generator.weight
[2026-01-17 15:30:21,055 INFO] └─> Sharing from embeddings matrix since `share_decoder_embeddings` flag is enabled.
[2026-01-17 15:30:25,144 INFO] Initialized tokenizers from HF model: google/gemma-3-1b-it
[2026-01-17 15:30:25,144 INFO] Transforms applied: ['huggingface_tokenize']
[2026-01-17 15:30:25,144 INFO] Build and loading model took 4.77 sec.
[2026-01-17 15:30:35,106 INFO] PRED SCORE: -0.2096, PRED PPL: 1.23 NB SENTENCES: 4
[2026-01-17 15:30:35,106 INFO] ESTIM SCORE: 1.0000, ESTIM PPL: 0.37 NB SENTENCES: 4
[2026-01-17 15:30:35,106 INFO] Step 0 time (s): 0.25
[2026-01-17 15:30:35,106 INFO] Enc/Step 0 tokens / sec: 274.7
[2026-01-17 15:30:35,106 INFO] Subsequent prediction time including all (s): 9.49
[2026-01-17 15:30:35,106 INFO] Average prediction time (ms): 2372.7
## [2026-01-17 15:30:35,106 INFO] Tokens per second: 287.8
[2026-01-17 15:30:35,106 INFO] pred_words_total: 2731.0

* Without eole_ops:
python generate-eole.py 
[2026-01-17 15:31:18,464 INFO] Init single process mode
[2026-01-17 15:31:18,464 INFO] Loading metadata from /mnt/InternalCrucial4/LLM_work/gemma-3-1b-it
[2026-01-17 15:31:18,667 INFO] Building model...
[2026-01-17 15:31:18,716 INFO] Loading data into the model
[2026-01-17 15:31:19,158 INFO] Missing key in safetensors checkpoint: generator.weight
[2026-01-17 15:31:19,158 INFO] └─> Sharing from embeddings matrix since `share_decoder_embeddings` flag is enabled.
[2026-01-17 15:31:23,151 INFO] Initialized tokenizers from HF model: google/gemma-3-1b-it
[2026-01-17 15:31:23,151 INFO] Transforms applied: ['huggingface_tokenize']
[2026-01-17 15:31:23,151 INFO] Build and loading model took 4.69 sec.
[2026-01-17 15:31:44,635 INFO] PRED SCORE: -0.2353, PRED PPL: 1.27 NB SENTENCES: 4
[2026-01-17 15:31:44,635 INFO] ESTIM SCORE: 1.0000, ESTIM PPL: 0.37 NB SENTENCES: 4
[2026-01-17 15:31:44,635 INFO] Step 0 time (s): 0.33
[2026-01-17 15:31:44,636 INFO] Enc/Step 0 tokens / sec: 206.3
[2026-01-17 15:31:44,636 INFO] Subsequent prediction time including all (s): 20.93
[2026-01-17 15:31:44,636 INFO] Average prediction time (ms): 5232.5
## [2026-01-17 15:31:44,636 INFO] Tokens per second: 130.3
[2026-01-17 15:31:44,636 INFO] pred_words_total: 2727.0
















