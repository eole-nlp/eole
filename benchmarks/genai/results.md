
## Hugging Face

################ BATCH SIZE 4 ################################

Number of generated tokens 3346, throughput 80 tok/sec

################ BATCH SIZE 1 ################################

input 0: number of generated tokens 305, throughput 31 tok/sec
input 1: number of generated tokens 1001, throughput 31 tok/sec
input 2: number of generated tokens 958, throughput 31 tok/sec
input 3: number of generated tokens 1131, throughput 31 tok/sec


## Eole
################ BATCH SIZE 4 ################################
[2026-01-20 20:17:42,660 INFO] Init single process mode
[2026-01-20 20:17:42,660 INFO] Loading metadata from gemma-3-1b-it
[2026-01-20 20:17:42,876 INFO] Building model...
[2026-01-20 20:17:42,944 INFO] Loading data into the model
[2026-01-20 20:17:43,403 INFO] Missing key in safetensors checkpoint: generator.weight
[2026-01-20 20:17:43,403 INFO] └─> Sharing from embeddings matrix since `share_decoder_embeddings` flag is enabled.
[2026-01-20 20:17:47,554 INFO] Initialized tokenizers from HF model: google/gemma-3-1b-it
[2026-01-20 20:17:47,554 INFO] Transforms applied: ['huggingface_tokenize']
[2026-01-20 20:17:47,554 INFO] Build and loading model took 4.89 sec.
[2026-01-20 20:17:55,108 INFO] PRED SCORE: -0.2315, PRED PPL: 1.26 NB SENTENCES: 4
[2026-01-20 20:17:55,108 INFO] ESTIM SCORE: 1.0000, ESTIM PPL: 0.37 NB SENTENCES: 4
[2026-01-20 20:17:55,108 INFO] Step 0 time (s): 0.25
[2026-01-20 20:17:55,108 INFO] Enc/Step 0 tokens / sec: 274.9
[2026-01-20 20:17:55,108 INFO] Subsequent prediction time including all (s): 7.09
[2026-01-20 20:17:55,108 INFO] Average prediction time (ms): 1773.5
## [2026-01-20 20:17:55,108 INFO] Tokens per second: 359.2
[2026-01-20 20:17:55,108 INFO] pred_words_total: 2548.0
real	0m15,312s


################ BATCH SIZE 1 ################################
[2026-01-20 20:18:59,307 INFO] Tokens per second: 109.6
[2026-01-20 20:18:59,307 INFO] pred_words_total: 312.0
[2026-01-20 20:19:06,373 INFO] Tokens per second: 111.2
[2026-01-20 20:19:06,373 INFO] pred_words_total: 762.0
[2026-01-20 20:19:12,862 INFO] Tokens per second: 113.5
[2026-01-20 20:19:12,862 INFO] pred_words_total: 713.0
[2026-01-20 20:19:19,929 INFO] Tokens per second: 114.5
[2026-01-20 20:19:19,929 INFO] pred_words_total: 785.0

## CT2
################ BATCH SIZE 1 ################################ 
Number of generated tokens 307, throughput 222 tok/sec
Number of generated tokens 961, throughput 236 tok/sec
Number of generated tokens 953, throughput 236 tok/sec
Number of generated tokens 1072, throughput 237 tok/sec
real	0m21,894s


## vLLM with compilation_config = {"mode": 3, "cudagraph_mode": "NONE"} with CUDAGraph it can go much faster but start-up is longer.
time python generate-vllm.py 
INFO 01-16 15:53:51 [utils.py:253] non-default args: {'max_model_len': 2048, 'gpu_memory_utilization': 0.95, 'disable_log_stats': True, 'compilation_config': {'level': None, 'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': [], 'splitting_ops': None, 'compile_mm_encoder': False, 'compile_sizes': None, 'compile_ranges_split_points': None, 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': None, 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': None, 'pass_config': {}, 'max_cudagraph_capture_size': None, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False}, 'local_cache_dir': None}, 'model': 'google/gemma-3-1b-it'}
INFO 01-16 15:53:53 [model.py:514] Resolved architecture: Gemma3ForCausalLM
INFO 01-16 15:53:53 [model.py:1661] Using max model len 2048
INFO 01-16 15:53:53 [scheduler.py:230] Chunked prefill is enabled with max_num_batched_tokens=8192.
(EngineCore_DP0 pid=614224) INFO 01-16 15:53:56 [core.py:93] Initializing a V1 LLM engine (v0.13.0) with config: model='google/gemma-3-1b-it', speculative_config=None, tokenizer='google/gemma-3-1b-it', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=2048, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False), seed=0, served_model_name=google/gemma-3-1b-it, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'level': None, 'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::kda_attention', 'vllm::sparse_attn_indexer'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_split_points': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'eliminate_noops': True, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False}, 'local_cache_dir': None}
(EngineCore_DP0 pid=614224) INFO 01-16 15:53:58 [parallel_state.py:1203] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://192.168.1.19:40507 backend=nccl
(EngineCore_DP0 pid=614224) INFO 01-16 15:53:58 [parallel_state.py:1411] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank 0
(EngineCore_DP0 pid=614224) INFO 01-16 15:53:58 [gpu_model_runner.py:3562] Starting to load model google/gemma-3-1b-it...
(EngineCore_DP0 pid=614224) INFO 01-16 15:53:59 [cuda.py:351] Using FLASH_ATTN attention backend out of potential backends: ('FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION')
(EngineCore_DP0 pid=614224) INFO 01-16 15:53:59 [weight_utils.py:527] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.01it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.01it/s]
(EngineCore_DP0 pid=614224) 
(EngineCore_DP0 pid=614224) INFO 01-16 15:54:00 [default_loader.py:308] Loading weights took 0.39 seconds
(EngineCore_DP0 pid=614224) INFO 01-16 15:54:00 [gpu_model_runner.py:3659] Model loading took 1.9148 GiB memory and 1.154851 seconds
(EngineCore_DP0 pid=614224) INFO 01-16 15:54:04 [backends.py:643] Using cache directory: $HOME/.cache/vllm/torch_compile_cache/de6a8866bd/rank_0_0/backbone for vLLM's torch.compile
(EngineCore_DP0 pid=614224) INFO 01-16 15:54:04 [backends.py:703] Dynamo bytecode transform time: 3.78 s
(EngineCore_DP0 pid=614224) INFO 01-16 15:54:07 [backends.py:226] Directly load the compiled graph(s) for compile range (1, 8192) from the cache, took 0.700 s
(EngineCore_DP0 pid=614224) INFO 01-16 15:54:07 [monitor.py:34] torch.compile takes 4.48 s in total
(EngineCore_DP0 pid=614224) INFO 01-16 15:54:08 [gpu_worker.py:375] Available KV cache memory: 25.39 GiB
(EngineCore_DP0 pid=614224) WARNING 01-16 15:54:08 [kv_cache_utils.py:1033] Add 2 padding layers, may waste at most 9.09% KV cache memory
(EngineCore_DP0 pid=614224) INFO 01-16 15:54:08 [kv_cache_utils.py:1291] GPU KV cache size: 950,864 tokens
(EngineCore_DP0 pid=614224) INFO 01-16 15:54:08 [kv_cache_utils.py:1296] Maximum concurrency for 2,048 tokens per request: 461.21x
(EngineCore_DP0 pid=614224) INFO 01-16 15:54:08 [gpu_worker.py:449] Compile and warming up model for size 8192
(EngineCore_DP0 pid=614224) 2026-01-16 15:54:08,860 - INFO - autotuner.py:256 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
(EngineCore_DP0 pid=614224) 2026-01-16 15:54:08,867 - INFO - autotuner.py:262 - flashinfer.jit: [Autotuner]: Autotuning process ends
(EngineCore_DP0 pid=614224) WARNING 01-16 15:54:08 [gpu_model_runner.py:4491] Skipping CUDA graph capture. To turn on CUDA graph capture, ensure `cudagraph_mode` was not manually set to `NONE`
(EngineCore_DP0 pid=614224) INFO 01-16 15:54:09 [core.py:259] init engine (profile, create kv cache, warmup model) took 8.45 seconds
INFO 01-16 15:54:10 [llm.py:360] Supported tasks: ['generate']
Adding requests: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 1296.64it/s]
## Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:09<00:00,  2.28s/it, est. speed input: 7.44 toks/s, output: 350.69 toks/s]

real	0m32,966s

