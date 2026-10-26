# Paired GDN control projections: RTX 5090 measurements

The experimental implementation packs the dense B/A control weights once during inference setup and replaces eight small GEMVs per GDN layer with two Triton kernels for four-token MTP verification. FP32 accumulation and BF16 outputs are retained; FP32 normalization and the FLA recurrence are unchanged.

## Conditions

Qwen3.8-27B RedHat INT4 decoder, BF16 compute, batch/beam 1, three drafts, context 64000, output cap 512, fused KVQ, prefill chunk 1024, no prefill cache. Torch 2.12.1+cu132, RTX 5090. Two complete warm requests then three measured requests per case. Rates exclude the first prefill token and compilation, using Eole decode timing. Native means the existing eight-GEMV specialization, not a generic four-row GEMM. Vocabulary heads are BF16 unless INT8/INT4 is specified.

This PR is based directly on main 69c3b92d and contains only the control-kernel change. The earlier benchmark matrix below was measured with the existing quantized-head implementation (c6c1ebc1) layered on that base; those head changes are not included here. The full-length user-script comparison was rerun on this kernel-only branch.

## Full-length Orwell script comparison

The unchanged settings and prompt from `recipes/qwen35/test_inference3.py` were imported directly: BF16 heads, three drafts, mode 0, `max_length=4096`, context 64000, top-k 1, top-p 0.95, temperature 0.6, seed 1, and the script's default prefill/cache settings. The script itself was not edited. The baseline only disables preparation of the paired weights; both paths otherwise use the same branch. Two full warm requests preceded three measured requests.

| Path | Median decode tok/s | Decode tokens | Accepted/drafted |
|---|---:|---:|---:|
| Existing | 102.04 | 827 | 482/1038 |
| Paired | 111.62 | 756 | 452/915 |

The observed throughput gain is **9.39%**. The continuation, output length and acceptance change, so this is a user-workload result, not an isolated kernel comparison. Output is repeatable across the three measured requests within each path. Raw logs, configuration, outputs and the harness are in `/tmp/eole-gdn-control-script-ab` on the test host. This is the relevant comparison for the user's ~111.3 tok/s observation; the 147.7/153.3 figures below are for code generation.

## Warmed decoding throughput

| Workload / heads | Existing tok/s | Paired tok/s | Change | Identical output | Acceptance: existing → paired |
|---|---:|---:|---:|---|---|
| fixed-int4-orwell | 209.52 | 226.48 | +8.10% | yes | 383/383 → 383/383 |
| int4-code | 180.06 | 192.86 | +7.11% | yes | 363/442 → 363/442 |
| int4-orwell | 136.57 | 135.96 | -0.45% | no | 313/593 → 299/634 |
| int8-code | 175.35 | 184.47 | +5.20% | no | 365/438 → 366/434 |
| int8-orwell | 125.74 | 126.22 | +0.38% | no | 307/612 → 301/626 |
| mode2-int4-code | 171.88 | 188.39 | +9.61% | no | 362/444 → 368/426 |
| none-code | 147.67 | 153.29 | +3.80% | yes | 363/441 → 363/441 |
| none-orwell | 101.21 | 104.84 | +3.59% | no | 297/637 → 293/652 |
| sampling-int4-code | 169.07 | 180.46 | +6.74% | no | 360/451 → 361/448 |

Each request uses seed 42. Greedy code/Orwell cases use top-k 1. Sampling uses top-k 20, top-p 0.95, temperature 0.6, presence penalty 1.0. Mode 2 is explicitly labeled. Sampling outputs need not match because acceptance affects the random sampling schedule.

The fixed-token diagnostic runs real model computations but forces the same continuation and accepts all drafts. Its absolute throughput is not representative of ordinary generation; its purpose is to isolate execution cost from continuation/acceptance changes.

## Numerical behavior and scope

The kernel is numerically close, not bit-identical to cuBLAS GEMVs. Small BF16 rounding differences can change continuation and acceptance. In particular, INT4 Orwell loses enough acceptance to cancel its execution-speed benefit. Do not advertise the code-generation gain as a universal speedup or assume quality equivalence from these tests alone.

Dispatch requires an evaluated model, NVIDIA compute capability 8.0 or newer (Ampere+), dense unbiased BF16 48×5120 controls, contiguous [1,4,5120] inputs, single GPU, batch/beam 1 and MTP. Unsupported cases retain the prior implementation. Only supported MTP configurations allocate packed weights (~47 MB decimal across 48 layers); these are nonpersistent buffers. No extension rebuild is needed.

## Validation

CPU tests cover setup policy, B/A ordering, fallback behavior and existing MTP/Marlin and verifier-cache integration. CUDA tests cover FP64-reference tolerances, eager execution, fullgraph compilation and changing-input CUDA graph replay. Both RTX 5090 and RTX 4090 are tested and automatic dispatch is enabled on both.

Raw configurations, individual timing repetitions, hashes, outputs and test logs are retained in `/tmp/eole-gdn-control-hardened`; the harness is `run.py` and the paired benchmark command lists are `bench.sh` and `extra.sh`. Earlier profiling and exploratory data are in `/tmp/eole-mtp-deep-20261026`.

All-layer real-weight check against FP64: 48 layers, 384 cases, 147456 outputs. Existing versus paired differs at 29 values, maximum absolute difference 0.001953125; maximum relative RMSE 0.000207342. Against the FP64 result rounded to BF16, existing differs at 20 values and paired at 28. These are arithmetic checks, not a downstream quality benchmark.

## Additional results and checks

For the INT4 code case, total measured request latency including prefill fell from a median 3.149 s to 2.957 s (6.1% lower), versus the 7.1% increase in decode-only throughput. Sampling total request latency fell 6.1% as well. Compilation is excluded from these warmed-request measurements.

In the fixed 512-position continuation, target top-1 predictions differ at two positions: one is an exact tie in the native path, the other in the paired path. This confirms that identical forced outputs do not establish identical model logits.

Validation: 55 CPU integration tests passed (9 CUDA-specific skips, with optional FLA disabled on CPU), 13 CUDA tests passed on each of RTX 5090 and RTX 4090, and Black checks passed. CUDA tests include saving/reloading the verifier and binding a different nonpersistent packed buffer without retracing.

## NVIDIA architecture portability

The paired kernel uses BF16 tensor-core dot products, FP32 accumulation and ordinary Triton reductions. It contains no Blackwell-specific instructions. Dispatch uses NVIDIA compute capability >= 8.0, covering Ampere, Ada, Hopper and Blackwell; pre-Ampere, non-CUDA, unavailable Triton and unsupported shapes retain the prior implementation. The old four-GEMV fallback remains SM120-only and is independent of paired-kernel dispatch.

Hardware measurements and correctness/capture/reload tests have been run on SM89 (RTX 4090) and SM120 (RTX 5090). SM80/86/90/100/103 eligibility is covered by policy tests, not measurements on those GPUs. Performance on those other architectures is not established by these results. The projection shape remains specialized to four-token verification with dense 48×5120 BF16 controls.

RTX 4090 comparison: same user-script configuration except context 32768 and output cap 512 to fit VRAM; BF16 heads, mode 0, two warm requests and three measured requests. Baseline on SM89 is the existing generic four-row GEMM path (the legacy eight-GEMV specialization is SM120-only).

| RTX 4090 workload | Existing tok/s | Paired tok/s | Change | Identical output |
|---|---:|---:|---:|---|
| orwell | 63.24 | 68.12 | +7.72% | no |
| code | 92.45 | 93.62 | +1.26% | yes |

Orwell has 301 accepted drafts in both paths (625 versus 626 drafted); code has identical output and 366/434 acceptance in both paths. All measured repeats within each path have identical output. The 4090 rates use a different context/output cap from the full-length 5090 script comparison and should not be used as a direct GPU-to-GPU comparison. Raw configurations and logs are in `/tmp/eole-gdn-control-arch` on the test host.
