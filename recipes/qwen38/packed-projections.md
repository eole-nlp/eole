# Packed Marlin projections

`fuse_gate: true` packs compatible Marlin MLP gate/up projections. `fuse_kvq: true`
also packs compatible GDN QKV/Z projections, in addition to attention KVQ.
Both options default to false. Packing happens during inference preparation,
before Marlin weight repacking; incompatible pairs retain separate projections
with a warning. MoE expert stacking keeps its existing initialization path.

The paired BF16 GDN A/B kernel now handles one-token decoding automatically,
as well as four-token MTP verification. Its specialized path requires a single
GPU, batch/beam size one, BF16 inputs, and the supported 5120-to-96 shape on
Ampere or newer. Other configurations retain their existing projection path.
The prepared A/B buffers use about 47 MB across 48 GDN layers.

No checkpoint conversion or native-extension rebuild is required. The feature
uses existing Marlin kernels and a lazily compiled Triton one-token kernel.
FP32 normalization and BF16 vocabulary heads remain unchanged. This change does
not include cache-view, recurrent-state commit, or vocabulary quantization work.

## RTX 5090 measurements

Baseline: main `84f99d405ab6e560f880256a64a812b7ee2308f0`.
Implementation: `077e1ae4`. RedHat Qwen3.8-27B INT4, BF16 compute and vocabulary
heads, PyTorch 2.12.1+cu132, compile mode 0, FlashAttention, batch/beam one,
64k reserved context and 1024-token prefill chunks. Each process ran two warm
requests followed by three measured requests, with seed 42 and greedy decoding.
The Orwell prompt contains 25 tokens; the output cap is 512 tokens.

| Implementation | MTP | fuse_kvq | fuse_gate | Decode tok/s | Change vs main | Draft acceptance |
|---|---|---|---|---:|---:|---:|
| main | off | true | false | 76.85 | — | — |
| PR | off | true | false | 80.58 | +4.8% | — |
| PR | off | true | true | 81.64 | +6.2% | — |
| main | 3 drafts | true | false | 107.61 | — | 44.6% |
| PR | 3 drafts | true | false | 112.94 | +5.0% | 47.7% |
| PR | 3 drafts | true | true | 112.79 | +4.8% | 46.0% |

The baseline uses `fuse_gate: false` because its Marlin gate/up fusion is not
supported. Each measurement counts 511 delivered decode tokens, excluding the
first token and prefill/startup time. MTP timing includes drafting, verification
and state commit. Phase-specific CUDA events were disabled for both versions.

Outputs repeat within each configuration but differ between configurations:
packing and the paired reduction can change floating-point rounding and greedy
continuations. MTP acceptance changes as well, so these are observed end-to-end
results, not an isolated estimate of kernel savings. Gate packing adds about
1.3% beyond the other changes without MTP in this run; it gives no additional
MTP gain here. No task-quality benchmark or other-architecture throughput claim
is made. Full configurations, outputs, hashes and individual timings are in
[the results file](packed-projections-results.json).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
  python recipes/qwen38/benchmark_marlin.py \
  --model "$QWEN38_MODEL" --fuse-gate --output /tmp/packed-no-mtp.json
```

Add `--mtp` for three-draft MTP. Omit `--fuse-gate` to keep gate/up separate.
`--no-fuse-kvq` disables attention KVQ and GDN QKV/Z packing; the automatic
paired A/B dispatch is independent of that flag.

## Validation

CPU coverage includes inference wiring, incompatible-pair fallback, missing
configuration fields, dense fusion and MoE initialization. CUDA coverage checks
packed projection outputs, compiled execution, changing-input graph replay,
and the paired A/B kernel against FP64 references on RTX 5090 and RTX 4090.
Short compile-mode-2 runs cover ordinary decoding and MTP; these are compatibility
checks rather than throughput comparisons with the mode-0 measurements above.

The CPU suite passed 752 tests (39 skipped). The focused CUDA suite passed 32
tests on each GPU, including INT4/INT8 packed projections in FP16 and BF16.
Full-model compiled smoke runs passed for batch four, beam four, worker-thread
streaming, MTP sampling, and two-GPU tensor parallel (RTX 5090 + RTX 4090), with
both fusion flags enabled. These smoke checks establish execution compatibility,
not throughput gains for those configurations.
