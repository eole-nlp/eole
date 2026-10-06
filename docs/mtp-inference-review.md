# MTP inference: review and commit structure

Reviewed against the local `main` branch on 2026-10-06.

## Functional commits

1. Native MTP heads and Qwen causal-context refresh: configuration, checkpoint projection layout, prompt priming, shifted verified target inputs, recurrent drafting and causal-prefix tests.
2. GDN speculative transactions: verifier buffers, accepted-prefix state replay, grouped copies, reusable packing storage and compiled short convolution.
3. Greedy speculative inference: one target verification pass per chunk, dynamic cache reservation, rollback, token streaming, constrained-input fallback, phase timing and actual speculative-loop tests.
4. FLA compilation compatibility: inference custom-operator boundaries for convolution cache mutation and gated RMSNorm, preserving the selected FLA kernels and training behavior.
5. Backend diagnostics: selected implementations and package versions, compile configuration and graph-capture caveats.

The development experiment and its revert are absent from the rewritten history. The original history is preserved on a local backup branch.

## Review fixes

- Speculative test fixtures previously set min_length=100, disabling speculation. They now use min_length=0, suppress EOS only inside the test fixture, and assert that drafting actually ran.
- The corrected tests exposed generated PAD-token handling: sequential decoding consumed those tokens but chunk verification masked them as padding. Generated tokens are now valid cache inputs regardless of token ID; prompt/scoring padding remains masked.
- Failed requests restore MTP attention caches and discard speculative decoder state. A regression covers failure after MTP priming.
- FlashAttention MTP priming no longer allocates an unused prompt-length by cache-capacity causal mask.
- Speculative buffers are invalidated when state dtype/device changes. Grouped replay checks all input shapes, dtypes and devices before combining layers.
- Image inputs fall back to ordinary decoding: MTP priming currently assumes text tokens align with decoder hidden positions.

Earlier fixes retained in the functional commits include forced-prefix fallback, streamed token returns, full-chunk dynamic KV reservation, bonus-position context refresh and exclusion of speculative hidden states from the persistent target prefix.

## Verification and limits

34 CPU fallback tests and 2 FLA compile-boundary tests pass. The latter replace GPU-only FLA calls with compiler-disabled stand-ins and verify the fullgraph integration and convolution cache mutation. GPU kernel execution is not validated by those tests.

CPU float32 GDN and Qwen integration tests now exercise speculation and match ordinary greedy output, including the generated PAD-token fixture. Multiple Qwen proposal cycles are compared to full causal-prefix computations across full and partial acceptance.

BF16 chunk recurrence still has different rounding boundaries from sequential recurrence. Compiled stencil and batched quantized projections can also change rounding. Exact token parity for the production BF16/Marlin/Flash checkpoint is not established. These differences were preserved rather than changing state precision and the established baseline during this history cleanup.

The verifier is still one target forward per cycle. Partial GDN state commits replay only the accepted recurrent prefix, not the full target model. Unsupported sampling/constraints use ordinary decoding. Backend logs describe selected paths and configured CUDA graphs; they do not certify successful graph capture or identify PyTorch's per-call SDPA dispatch.

Use repeated warm runs with identical prompts and fixed decode lengths when comparing throughput. No additional GPU speedup is claimed by this review.
