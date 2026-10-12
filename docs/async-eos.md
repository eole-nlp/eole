# Asynchronous EOS lookahead

Set `EOLE_EOS_PIPELINE=1` when running inference to overlap the EOS decision's
GPU-to-CPU transfer with submission of the next decoder/vocabulary step.
The option defaults to off. It preserves the existing EOS list representation,
minimum/maximum output lengths, scores and FP32 normalization.

| Decoding mode | Behavior with the option enabled |
| --- | --- |
| CUDA deterministic greedy, no MTP | One-token lookahead, including batches |
| Single-sequence greedy streaming, no MTP | Lookahead; emitted token copied with EOS flags |
| Greedy with active MTP | Existing verified-chunk EOS handling |
| Beam search, with MTP requested or disabled | Existing beam EOS handling; MTP is not supported by beam search |
| CPU, sampling, attention output, estimator, batched streaming | Existing synchronous handling |

A pinned host buffer and a separate CUDA stream receive the EOS flags. The next
forward is submitted before the CPU waits for those flags. An EOS result discards
the lookahead output, so no extra token is emitted. Shrinking batches filter
lookahead logits and decoder state together. Maximum-length stopping submits no
extra forward. Exception cleanup drains outstanding copies before request-local
buffers can be released.

Early EOS may execute one extra forward; this is bounded lookahead, not a
continuous-batching scheduler. It does not reduce KV-cache capacity requirements.
MTP and beam search deliberately keep their existing paths: MTP commits a
variable accepted prefix and beam search reorders caches before the next forward.
Neither receives a claimed speedup from this option.

Run the search and routing regression tests with:

```sh
python -m unittest eole.tests.test_async_greedy_search \
  eole.tests.test_greedy_search eole.tests.test_beam_search
```

CUDA tests check multiple EOS IDs, minimum/maximum lengths, staggered completion
with shrinking batches, and streaming token copies. Routing tests cover MTP and
beam fallback plus exception cleanup. The synthetic fixed-length EOS-removal
benchmark measured roughly 9% headroom; that result does not measure this
EOS-preserving implementation's gain.
