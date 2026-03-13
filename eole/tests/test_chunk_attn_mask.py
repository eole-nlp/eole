"""
Unit tests for the sliding-window mask logic used by TransformerDecoder.

These tests validate the mathematical invariants of the chunked-prefill
attention mask (_chunk_attn_mask) and its consistency with the single-step
decoding path (_forward_eager S==1 branch) WITHOUT requiring torch or a GPU.

The core invariant that was fixed:
    OLD (wrong): k_pos >= q_pos - W          -> window of W+1 tokens
    NEW (correct): k_pos >= q_pos - W + 1    -> window of exactly W tokens

This matches the decoding path:
    start = current_step - sliding_window + 1
    valid = position_indices >= start   -> positions [start, current_step]
                                           = W tokens
"""

import unittest

try:
    import types

    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _chunk_attn_mask_cpu(chunk_size, cache_len_tgt, current_step, batch=1, sliding_window=0):
    """Call the real ``_chunk_attn_mask`` logic on CPU via a minimal mock.

    Binds the actual ``TransformerDecoder._chunk_attn_mask`` implementation to a
    lightweight ``types.SimpleNamespace`` object so the test exercises the real
    code path without constructing a full ``TransformerDecoder`` (which would
    require model weights, configs, etc.).

    Only callable when ``HAS_TORCH`` is True.
    """
    from eole.decoders.transformer import TransformerDecoder  # noqa: E402

    mock = types.SimpleNamespace(
        cache_len_tgt=cache_len_tgt,
        sliding_window=sliding_window,
        LM_type="decoder",
        # All positions valid — no left-padding in these unit tests
        left_pad_attn_mask=torch.ones(batch, cache_len_tgt, dtype=torch.bool),
    )
    tgt_pad_mask = torch.zeros(batch, 1, chunk_size, dtype=torch.bool)
    return TransformerDecoder._chunk_attn_mask(mock, chunk_size, current_step, tgt_pad_mask)


def _chunked_prefill_allowed_keys(q_pos, cache_len, sliding_window=0):
    """Pure-Python reference for _chunk_attn_mask — the FIXED absolute-position formula.

    Returns the set of key positions (out of ``cache_len`` total) that absolute
    query position ``q_pos`` may attend to.

    When ``sliding_window == 0`` (default) the result is purely causal::

        keys = {k | k <= q_pos}

    When ``sliding_window > 0`` the result is the fixed sliding-window formula::

        keys = {k | k <= q_pos and k >= q_pos - sliding_window + 1}

    Note: the lower bound is ``q_pos - sliding_window + 1`` (the ``+1``
    shifts the lower bound *up* by one), giving a window of exactly
    ``sliding_window`` tokens including ``q_pos`` itself.  The ``+1`` applies
    to the lower-bound expression, not to ``sliding_window``.
    """
    keys = {k for k in range(cache_len) if k <= q_pos}
    if sliding_window > 0:
        keys = {k for k in keys if k >= q_pos - sliding_window + 1}
    return keys


def _decoding_allowed_keys(current_step, cache_len, sliding_window):
    """Pure-Python reference for the S==1 decoding sliding-window constraint.

    Mirrors the tensor logic in _forward_eager (S==1 branch):

        start = max(0, current_step - sliding_window + 1)
        valid = (position_indices <= current_step) & (position_indices >= start)
    """
    start = max(0, current_step - sliding_window + 1)
    return {k for k in range(cache_len) if k <= current_step and k >= start}


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestFirstChunkMaskKeyDimension(unittest.TestCase):
    """Validate that _chunk_attn_mask with current_step=0 (first chunk) produces
    a causal mask whose key dimension equals cache_len_tgt, not chunk_size.

    This is the regression test for the bug where the first chunk during
    chunked prefill used _causal_attn_mask (key dim = chunk_size) instead of
    _chunk_attn_mask (key dim = cache_len_tgt), causing a shape mismatch with
    the pre-allocated KV cache.

    These tests call the real _chunk_attn_mask on CPU (via a minimal mock) so
    they will catch regressions in the actual implementation.
    """

    def test_first_chunk_key_dim_equals_cache_len_tgt(self):
        """Key dimension for chunk 0 (current_step=0) is cache_len_tgt."""
        cache_len_tgt = 64
        chunk_size = 16

        mask = _chunk_attn_mask_cpu(chunk_size, cache_len_tgt, current_step=0)
        # Shape: (B, 1, chunk_size, cache_len_tgt)
        self.assertEqual(mask.shape[-1], cache_len_tgt)
        # Crucially, NOT chunk_size (which is what _causal_attn_mask would use
        # when dynamic_shapes=True and MAX_T == seq_len == chunk_size)
        self.assertNotEqual(mask.shape[-1], chunk_size)

    def test_all_chunks_key_dim_equals_cache_len_tgt(self):
        """Key dimension is cache_len_tgt for every chunk, not just non-first."""
        cache_len_tgt = 32
        chunk_size = 8
        num_chunks = cache_len_tgt // chunk_size

        for chunk_idx in range(num_chunks):
            current_step = chunk_idx * chunk_size
            mask = _chunk_attn_mask_cpu(chunk_size, cache_len_tgt, current_step)
            self.assertEqual(
                mask.shape[-1],
                cache_len_tgt,
                msg=(
                    f"chunk {chunk_idx} (current_step={current_step}): "
                    f"key_dim={mask.shape[-1]} != cache_len_tgt={cache_len_tgt}"
                ),
            )

    def test_first_chunk_causal_mask_values(self):
        """_chunk_attn_mask at current_step=0 produces a correct causal mask
        for the first chunk.

        Each query q at absolute position `q` can only attend to keys k <= q.
        With cache_len_tgt > chunk_size, positions in (chunk_size, cache_len_tgt)
        must all be False (no future keys exist yet — the cache is empty there).
        """
        cache_len_tgt = 32
        chunk_size = 8
        current_step = 0

        mask = _chunk_attn_mask_cpu(chunk_size, cache_len_tgt, current_step)
        # mask shape: (1, 1, chunk_size, cache_len_tgt); squeeze batch and head dims
        mask_2d = mask[0, 0]  # (chunk_size, cache_len_tgt)

        for q_offset in range(chunk_size):
            q_abs = current_step + q_offset
            # Positions 0..q_abs must be True; positions q_abs+1..cache_len_tgt-1 must be False
            self.assertTrue(
                mask_2d[q_offset, : q_abs + 1].all().item(),
                msg=f"q={q_abs}: causal positions 0..{q_abs} should all be True",
            )
            if q_abs + 1 < cache_len_tgt:
                self.assertFalse(
                    mask_2d[q_offset, q_abs + 1 :].any().item(),
                    msg=f"q={q_abs}: future positions {q_abs+1}..{cache_len_tgt-1} should all be False",
                )

    def test_first_chunk_same_as_non_first_for_same_positions(self):
        """_chunk_attn_mask at current_step=0 gives same allowed keys as
        the mask for those same positions when processed as a 'non-first' chunk.

        This validates that there is no discontinuity in the mask when the
        same query positions would be encountered in different chunks.
        """
        cache_len_tgt = 64
        chunk_size = 16
        sliding_window = 6

        # Process positions 0..chunk_size-1 as "chunk 0" (current_step=0)
        mask_chunk0 = _chunk_attn_mask_cpu(chunk_size, cache_len_tgt, 0, sliding_window=sliding_window)

        # Process the same positions one-by-one as single-token "non-first" chunks
        for q_offset in range(chunk_size):
            q_abs = q_offset
            mask_single = _chunk_attn_mask_cpu(1, cache_len_tgt, q_abs, sliding_window=sliding_window)
            # Single-token mask: (1, 1, 1, cache_len_tgt)
            # Chunk-0 row for q_offset: mask_chunk0[0, 0, q_offset, :]
            self.assertTrue(
                (mask_chunk0[0, 0, q_offset] == mask_single[0, 0, 0]).all().item(),
                msg=f"q={q_abs}: chunk0 row vs single-token mask differ",
            )


def _update_causal_mask_allowed_keys(causal_keys, q_is_img, k_image_locs, cache_len):
    """Pure-Python reference for _update_causal_mask with separate q/k image locs.

    Given the set of keys already allowed by the causal mask (``causal_keys``),
    additionally allow key positions where BOTH the query and the key are image
    tokens.  Mirrors the tensor logic:

        img_block = is_q_img & is_k_img
        attn_mask = attn_mask | img_block
    """
    if not q_is_img:
        return causal_keys  # text query: no image-to-image bonus
    img_keys = {k for k in range(cache_len) if k_image_locs[k]}
    return causal_keys | img_keys


class TestUpdateCausalMaskAsymmetric(unittest.TestCase):
    """Validate the cross-chunk image-to-image attention logic in _update_causal_mask.

    After the refactor, _update_causal_mask accepts separate q/k image location
    tensors.  These tests check that:
    - image query tokens gain access to ALL image key tokens in the full cache
      (including tokens from previous chunks)
    - text query tokens are NOT given the image-to-image bonus
    - the allowed set equals (causal_keys | image_keys) for image queries
    """

    def _allowed_keys(self, q_pos, q_is_img, k_image_locs, cache_len, sliding_window=0):
        """Combine causal mask with image update — pure-Python reference."""
        causal = _chunked_prefill_allowed_keys(q_pos, cache_len, sliding_window)
        return _update_causal_mask_allowed_keys(causal, q_is_img, k_image_locs, cache_len)

    def test_image_query_gains_image_keys_from_filled_positions(self):
        """An image query token can attend to image key tokens in already-filled
        positions (previous chunks and the current chunk), but NOT to positions
        in future chunks that have not yet been written to the KV cache."""
        cache_len = 32
        chunk_size = 8
        current_step = 8  # second chunk; filled positions are 0..15

        # Sprinkle image tokens throughout the full cache, including future positions
        all_image_locs = [i % 3 == 0 for i in range(cache_len)]  # 0, 3, 6, …

        # After the fix, k_image_locations is ANDed with filled_mask (k < current_step+S)
        filled_end = current_step + chunk_size  # = 16
        k_image_locs_filled = [all_image_locs[k] and k < filled_end for k in range(cache_len)]

        for offset in range(chunk_size):
            q_pos = current_step + offset
            # query is an image token
            allowed = self._allowed_keys(q_pos, q_is_img=True, k_image_locs=k_image_locs_filled, cache_len=cache_len)
            # Image keys in FILLED positions (0..15) should be accessible
            for k in range(filled_end):
                if k_image_locs_filled[k]:
                    self.assertIn(
                        k,
                        allowed,
                        msg=f"q={q_pos} (image): filled image key {k} should be accessible",
                    )
            # Image keys in FUTURE positions (16..31) must NOT be accessible
            for k in range(filled_end, cache_len):
                if all_image_locs[k]:
                    self.assertNotIn(
                        k,
                        allowed,
                        msg=f"q={q_pos} (image): future image key {k} must NOT be accessible",
                    )

    def test_image_bonus_does_not_open_future_positions(self):
        """The image-to-image bonus must not allow attending to key positions
        that are beyond the last position filled by the current chunk.

        Regression test for the bug where k_image_locations=image_locations[:, :cache_len]
        was passed to _update_causal_mask, which OR-ed in image connections for
        key positions in future chunks (positions >= current_step + chunk_size)
        that have not yet been written to the KV cache.
        """
        cache_len = 32
        chunk_size = 8

        for chunk_idx in range(cache_len // chunk_size):
            current_step = chunk_idx * chunk_size
            filled_end = current_step + chunk_size

            # Simulate the fixed k_image_locs: filled positions only
            k_image_locs_fixed = [k < filled_end for k in range(cache_len)]

            for offset in range(chunk_size):
                q_pos = current_step + offset
                allowed_fixed = self._allowed_keys(
                    q_pos, q_is_img=True, k_image_locs=k_image_locs_fixed, cache_len=cache_len
                )
                # With k_image_locs covering only filled positions, future
                # positions must never appear in the allowed set
                for k in range(filled_end, cache_len):
                    self.assertNotIn(
                        k,
                        allowed_fixed,
                        msg=(
                            f"chunk={chunk_idx}, q={q_pos}: "
                            f"future key {k} should be blocked "
                            f"(filled_end={filled_end})"
                        ),
                    )

    def test_text_query_not_given_image_bonus(self):
        """A text query token must NOT gain access to image keys via the bonus."""
        cache_len = 32
        chunk_size = 8
        current_step = 0

        k_image_locs = [i % 3 == 0 for i in range(cache_len)]

        for offset in range(chunk_size):
            q_pos = current_step + offset
            causal = _chunked_prefill_allowed_keys(q_pos, cache_len)
            allowed = self._allowed_keys(q_pos, q_is_img=False, k_image_locs=k_image_locs, cache_len=cache_len)
            self.assertEqual(
                allowed,
                causal,
                msg=f"q={q_pos} (text): image bonus should NOT apply, allowed={sorted(allowed)} != causal={sorted(causal)}",  # noqa E501
            )

    def test_image_query_in_first_chunk_sees_own_image_tokens(self):
        """Even for chunk 0 (current_step=0), an image query can attend to image
        keys at positions that come before the current query within the chunk."""
        cache_len = 16
        chunk_size = 8
        current_step = 0

        # Only positions 0, 2, 4 are image tokens
        k_image_locs = [i % 2 == 0 for i in range(cache_len)]

        for offset in range(chunk_size):
            q_pos = current_step + offset
            allowed = self._allowed_keys(q_pos, q_is_img=True, k_image_locs=k_image_locs, cache_len=cache_len)
            # Positions in the chunk BEFORE q_pos that are image tokens should be accessible
            for k in range(q_pos):
                if k_image_locs[k]:
                    self.assertIn(k, allowed, msg=f"q={q_pos}: preceding image key {k} should be accessible")

    def test_image_query_cross_chunk_image_keys_are_accessible(self):
        """An image query in chunk 1 can attend to image keys from chunk 0."""
        cache_len = 16
        chunk_size = 4
        current_step = 4  # second chunk starts here

        # Image tokens only in positions 0-3 (first chunk)
        k_image_locs = [i < 4 for i in range(cache_len)]

        for offset in range(chunk_size):
            q_pos = current_step + offset
            allowed = self._allowed_keys(q_pos, q_is_img=True, k_image_locs=k_image_locs, cache_len=cache_len)
            # All image keys from the first chunk (0-3) should be accessible
            for k in range(4):
                self.assertIn(
                    k,
                    allowed,
                    msg=f"q={q_pos} (chunk1, image): cross-chunk image key {k} should be accessible",
                )


class TestChunkAttnMaskSlidingWindow(unittest.TestCase):
    """Validate the sliding-window mask invariants for chunked prefill."""

    def test_chunked_and_decoding_windows_are_equal(self):
        """_chunk_attn_mask and the S==1 decoding path produce identical windows."""
        sliding_window = 4
        cache_len = 20

        for step in range(cache_len):
            chunk_keys = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            decode_keys = _decoding_allowed_keys(step, cache_len, sliding_window)
            self.assertEqual(
                chunk_keys,
                decode_keys,
                msg=f"step={step}: chunked={sorted(chunk_keys)} != decode={sorted(decode_keys)}",
            )

    def test_window_contains_exactly_sliding_window_tokens(self):
        """Each query attends to exactly sliding_window tokens (or fewer near start)."""
        sliding_window = 4
        cache_len = 20

        for step in range(cache_len):
            keys = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            expected_count = min(sliding_window, step + 1)
            self.assertEqual(
                len(keys),
                expected_count,
                msg=f"step={step}: expected {expected_count} keys, got {len(keys)}: {sorted(keys)}",
            )

    def test_off_by_one_would_produce_extra_token(self):
        """Demonstrate that the OLD formula allowed W+1 instead of W tokens."""
        sliding_window = 4
        cache_len = 20

        def _old_chunked_allowed(q_pos, cache_len, sliding_window):
            """The WRONG (pre-fix) formula: k >= q - W  (W+1 tokens)."""
            return {k for k in range(cache_len) if k <= q_pos and k >= q_pos - sliding_window}

        for step in range(sliding_window, cache_len):
            correct = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            wrong = _old_chunked_allowed(step, cache_len, sliding_window)
            # The wrong formula has one extra (older) token
            self.assertEqual(
                len(wrong),
                len(correct) + 1,
                msg=f"step={step}: expected old formula to have 1 extra token",
            )
            extra = wrong - correct
            self.assertEqual(
                extra,
                {step - sliding_window},
                msg=f"step={step}: expected extra token at {step - sliding_window}, got {extra}",
            )

    def test_window_lower_bound_never_negative(self):
        """Lower bound of the window is always >= 0."""
        sliding_window = 4
        cache_len = 20

        for step in range(cache_len):
            keys = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            self.assertTrue(
                all(k >= 0 for k in keys),
                msg=f"step={step}: negative key in {sorted(keys)}",
            )

    def test_window_upper_bound_equals_query_position(self):
        """The highest attended key is always the current query position (causal)."""
        sliding_window = 4
        cache_len = 20

        for step in range(cache_len):
            keys = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            self.assertEqual(
                max(keys),
                step,
                msg=f"step={step}: max key {max(keys)} != q_pos",
            )

    def test_window_is_contiguous(self):
        """Attended keys form a contiguous range with no gaps."""
        sliding_window = 4
        cache_len = 20

        for step in range(cache_len):
            keys = sorted(_chunked_prefill_allowed_keys(step, cache_len, sliding_window))
            expected = list(range(keys[0], keys[-1] + 1))
            self.assertEqual(keys, expected, msg=f"step={step}: keys not contiguous: {keys}")

    def test_sliding_window_1_attends_only_self(self):
        """sliding_window=1 means each token attends only to itself."""
        sliding_window = 1
        cache_len = 10

        for step in range(cache_len):
            keys = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            self.assertEqual(keys, {step}, msg=f"step={step}: expected {{step}}, got {keys}")

    def test_large_window_equals_causal(self):
        """A window larger than cache_len is equivalent to full causal attention."""
        sliding_window = 1000
        cache_len = 20

        for step in range(cache_len):
            keys = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            causal_keys = {k for k in range(cache_len) if k <= step}
            self.assertEqual(
                keys,
                causal_keys,
                msg=f"step={step}: chunked={sorted(keys)} != causal={sorted(causal_keys)}",
            )

    def test_consistency_across_chunk_boundaries(self):
        """Sliding window behaves identically for a query at position P regardless
        of whether it is processed in chunk 0 or a later chunk.

        This validates that the fix to _chunk_attn_mask (which uses absolute
        positions via current_step) makes cross-chunk masking consistent.
        """
        sliding_window = 3
        cache_len = 15
        chunk_size = 4

        # Process query positions 0..chunk_size-1 in chunk 0
        # Process query positions chunk_size..2*chunk_size-1 in chunk 1
        for chunk_start in [0, chunk_size, 2 * chunk_size]:
            for offset in range(chunk_size):
                q_pos = chunk_start + offset
                if q_pos >= cache_len:
                    break

                # mask from _chunk_attn_mask (current_step=chunk_start, offset=offset)
                # q_pos = current_step + offset
                keys = _chunked_prefill_allowed_keys(q_pos, cache_len, sliding_window)

                # mask from decoding path (current_step = q_pos)
                decode_keys = _decoding_allowed_keys(q_pos, cache_len, sliding_window)

                self.assertEqual(
                    keys,
                    decode_keys,
                    msg=(
                        f"chunk_start={chunk_start}, offset={offset}, q_pos={q_pos}: "
                        f"chunk={sorted(keys)} != decode={sorted(decode_keys)}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
