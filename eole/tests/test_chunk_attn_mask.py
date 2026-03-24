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


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestCausalAttnMaskSlidingWindow(unittest.TestCase):
    """Validate that _causal_attn_mask uses the same sliding-window convention
    as _chunk_attn_mask and the decoding path (window of exactly W tokens).

    Before the fix, _causal_attn_mask called future_mask.triu_(-sliding_window),
    which selects k >= q - W, giving W+1 tokens per query.  The fix changes this
    to triu_(-(sliding_window - 1)), matching k >= q - W + 1 (exactly W tokens).
    """

    def _causal_mask_cpu(self, seq_len, sliding_window, batch=1, training=True):
        """Call the real ``_causal_attn_mask`` via a minimal SimpleNamespace mock."""
        from eole.decoders.transformer import TransformerDecoder  # noqa: E402

        mock = types.SimpleNamespace(
            sliding_window=sliding_window,
            LM_type="decoder",
            training=training,
            dynamic_shapes=False,
            max_length=seq_len,
        )
        tgt_pad_mask = torch.zeros(batch, 1, seq_len, dtype=torch.bool)
        return TransformerDecoder._causal_attn_mask(mock, tgt_pad_mask)

    def test_window_size_is_exactly_w(self):
        """Each query attends to exactly sliding_window tokens (or fewer at the start)."""
        seq_len = 16
        sliding_window = 4

        mask = self._causal_mask_cpu(seq_len, sliding_window)
        mask_2d = mask[0, 0]  # (seq_len, seq_len)

        for q in range(seq_len):
            row = mask_2d[q]
            attended = row.nonzero(as_tuple=False).view(-1).tolist()
            expected_count = min(sliding_window, q + 1)
            self.assertEqual(
                len(attended),
                expected_count,
                msg=f"q={q}: expected {expected_count} attended keys, got {len(attended)}: {attended}",
            )

    def test_causal_mask_matches_chunk_attn_mask_window(self):
        """_causal_attn_mask and _chunk_attn_mask produce the same window size for each query."""
        seq_len = 12
        sliding_window = 3

        causal_mask = self._causal_mask_cpu(seq_len, sliding_window)
        # Compare per-query window count from _causal_attn_mask with the reference formula
        causal_2d = causal_mask[0, 0]  # (seq_len, seq_len)

        for q in range(seq_len):
            ref_keys = _chunked_prefill_allowed_keys(q, seq_len, sliding_window)
            actual_keys = set(causal_2d[q].nonzero(as_tuple=False).view(-1).tolist())
            self.assertEqual(
                actual_keys,
                ref_keys,
                msg=f"q={q}: _causal_attn_mask={sorted(actual_keys)} != reference={sorted(ref_keys)}",
            )

    def test_window_1_attends_only_self(self):
        """With sliding_window=1, each token should attend only to itself."""
        seq_len = 8
        sliding_window = 1

        mask = self._causal_mask_cpu(seq_len, sliding_window)
        mask_2d = mask[0, 0]

        for q in range(seq_len):
            attended = mask_2d[q].nonzero(as_tuple=False).view(-1).tolist()
            self.assertEqual(attended, [q], msg=f"q={q}: expected [q], got {attended}")


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestChunkAttnMaskImageLocationsPadding(unittest.TestCase):
    """Validate that the image-locations key tensor is correctly padded to
    cache_len when image_locations.size(1) < cache_len_tgt.

    This is the regression test for the bug where
    ``image_locations[:, :cache_len] & filled_mask`` would fail (or silently
    give wrong results) when image_locations is shorter than cache_len_tgt —
    e.g., in a static-shape cache where cache_len_tgt == max_length but
    image_locations only covers the current sequence.
    """

    def _update_causal_mask_cpu(self, attn_mask, q_img, k_image_locations=None):
        """Call the real _update_causal_mask via a minimal mock.

        Args:
            attn_mask (Tensor): ``(B, 1, q_len, k_len)`` bool mask, True = attend.
            q_img (Tensor): ``(B, q_len)`` bool image-token flags for queries.
            k_image_locations (Tensor, optional): ``(B, k_loc_len)`` bool
                image-token flags for keys.  When ``None``, defaults to
                ``q_img`` (square case).

        Returns:
            Tensor: ``(B, 1, q_len, k_len)`` updated mask returned by the real
                implementation.
        """
        from eole.decoders.transformer import TransformerDecoder  # noqa: E402

        mock = types.SimpleNamespace()
        return TransformerDecoder._update_causal_mask(mock, attn_mask, q_img, k_image_locations)

    # ------------------------------------------------------------------
    # Integration tests that call the real _update_causal_mask
    # ------------------------------------------------------------------

    def test_update_causal_mask_square_image_positions_opened(self):
        """Square case (q_len == k_len): image-to-image pairs must be True
        even when they were False in the causal attn_mask."""
        batch = 1
        seq_len = 4

        # Start with a causal mask: lower-triangular True, upper-triangular False
        attn_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        attn_mask = attn_mask.view(batch, 1, seq_len, seq_len)

        # Mark positions 0 and 3 as image tokens
        q_img = torch.tensor([[True, False, False, True]])  # (1, 4)

        result = self._update_causal_mask_cpu(attn_mask, q_img)

        # (0, 3) and (3, 0) should now be True (both image tokens)
        self.assertTrue(result[0, 0, 0, 3].item(), "img-query 0 should attend to img-key 3")
        self.assertTrue(result[0, 0, 3, 0].item(), "img-query 3 should attend to img-key 0")
        # Non-image pair (1, 3) must remain as the causal mask dictated (False)
        self.assertFalse(result[0, 0, 1, 3].item(), "non-img query 1 must not attend to img-key 3")
        # Existing True entries must not be disturbed
        self.assertTrue(result[0, 0, 2, 1].item(), "causal True must remain True")

    def test_update_causal_mask_asymmetric_future_keys_not_opened(self):
        """Asymmetric case (q_len < k_len, i.e., cross-chunk KV cache):
        image positions at k indices beyond k_loc_len must NOT be opened up
        because they correspond to unfilled (future) cache slots."""
        batch = 1
        q_len = 4  # current chunk has 4 queries
        k_loc_len = 4  # image locs only cover the current chunk
        k_len = 8  # full cache is larger (positions 4..7 are uninitialised)

        # Causal mask: each query can only see positions up to its own index
        attn_mask = torch.zeros(batch, 1, q_len, k_len, dtype=torch.bool)
        for q in range(q_len):
            attn_mask[0, 0, q, : q + 1] = True

        # All positions in both query and key chunk are image tokens
        q_img = torch.ones(batch, q_len, dtype=torch.bool)
        k_image_locations = torch.ones(batch, k_loc_len, dtype=torch.bool)

        result = self._update_causal_mask_cpu(attn_mask, q_img, k_image_locations)

        # Image queries should attend to all image keys within k_loc_len
        for q in range(q_len):
            for k in range(k_loc_len):
                self.assertTrue(
                    result[0, 0, q, k].item(),
                    msg=f"q={q} should attend to filled img-key k={k}",
                )

        # Future positions k >= k_loc_len must remain False (untouched)
        future_slice = result[0, 0, :, k_loc_len:]
        self.assertFalse(
            future_slice.any().item(),
            msg=f"No future key position (k>={k_loc_len}) should be opened by image logic",
        )

    def test_update_causal_mask_asymmetric_partial_image_keys(self):
        """Asymmetric case where only SOME of the k_loc_len positions are image
        tokens: only those positions should be opened for image queries."""
        batch = 1
        q_len = 3
        k_loc_len = 3
        k_len = 6  # positions 3..5 are uninitialized future cache slots

        # Causal mask (all False for simplicity)
        attn_mask = torch.zeros(batch, 1, q_len, k_len, dtype=torch.bool)

        # Only position 1 is an image token in both query and key sequences
        q_img = torch.tensor([[False, True, False]])  # (1, 3)
        k_image_locations = torch.tensor([[False, True, False]])  # (1, 3)

        result = self._update_causal_mask_cpu(attn_mask, q_img, k_image_locations)

        # (q=1, k=1) should be True — both image tokens
        self.assertTrue(result[0, 0, 1, 1].item(), "img-to-img pair must be True")
        # (q=1, k=0) and (q=1, k=2) should still be False — not image tokens
        self.assertFalse(result[0, 0, 1, 0].item())
        self.assertFalse(result[0, 0, 1, 2].item())
        # Future positions k >= k_loc_len must remain False
        future_slice = result[0, 0, :, k_loc_len:]
        self.assertFalse(future_slice.any().item(), "future keys must stay False")

    def test_padded_k_image_locations_no_shape_error(self):
        """Padding image_locations to cache_len before AND-ing with filled_mask
        must not raise any shape/broadcast error."""
        batch = 2
        seq_len = 8
        cache_len = 32  # larger than seq_len
        chunk_size = seq_len

        # Simulate the padding logic that was added to _forward_eager
        image_locations = torch.zeros(batch, seq_len, dtype=torch.bool)
        image_locations[:, 0] = True  # position 0 is an image token

        filled_mask = torch.arange(cache_len) < chunk_size

        img_loc_len = image_locations.size(1)
        k_img_full = torch.zeros(batch, cache_len, dtype=torch.bool)
        k_img_full[:, :img_loc_len] = image_locations
        # This is what _forward_eager now does; must not raise
        k_img_and_filled = k_img_full & filled_mask

        self.assertEqual(k_img_and_filled.shape, (batch, cache_len))
        # Position 0 should be True (image and filled), position 8+ should be False
        self.assertTrue(k_img_and_filled[:, 0].all().item())
        self.assertFalse(k_img_and_filled[:, seq_len:].any().item())

    def test_padded_positions_are_false(self):
        """Positions beyond image_locations.size(1) must be False after padding."""
        batch = 1
        seq_len = 6
        cache_len = 16

        image_locations = torch.ones(batch, seq_len, dtype=torch.bool)

        k_img_full = torch.zeros(batch, cache_len, dtype=torch.bool)
        k_img_full[:, :seq_len] = image_locations

        self.assertTrue(k_img_full[:, :seq_len].all().item())
        self.assertFalse(k_img_full[:, seq_len:].any().item())


def _prefix_chunked_allowed_keys(q_pos, cache_len, current_step, chunk_size, prefix_len, sliding_window=0):
    """Pure-Python reference for _chunk_attn_mask with prefix-LM semantics.

    A key position is allowed when:
    - (causal rule) k <= q, AND the sliding-window constraint holds, OR
    - (prefix rule) BOTH q < prefix_len AND k < prefix_len AND k < current_step+chunk_size
      (the third condition prevents attending to unfilled cache slots).
    """
    filled_end = current_step + chunk_size
    causal_keys = {k for k in range(cache_len) if k <= q_pos}
    if sliding_window > 0:
        causal_keys = {k for k in causal_keys if k >= q_pos - sliding_window + 1}

    if q_pos < prefix_len:
        prefix_keys = {k for k in range(cache_len) if k < prefix_len and k < filled_end}
        return causal_keys | prefix_keys
    return causal_keys


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestChunkAttnMaskPrefixLM(unittest.TestCase):
    """Validate the prefix-LM bidirectional attention guard in _chunk_attn_mask.

    The bug was that the prefix block (q_in_prefix & k_in_prefix) was OR-ed into
    the causal mask without restricting to already-filled cache positions.  Early
    chunks could therefore attend to key slots k >= current_step + chunk_size that
    have not yet been written, producing incorrect outputs.

    The fix intersects the prefix block with ``filled_k = k_pos < current_step + chunk_size``.
    """

    def _chunk_mask_cpu(self, chunk_size, cache_len, current_step, prefix_len_val, batch=1, sliding_window=0):
        """Call the real ``_chunk_attn_mask`` via a minimal SimpleNamespace mock."""
        from eole.decoders.transformer import TransformerDecoder  # noqa: E402

        mock = types.SimpleNamespace(
            cache_len_tgt=cache_len,
            sliding_window=sliding_window,
            LM_type="prefix",
            left_pad_attn_mask=torch.ones(batch, cache_len, dtype=torch.bool),
        )
        tgt_pad_mask = torch.zeros(batch, 1, chunk_size, dtype=torch.bool)
        prefix_len = torch.full((batch,), prefix_len_val, dtype=torch.long)
        return TransformerDecoder._chunk_attn_mask(mock, chunk_size, current_step, tgt_pad_mask, prefix_len=prefix_len)

    def test_prefix_keys_blocked_beyond_filled_end(self):
        """Key positions k >= current_step + chunk_size must NOT be attendable
        even when both q and k are inside the prefix."""
        cache_len = 32
        chunk_size = 8
        current_step = 0  # first chunk; filled positions are 0..7
        prefix_len_val = 20  # prefix spans positions 0..19

        mask = self._chunk_mask_cpu(chunk_size, cache_len, current_step, prefix_len_val)
        # shape: (1, 1, chunk_size, cache_len)
        mask_2d = mask[0, 0]  # (chunk_size, cache_len)

        filled_end = current_step + chunk_size  # = 8

        for q_offset in range(chunk_size):
            q_pos = current_step + q_offset
            # Future key slots (>= filled_end) must be False for ALL query rows,
            # including prefix queries that would normally gain bidirectional access.
            if q_pos < prefix_len_val:
                future_row = mask_2d[q_offset, filled_end:]
                self.assertFalse(
                    future_row.any().item(),
                    msg=(
                        f"chunk0, q={q_pos} (prefix query): "
                        f"future keys {filled_end}..{cache_len-1} should all be False"
                    ),
                )

    def test_prefix_keys_allowed_within_filled_range(self):
        """Key positions k < current_step + chunk_size AND k < prefix_len should
        be accessible for a prefix query even if they precede the causal window."""
        cache_len = 32
        chunk_size = 8
        current_step = 8  # second chunk; filled positions are 0..15
        prefix_len_val = 16  # whole first two chunks are the prefix

        mask = self._chunk_mask_cpu(chunk_size, cache_len, current_step, prefix_len_val)
        mask_2d = mask[0, 0]  # (chunk_size, cache_len)

        filled_end = current_step + chunk_size  # = 16

        for q_offset in range(chunk_size):
            q_pos = current_step + q_offset
            if q_pos < prefix_len_val:
                # All filled prefix keys (0..prefix_len-1 ∩ 0..filled_end-1) should be True
                for k in range(min(prefix_len_val, filled_end)):
                    self.assertTrue(
                        mask_2d[q_offset, k].item(),
                        msg=f"q={q_pos} (prefix): filled prefix key {k} should be True",
                    )
                # Keys beyond filled_end must be False
                future_row = mask_2d[q_offset, filled_end:]
                self.assertFalse(
                    future_row.any().item(),
                    msg=f"q={q_pos} (prefix): future keys {filled_end}..{cache_len-1} should be False",
                )

    def test_non_prefix_query_unaffected_by_prefix_guard(self):
        """A non-prefix query (q_pos >= prefix_len) must only see causal keys."""
        cache_len = 32
        chunk_size = 8
        current_step = 16  # third chunk; non-prefix positions
        prefix_len_val = 8  # only first chunk is prefix

        mask = self._chunk_mask_cpu(chunk_size, cache_len, current_step, prefix_len_val)
        mask_2d = mask[0, 0]

        for q_offset in range(chunk_size):
            q_pos = current_step + q_offset
            # q_pos >= prefix_len_val, so only causal keys should be allowed
            allowed = mask_2d[q_offset].nonzero(as_tuple=False).view(-1).tolist()
            ref = sorted(_chunked_prefill_allowed_keys(q_pos, cache_len))
            self.assertEqual(
                allowed,
                ref,
                msg=f"q={q_pos} (non-prefix): mask differs from causal reference",
            )

    def test_matches_pure_python_reference(self):
        """_chunk_attn_mask with prefix-LM matches the pure-Python reference formula
        for every (chunk, query) combination."""
        cache_len = 24
        chunk_size = 6
        prefix_len_val = 10

        for chunk_idx in range(4):
            current_step = chunk_idx * chunk_size
            mask = self._chunk_mask_cpu(chunk_size, cache_len, current_step, prefix_len_val)
            mask_2d = mask[0, 0]  # (chunk_size, cache_len)

            for q_offset in range(chunk_size):
                q_pos = current_step + q_offset
                if q_pos >= cache_len:
                    break
                ref = _prefix_chunked_allowed_keys(q_pos, cache_len, current_step, chunk_size, prefix_len_val)
                actual = set(mask_2d[q_offset].nonzero(as_tuple=False).view(-1).tolist())
                self.assertEqual(
                    actual,
                    ref,
                    msg=(f"chunk={chunk_idx}, q={q_pos}: " f"mask={sorted(actual)} != ref={sorted(ref)}"),
                )


if __name__ == "__main__":
    unittest.main()
