"""Request cache ownership and independently positioned rotary rows."""

import unittest
from unittest.mock import patch

import torch

from eole.config.inference import InferenceConfig
from eole.config.models import TransformerDecoderConfig
from eole.decoders.transformer import TransformerDecoder
from eole.modules.attention.rope import apply_rotary_emb
from eole.utils.request_cache import RequestCache


class RequestCacheTests(unittest.TestCase):
    def decoder(self):
        return TransformerDecoder(
            TransformerDecoderConfig(
                layers=2,
                hidden_size=64,
                heads=4,
                heads_kv=2,
                transformer_ff=128,
                with_cross_attn=False,
                max_position_embeddings=16,
                layer_types=["linear_attention", "full_attention"],
                linear_conv_kernel_dim=3,
                linear_key_head_dim=16,
                linear_value_head_dim=16,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
            ),
            running_config=InferenceConfig(self_attn_backend="pytorch"),
        ).eval()

    def allocate(self, decoder, batch):
        decoder._disable_cache()  # detach previous ownership before fresh allocation
        decoder._init_cache(torch.zeros(batch, 2, 64), torch.zeros(batch, 1, 2, dtype=torch.bool))
        return RequestCache.take(decoder)

    def test_rebind_restores_kv_gdn_and_positions_without_copy(self):
        decoder = self.decoder()
        first = self.allocate(decoder, 1)
        first.metadata["cache_seqlens"].fill_(7)
        first.layers[0][1]["recurrent_state"].fill_(42)
        second = self.allocate(decoder, 1)
        second.metadata["cache_seqlens"].fill_(3)
        first.bind()
        self.assertIs(decoder.cache_seqlens, first.metadata["cache_seqlens"])
        self.assertEqual(decoder.cache_seqlens.item(), 7)
        self.assertEqual(decoder.transformer_layers[0].linear_attn.recurrent_state.flatten()[0], 42)
        second.bind()
        self.assertEqual(decoder.cache_seqlens.item(), 3)
        self.assertEqual(decoder.transformer_layers[0].linear_attn.recurrent_state.count_nonzero(), 0)

    def test_slot_reuse_replaces_all_states_and_leaves_neighbour_intact(self):
        decoder = self.decoder()
        bank = self.allocate(decoder, 2)
        bank.metadata["cache_seqlens"].fill_(12)
        for _, state in bank.layers:
            for value in state.values():
                value.fill_(99)
        source = self.allocate(decoder, 1)
        source.metadata["cache_seqlens"].fill_(2)
        bank.copy_row(0, source)
        self.assertEqual(bank.metadata["cache_seqlens"].tolist(), [2, 12])
        for (_, target), (_, original) in zip(bank.layers, source.layers):
            for name, value in target.items():
                torch.testing.assert_close(value[:1], original[name])
                self.assertTrue(value[1].eq(99).all())

    def test_singleton_view_updates_only_its_owned_bank_row(self):
        decoder = self.decoder()
        bank = self.allocate(decoder, 3)
        view = bank.row(2)
        view.bind()
        decoder.cache_seqlens.add_(4)
        decoder.transformer_layers[0].linear_attn.recurrent_state.fill_(7)
        self.assertEqual(bank.metadata["cache_seqlens"].tolist(), [0, 0, 4])
        state = bank.layers[0][1]["recurrent_state"]
        self.assertEqual(state[:2].count_nonzero(), 0)
        self.assertTrue(state[2].eq(7).all())

    def test_incompatible_admission_does_not_partially_mutate(self):
        decoder = self.decoder()
        bank = self.allocate(decoder, 2)
        source = self.allocate(decoder, 1)
        source.metadata["cache_seqlens"].fill_(4)
        source.layers[-1][1]["vcache"] = torch.zeros(1, 8, 2, 16)
        with self.assertRaisesRegex(ValueError, "layouts"):
            bank.copy_row(0, source)
        self.assertEqual(bank.metadata["cache_seqlens"].count_nonzero(), 0)

    def test_rotary_mixed_positions_equal_separate_rows(self):
        torch.manual_seed(51)
        for interleave in (True, False):
            q, k = torch.randn(3, 1, 4, 16), torch.randn(3, 1, 2, 16)
            theta = torch.randn(16, 4)
            table = torch.cat((theta.cos(), theta.sin()), -1)
            selected = table[torch.tensor([[2], [7], [11]])]
            with patch("eole.modules.attention.rope._CPP_OPS_AVAILABLE", False):
                actual = apply_rotary_emb(q.clone(), k.clone(), selected, interleave)
                separate = [
                    apply_rotary_emb(q[i : i + 1].clone(), k[i : i + 1].clone(), selected[i], interleave)
                    for i in range(3)
                ]
            for index in range(2):
                torch.testing.assert_close(actual[index], torch.cat([r[index] for r in separate]))

    def test_broadcast_uniform_rotary_tables_match_legacy_and_compiled_pytorch(self):
        torch.manual_seed(73)
        for interleave in (False, True):
            q, k = torch.randn(3, 4, 4, 16), torch.randn(3, 4, 2, 16)
            theta = torch.randn(4, 4)
            table = torch.cat((theta.cos(), theta.sin()), -1)
            rows = table[None].expand(3, -1, -1).contiguous()
            with patch("eole.modules.attention.rope._CPP_OPS_AVAILABLE", False):
                old = apply_rotary_emb(q.clone(), k.clone(), table, interleave)
                new = apply_rotary_emb(q.clone(), k.clone(), rows, interleave)
                function = torch.compile(
                    lambda a, b, c: apply_rotary_emb(a, b, c, interleave), backend="eager", fullgraph=True
                )
                compiled = function(q.clone(), k.clone(), rows)
            for index in range(2):
                torch.testing.assert_close(new[index], old[index], rtol=0, atol=0)
                torch.testing.assert_close(compiled[index], old[index], rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RequestRotaryCudaTests(unittest.TestCase):
    def test_cuda_mixed_rows_match_independent_positions(self):
        from eole.modules.attention.rope import _CPP_OPS_AVAILABLE

        if not _CPP_OPS_AVAILABLE:
            self.skipTest("Eole extension unavailable")
        torch.manual_seed(19)
        for dtype in (torch.float16, torch.bfloat16):
            for interleave in (False, True):
                q = torch.randn(3, 1, 4, 64, device="cuda", dtype=dtype)
                k = torch.randn(3, 1, 2, 64, device="cuda", dtype=dtype)
                theta = torch.randn(16, 16, device="cuda")
                table = torch.cat((theta.cos(), theta.sin()), -1)
                selected = table[torch.tensor([[2], [7], [11]], device="cuda")].to(dtype)
                actual = apply_rotary_emb(q.clone(), k.clone(), selected, interleave)
                separate = [
                    apply_rotary_emb(q[i : i + 1].clone(), k[i : i + 1].clone(), selected[i], interleave)
                    for i in range(3)
                ]
                for index in range(2):
                    torch.testing.assert_close(actual[index], torch.cat([r[index] for r in separate]), rtol=0, atol=0)

    def test_uniform_cuda_tables_match_legacy_for_multiple_rows_and_tokens(self):
        from eole.modules.attention.rope import _CPP_OPS_AVAILABLE

        if not _CPP_OPS_AVAILABLE:
            self.skipTest("Eole extension unavailable")
        torch.manual_seed(75)
        for dtype in (torch.float16, torch.bfloat16):
            for interleave in (False, True):
                q = torch.randn(3, 4, 4, 64, device="cuda", dtype=dtype)
                k = torch.randn(3, 4, 2, 64, device="cuda", dtype=dtype)
                theta = torch.randn(4, 16, device="cuda")
                table = torch.cat((theta.cos(), theta.sin()), -1).to(dtype)
                rows = table[None].expand(3, -1, -1).contiguous()
                old = apply_rotary_emb(q.clone(), k.clone(), table, interleave)
                new = apply_rotary_emb(q.clone(), k.clone(), rows, interleave)
                for index in range(2):
                    torch.testing.assert_close(new[index], old[index], rtol=0, atol=0)
