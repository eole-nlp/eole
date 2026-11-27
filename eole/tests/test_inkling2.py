import unittest
from types import SimpleNamespace

import torch

from eole.config.models import TransformerDecoderConfig
from eole.config.run import PredictConfig
from eole.decoders.transformer import TransformerDecoder
from eole.modules.block_float_experts import BlockFloatExperts, BlockFloatExpertWeight
from eole.modules.looped_transformer import SoftcappedLinear


def _expert_weight(tensors, shape):
    nvfp4 = "w_amax" in tensors
    layer = BlockFloatExpertWeight(shape, "nvfp4" if nvfp4 else "mxfp8").to(tensors["w"].device)
    layer.weight = tensors["w"]
    layer.scales = tensors["w_scale"]
    if nvfp4:
        layer.amax = tensors["w_amax"]
        layer.scale_denominator = tensors["w_outer_scale_denominator"]
    layer.prepare()
    return layer


def _tiny_decoder(device="cpu", dtype=torch.float32, *, packed=False):
    cfg = TransformerDecoderConfig(
        hidden_size=32,
        layers=4,
        heads=2,
        heads_kv=1,
        head_dim=16,
        sliding_window=3,
        layer_norm="rms",
        norm_eps=1e-6,
        num_experts=3,
        num_experts_per_tok=2,
        moe_transformer_ff=16,
        layer_types=["sliding_attention", "full_attention"] * 2,
        layer_loop=dict(
            core_start=1,
            core_end=3,
            passes=3,
            gate_rank=8,
            residual_block_size=1,
            convolution_width=3,
            relative_dim=4,
            relative_extent=3,
            query_clip=4,
            relative_clip=4,
            log_scale_floor=4,
            log_scale_alpha=0.1,
            expert_formats=["nvfp4"] * 4,
            route_scale=1,
            gate_softcap=4,
            up_softcap=25,
            logit_softcap=30,
            output_vocab_size=21,
        ),
    )
    running = PredictConfig(
        model_path=["dummy"], src="dummy", beam_size=1, batch_size=1, context_length=12, prefill_chunk_size=2
    )
    decoder = TransformerDecoder(cfg, running).to(device).eval()
    generator = SoftcappedLinear(32, 24, 21, 30, device=device).to(dtype)
    rng = torch.Generator(device=device).manual_seed(7)

    for parameter in decoder.parameters():
        parameter.data = (torch.randn(parameter.shape, device=device, generator=rng) * 0.2).to(dtype)
    for buffer in decoder.buffers():
        if buffer.dtype == torch.uint8:
            buffer.random_(0, 256, generator=rng)
        elif buffer.dtype == torch.float8_e4m3fn:
            buffer.copy_(torch.full(buffer.shape, 0.125, device=device))
        else:
            buffer.fill_(0.9)
    decoder._decode_graph_enabled = packed
    decoder.prepare_looped_inference(running)
    hidden = torch.randn((1, 10, 32), device=device, dtype=dtype, generator=rng)
    return decoder, generator, hidden


def _decode(decoder, hidden, *, step=0):
    pad = torch.zeros((1, 1, hidden.size(1)), device=hidden.device, dtype=torch.bool)
    return decoder(hidden.contiguous(), step=step, tgt_pad_mask=pad)[0]


class TestLoopedDecoder(unittest.TestCase):
    def test_expert_dequantization(self):
        fp4 = {
            "w": torch.tensor([[[0x21] * 8], [[0x97] * 8]], dtype=torch.uint8),
            "w_scale": torch.tensor([[[2]], [[1]]], dtype=torch.float8_e4m3fn),
            "w_amax": torch.tensor([1.0, 4.0]),
            "w_outer_scale_denominator": torch.tensor([2.0]),
        }
        layer = _expert_weight(fp4, (2, 1, 16))
        weights = layer(torch.tensor([1, 0]), torch.bfloat16)
        expected = torch.tensor([[[12.0, -1] * 8], [[0.5, 1.0] * 8]], dtype=torch.bfloat16)
        torch.testing.assert_close(weights, expected)

        fp8 = {
            "w": torch.full((2, 1, 32), -2, dtype=torch.float32).to(torch.float8_e4m3fn),
            "w_scale": torch.tensor([[[126]], [[128]]], dtype=torch.uint8),
        }
        layer = _expert_weight(fp8, (2, 1, 32))
        torch.testing.assert_close(layer(0, torch.float32), torch.full((1, 32), -1.0))
        torch.testing.assert_close(layer(1, torch.float32), torch.full((1, 32), -4.0))

    @torch.inference_mode()
    def test_chunked_prefill_matches_full_forward_across_window(self):
        decoder, generator, hidden = _tiny_decoder()
        hidden = hidden[:, :7].contiguous()
        expected = generator(_decode(decoder, hidden))
        pad = torch.zeros((1, 1, hidden.size(1)), dtype=torch.bool)
        decoder._init_cache(hidden, pad)
        actual = generator(_decode(decoder, hidden))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        assert decoder.cache_seqlens.item() == hidden.size(1)
        assert [state.key.size(-2) for state in decoder._visit_caches] == [2, 7] * 4
        assert len({state.conv["k"].data_ptr() for state in decoder._visit_caches}) == 8
        assert torch.isneginf(actual[..., 21:]).all()

    @torch.inference_mode()
    def test_verify_commit_matches_direct_decode_across_window(self):
        decoder, _, hidden = _tiny_decoder()
        prefix, proposals = hidden[:, :6].contiguous(), hidden[:, 6:9].contiguous()
        for accepted in (0, 1, 2, 3):
            with self.subTest(committed_inputs=accepted):
                decoder._init_cache(prefix, torch.zeros((1, 1, 6), dtype=torch.bool))
                _decode(decoder, prefix)
                decoder.begin_loop_verification()
                _decode(decoder, proposals, step=6)
                decoder.finish_loop_verification(accepted)
                actual = _decode(decoder, hidden[:, 9:], step=6 + accepted)
                decoder._init_cache(prefix, torch.zeros((1, 1, 6), dtype=torch.bool))
                _decode(decoder, prefix)
                for i in range(accepted):
                    _decode(decoder, proposals[:, i : i + 1], step=6 + i)
                expected = _decode(decoder, hidden[:, 9:], step=6 + accepted)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_vision_preprocessing_pads_rectangular_images(self):
        from PIL import Image

        from eole.inputters.image_utils import process_image

        image = Image.new("RGB", (41, 21), (255, 0, 0))
        result = process_image(image, adapter="inkling2", image_patch_size=40)
        self.assertEqual(result["tokens"], "<|unused_200054|>" * 2)
        pixels = torch.from_numpy(result["image"])
        self.assertEqual(tuple(pixels.shape), (3, 40, 80))
        torch.testing.assert_close(pixels[:, 20, 40], torch.tensor([1.0, -1.0, -1.0]))
        torch.testing.assert_close(pixels[:, 21, 40], torch.full((3,), -1.0 - 1 / 127.5))


@unittest.skipUnless(
    torch.version.cuda is not None and torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9),
    "requires NVIDIA SM89 or newer",
)
class TestLoopedDecoderCUDA(unittest.TestCase):
    @torch.inference_mode()
    def test_cuda_graph_matches_eager_across_ring_wrap(self):
        for prefix_length in (2, 5):
            with self.subTest(prefix_length=prefix_length):
                decoder, generator, hidden = _tiny_decoder("cuda", torch.bfloat16, packed=True)
                pad = torch.zeros((1, 1, prefix_length), device="cuda", dtype=torch.bool)
                prefix = hidden[:, :prefix_length].contiguous()
                decoder._init_cache(prefix, pad)
                _decode(decoder, prefix)
                expected = [
                    generator(_decode(decoder, token, step=i))[..., :21]
                    for i, token in enumerate(hidden[:, prefix_length:].split(1, dim=1), prefix_length)
                ]

                for _ in range(2):
                    decoder._disable_cache()
                    decoder._init_cache(prefix, pad)
                    decoder._loop_cache_capacity = hidden.size(1)
                    decoder._decode_graph_request_allowed = True
                    _decode(decoder, prefix)
                    for i, token in enumerate(hidden[:, prefix_length:].split(1, dim=1), prefix_length):
                        actual = generator(_decode(decoder, token, step=i))[..., :21]
                        assert decoder.cache_seqlens.item() == i + 1
                        divergence = torch.nn.functional.kl_div(
                            actual.log_softmax(-1),
                            expected[i - prefix_length].log_softmax(-1),
                            reduction="sum",
                            log_target=True,
                        )
                        assert divergence.item() < 5e-4
                    assert decoder._decode_graph.graph is not None

    @torch.inference_mode()
    def test_cuda_expert_gemv_matches_dequantized_reference(self):
        for nvfp4 in (True, False):
            with self.subTest(nvfp4=nvfp4):
                from eole.triton.block_float_moe import expert_gemv

                if nvfp4:
                    tensors = {
                        "w": torch.randint(0, 256, (3, 7, 48), dtype=torch.uint8, device="cuda"),
                        "w_scale": torch.rand(3, 7, 6, device="cuda").to(torch.float8_e4m3fn),
                        "w_amax": torch.tensor([1.0, 2.0, 3.0], device="cuda"),
                        "w_outer_scale_denominator": torch.tensor([2.0], device="cuda"),
                    }
                else:
                    tensors = {
                        "w": torch.randn(3, 7, 96, device="cuda").to(torch.float8_e4m3fn),
                        "w_scale": torch.randint(125, 129, (3, 7, 3), dtype=torch.uint8, device="cuda"),
                    }
                layer = _expert_weight(tensors, (3, 7, 96))
                selected = torch.tensor([2, 0], device="cuda")
                vectors = torch.randn(2, 96, device="cuda", dtype=torch.bfloat16)
                for shared_input in (False, True):
                    x = vectors[:1] if shared_input else vectors
                    expected = (layer(selected, x.dtype) @ x.unsqueeze(-1)).squeeze(-1)
                    actual = expert_gemv(layer, x, selected, shared_input=shared_input)
                    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.008)

    @torch.inference_mode()
    def test_grouped_experts_match_reference(self):
        from eole.ops import _CPP_OPS_AVAILABLE

        if not _CPP_OPS_AVAILABLE:
            self.skipTest("requires Eole MoE routing extension")
        config = SimpleNamespace(
            num_experts=5,
            hidden_size=96,
            moe_transformer_ff=96,
            layer_loop=SimpleNamespace(gate_softcap=4, up_softcap=25),
        )
        rng = torch.Generator(device="cuda").manual_seed(19)
        for weight_format in ("nvfp4", "mxfp8"):
            with self.subTest(weight_format=weight_format):
                experts = BlockFloatExperts(config, weight_format).cuda()
                for layer in (experts.gate_up, experts.down):
                    if layer.nvfp4:
                        layer.weight.random_(0, 256, generator=rng)
                        layer.scales.copy_(torch.rand(layer.scales.shape, device="cuda", generator=rng))
                        layer.amax.copy_(torch.arange(1, 6, device="cuda"))
                        layer.scale_denominator.fill_(16)
                    else:
                        layer.weight.copy_(torch.randn(layer.weight.shape, device="cuda", generator=rng))
                        layer.scales.random_(122, 125, generator=rng)
                experts.prepare()
                self.assertTrue(experts._grouped_prefill)
                for size in (7, 97):
                    with self.subTest(tokens=size):
                        inputs = torch.randn((size, 96), device="cuda", dtype=torch.bfloat16, generator=rng)
                        rows = torch.rand((size, 4), device="cuda", generator=rng)
                        weights, indices = rows.topk(2)
                        weights = weights / weights.sum(dim=-1, keepdim=True)
                        # Use FP32 accumulation so the reference does not depend on cuBLAS's BF16
                        # reduced-precision reductions. Round each projection back to BF16.
                        expected = torch.zeros_like(inputs, dtype=torch.float32)
                        for expert in range(4):
                            token, route = torch.where(indices == expert)
                            gate_up = torch.nn.functional.linear(
                                inputs[token].float(), experts.gate_up(expert, inputs.dtype).float()
                            ).to(inputs.dtype)
                            down = torch.nn.functional.linear(
                                experts._activate(gate_up).float(), experts.down(expert, inputs.dtype).float()
                            ).to(inputs.dtype)
                            expected.index_add_(0, token, down.float() * weights[token, route, None])
                        actual = experts(inputs, weights, indices).float()
                        expected = expected.to(inputs.dtype).float()
                        relative_error = (actual - expected).norm() / expected.norm()
                        self.assertLess(relative_error.item(), 0.002)

    @torch.inference_mode()
    def test_fused_attention_probabilities(self):
        for window, position in ((None, 6), (7, 3), (7, 11)):
            with self.subTest(window=window, position=position):
                from eole.triton.looped_decoder import attention_probabilities

                device, dtype = "cuda", torch.bfloat16
                rng = torch.Generator(device=device).manual_seed(1)
                dots = torch.randn(1, 4, 1, 7, device=device, dtype=dtype, generator=rng)
                bias = torch.randn(1, 4, 1, 5, device=device, dtype=dtype, generator=rng)
                sink = torch.randn(1, 1, 4, device=device, dtype=dtype, generator=rng)
                positions = torch.tensor([position], device=device)
                distances = positions[:, None] - torch.arange(7, device=device)
                if window:
                    distances = distances.remainder(7)
                relative = bias.gather(-1, distances.clamp(0, 4).expand_as(dots))
                relative = relative.masked_fill((distances < 0) | (distances >= 5), 0)
                scores = (dots * 0.25 + relative).masked_fill((distances < 0) | (distances > position), -torch.inf)
                expected = torch.cat((scores, sink.transpose(1, 2).unsqueeze(-1)), dim=-1).float().softmax(-1)[..., :-1]
                actual = attention_probabilities(dots, bias, sink, positions, window, 0.25)
                torch.testing.assert_close(actual, expected.to(dtype), rtol=0.008, atol=0)
