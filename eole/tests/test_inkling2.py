import pytest
import torch

from eole.models.inkling2 import Inkling2Mini, Inkling2PackedExperts
from eole.models.inkling2_decode import Inkling2Decode


def test_inkling2_expert_dequantization():
    fp4 = {
        "w": torch.tensor([[[0x21] * 8], [[0x97] * 8]], dtype=torch.uint8),
        "w_scale": torch.tensor([[[2]], [[1]]], dtype=torch.float8_e4m3fn),
        "w_amax": torch.tensor([1.0, 4.0]),
        "w_outer_scale_denominator": torch.tensor([2.0]),
    }
    layer = Inkling2PackedExperts(fp4.__getitem__, "w", (2, 1, 16))
    weights = layer(torch.tensor([1, 0]), torch.bfloat16)
    expected = torch.tensor([[[12.0, -1] * 8], [[0.5, 1.0] * 8]], dtype=torch.bfloat16)
    torch.testing.assert_close(weights, expected)

    fp8 = {
        "w": torch.full((2, 1, 32), -2, dtype=torch.float32).to(torch.float8_e4m3fn),
        "w_scale": torch.tensor([[[126]], [[128]]], dtype=torch.uint8),
    }
    layer = Inkling2PackedExperts(fp8.__getitem__, "w", (2, 1, 32))
    torch.testing.assert_close(layer(0, torch.float32), torch.full((1, 32), -1.0))
    torch.testing.assert_close(layer(1, torch.float32), torch.full((1, 32), -4.0))


def _tiny_model(device="cpu", dtype=torch.float32):
    cfg = {
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "unpadded_vocab_size": 21,
        "n_layers_per_attn_res_block": 1,
        "rms_norm_eps": 1e-6,
        "final_logit_softcapping": 30,
        "local_layer_ids": [0, 2],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "swa_num_attention_heads": 2,
        "swa_num_key_value_heads": 1,
        "swa_head_dim": 16,
        "sliding_window_size": 3,
        "qk_l2_clip_threshold": 4,
        "rel_l2_clip_threshold": 4,
        "log_scaling_n_floor": 4,
        "log_scaling_alpha": 0.1,
        "intermediate_size": 16,
        "n_routed_experts": 3,
        "num_experts_per_tok": 2,
        "n_shared_experts": 0,
        "route_scale": 1.0,
        "mlp_gate_softcap": 4,
        "mlp_up_softcap": 25,
        "q_bias": False,
        "o_bias": False,
        "use_gate_bias": True,
        "use_value_norm_no_gamma": True,
        "attn_res_normalize_values": True,
        "loop_config": {"core_start": 1, "core_end": 3, "num_passes": 3, "is_varloop": False},
    }
    rng = torch.Generator().manual_seed(7)

    def rand(*shape):
        return torch.randn(*shape, generator=rng) * 0.2

    weights = {
        "embed.weight": rand(21, 32),
        "embed_norm.weight": torch.ones(32),
        "norm.weight": torch.ones(32),
        "unembed.weight": rand(21, 32),
        "attn_res.output_query": rand(32),
        "middle_cycle_cross_pass_queries": rand(4, 32),
        "middle_cycle_cross_pass_kv_queries": rand(4, 32),
    }
    for i in range(cfg["num_hidden_layers"]):
        sliding = i in cfg["local_layer_ids"]
        if i:
            weights[f"attn_res.attn_queries.{i - 1}"] = rand(1 if sliding else 3, 32)
        weights[f"attn_res.mlp_queries.{i}"] = rand(32)
        shapes = {
            "attn.wq_du.weight": (32, 32),
            "attn.wk_dv.weight": (16, 32),
            "attn.wv_dv.weight": (16, 32),
            "attn.wr_du.weight": (8, 32),
            "attn.wo_ud.weight": (32, 32),
            "attn.gate_attn.weight": (32, 32),
            "attn.attn_sink_proj.weight": (2, 32),
            "attn.q_l2_clip.scale": (2, 16),
            "attn.rel_l2_clip.scale": (2, 4),
            "attn.rel_logits_proj.proj": (4, 3),
            "attn_sconv.weight": (32, 1, 3),
            "mlp_sconv.weight": (32, 1, 3),
            "mlp.gate.weight": (3, 32),
            "mlp.gate.bias": (3,),
        }
        if sliding:
            shapes["attn_norm.weight"] = (32,)
        else:
            for component in "qkv":
                shapes[f"{component}_input_norm.weight"] = (32,)
        for name in ("k", "v"):
            if sliding:
                shapes[f"attn.{name}_sconv.weight"] = (16, 1, 3)
            else:
                shapes[f"attn.{name}_sconv.static_w"] = (3, 16)
                shapes[f"attn.{name}_sconv.weight_proj"] = (3, 32)
        if 1 <= i < 3:
            for name in ("attn", "k", "v", "mlp"):
                shapes[f"controllers.{name}.down_weight"] = (8, 64)
                shapes[f"controllers.{name}.gain_weight"] = (32, 8)
                shapes[f"controllers.{name}.forget_weight"] = (32, 8)
        prefix = f"layers.{i}."
        weights.update({prefix + name: rand(*shape) for name, shape in shapes.items()})
        weights[prefix + "mlp_norm.weight"] = torch.ones(32)
        for name, out_dim, in_dim in (("w13_weight", 32, 32), ("w2_weight", 32, 16)):
            key = prefix + f"mlp.experts.{name}"
            weights[key] = torch.randint(0, 256, (3, out_dim, in_dim // 2), generator=rng, dtype=torch.uint8)
            weights[key + "_scale"] = torch.full((3, out_dim, in_dim // 16), 0.125).to(torch.float8_e4m3fn)
            weights[key + "_amax"] = torch.ones(3)
            weights[key + "_outer_scale_denominator"] = torch.ones(1)

    def read(name):
        value = weights[name]
        return value.to(device=device, dtype=dtype if value.dtype == torch.float32 else value.dtype)

    return Inkling2Mini(read, cfg).eval()


@torch.inference_mode()
def test_inkling2_cached_logits_match_full_forward_across_window():
    model = _tiny_model()
    tokens = torch.tensor([[2, 4, 5, 0, 7, 4, 14]])
    expected = model(tokens, all_logits=True)
    cache = model.new_cache()
    chunks = [model(chunk, cache, all_logits=True) for chunk in tokens.split((2, 1, 3, 1), dim=1)]
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, atol=5e-6, rtol=5e-5)
    assert cache.length == tokens.size(1)
    assert [layer.key.size(-2) for layer in cache.layers] == [2, 7, 2, 7, 2, 7, 2, 7]
    assert len({state.conv["k"].data_ptr() for state in cache.layers}) == model.num_visits


@pytest.mark.parametrize("prefix_length", [2, 5])
@pytest.mark.parametrize("capture", [False, True])
@torch.inference_mode()
def test_inkling2_fixed_decode_matches_eager_across_ring_wrap(prefix_length, capture):
    if capture and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = "cuda" if capture else "cpu"
    model = _tiny_model().to(device)
    tokens = torch.tensor([[2, 4, 5, 0, 7, 4, 14, 8, 9, 1]], device=device)
    eager = model.new_cache()
    model(tokens[:, :prefix_length], eager)
    fixed = Inkling2Decode(model, eager, tokens.size(1), capture=capture, packed_experts=False)
    for token in tokens[:, prefix_length:].split(1, dim=1):
        expected = model(token, eager)
        torch.testing.assert_close(fixed(token), expected, atol=5e-6, rtol=5e-5)
        assert fixed.length == fixed.position.item() == eager.length
    if capture:
        assert fixed.graph.graph is not None
    with pytest.raises(ValueError, match="exceeded"):
        fixed(tokens[:, -1:])


@pytest.mark.parametrize("nvfp4", [True, False])
@torch.inference_mode()
def test_inkling2_cuda_expert_gemv_matches_dequantized_reference(nvfp4):
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("requires NVIDIA SM89 or newer")
    from eole.triton.inkling2_moe import expert_gemv

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
    layer = Inkling2PackedExperts(tensors.__getitem__, "w", (3, 7, 96))
    selected = torch.tensor([2, 0], device="cuda")
    vectors = torch.randn(2, 96, device="cuda", dtype=torch.bfloat16)
    for shared_input in (False, True):
        x = vectors[:1] if shared_input else vectors
        expected = (layer(selected, x.dtype) @ x.unsqueeze(-1)).squeeze(-1)
        actual = expert_gemv(layer, x, selected, shared_input=shared_input)
        torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.008)


@torch.inference_mode()
def test_inkling2_fused_decode_matches_eager_across_ring_wrap():
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("requires NVIDIA SM89 or newer")
    model = _tiny_model("cuda", torch.bfloat16)
    tokens = torch.tensor([[2, 4, 5, 0, 7, 4, 14, 8, 9, 1]], device="cuda")
    eager = model.new_cache()
    model(tokens[:, :2], eager)
    fixed = Inkling2Decode(model, eager, tokens.size(1), capture=False)
    graph = Inkling2Decode(model, eager, tokens.size(1))
    for token in tokens[:, 2:].split(1, dim=1):
        expected = model(token, eager, packed_experts=True)
        actual = graph(token)
        torch.testing.assert_close(actual, fixed(token), atol=0, rtol=0)
        torch.testing.assert_close(actual.argmax(-1), expected.argmax(-1))
        # Fused reductions can shift BF16 rounding while retaining the distribution.
        divergence = torch.nn.functional.kl_div(
            actual.log_softmax(-1), expected.log_softmax(-1), reduction="sum", log_target=True
        )
        assert divergence.item() < 5e-4
        assert graph.position.item() == eager.length
    assert graph.graph.graph is not None


@pytest.mark.parametrize(("window", "position"), [(None, 6), (7, 3), (7, 11)])
@torch.inference_mode()
def test_inkling2_fused_attention_probabilities(window, position):
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("requires NVIDIA SM89 or newer")
    from eole.triton.inkling2_decode import attention_probabilities

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
