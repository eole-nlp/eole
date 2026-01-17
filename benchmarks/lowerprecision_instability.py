"""
Compare numerical stability with GQA vs standard MHA
"""

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt


@dataclass
class ModelConfig:
    hidden_size: int = 1024
    dim_per_head: int = 128
    heads: int = 16
    heads_kv: int = 16  # Will be varied
    add_qkvbias: bool = False
    add_final_linear_bias: bool = False
    attn_scaling: Optional[float] = None
    query_norm: bool = False
    key_norm: bool = False
    qk_norm_post_rope: bool = False
    position_encoding_type: str = "absolute"
    n_positions: int = 2048
    sliding_window: int = -1
    relative_positions_buckets: int = 0
    layer_norm: str = "standard"
    norm_eps: float = 1e-5
    head_dim: int = 128

    class RopeConfig:
        rotary_dim: int = 0
        rotary_interleave: bool = True
        xdrope_section: Optional[list] = None

    rope_config: RopeConfig = RopeConfig()


@dataclass
class RunningConfig:
    attention_dropout: list = None
    parallel_gpu: int = 1
    use_ckpting: list = None
    self_attn_backend: str = ""
    dropout: list = None

    def __post_init__(self):
        if self.attention_dropout is None:
            self.attention_dropout = [0.0]
        if self.use_ckpting is None:
            self.use_ckpting = []
        if self.dropout is None:
            self.dropout = [0.0]


class SimplifiedDecoderLayer(nn.Module):
    """Simplified decoder layer for testing"""

    def __init__(self, model_config, running_config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_norm = nn.LayerNorm(model_config.hidden_size, eps=model_config.norm_eps)

        from eole.modules.multi_headed_attn import SelfMHA

        self.self_attn = SelfMHA(model_config, running_config, is_decoder=True)

        # Simple MLP
        self.mlp = nn.Sequential(
            nn.Linear(model_config.hidden_size, model_config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(model_config.hidden_size * 4, model_config.hidden_size),
        )
        self.post_attn_norm = nn.LayerNorm(model_config.hidden_size, eps=model_config.norm_eps)

    def forward(self, x, attn_mask, step):
        # Pre-norm architecture
        normed = self.layer_norm(x)
        attn_out, _ = self.self_attn(
            normed, attn_mask=attn_mask, step=step, return_attn=False, position_embeddings=None
        )
        x = x + attn_out

        # MLP
        normed = self.post_attn_norm(x)
        mlp_out = self.mlp(normed)
        x = x + mlp_out

        return x


def create_causal_mask(seq_len, device, dtype=torch.bool):
    """Create a causal attention mask"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    return mask.unsqueeze(0).unsqueeze(0)


def test_configuration(heads_kv, num_layers=32, dtype=torch.bfloat16):
    """Test a specific KV head configuration"""

    batch_size = 1
    seq_len = 756
    device = "cuda"

    config_name = "MHA (heads=heads_kv=16)" if heads_kv == 16 else f"GQA (heads=16, heads_kv={heads_kv})"
    dtype_name = str(dtype).split(".")[-1]

    print(f"\n{'='*80}")
    print(f"Testing: {config_name} with {dtype_name}")
    print(f"{'='*80}")

    model_config = ModelConfig(heads_kv=heads_kv)

    # Create input
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, seq_len, model_config.hidden_size, device=device, dtype=dtype) * 0.02

    attn_mask = create_causal_mask(seq_len, device)

    # Build non-flash model
    running_config_no_flash = RunningConfig(self_attn_backend="")
    layers_no_flash = (
        nn.ModuleList([SimplifiedDecoderLayer(model_config, running_config_no_flash, i) for i in range(num_layers)])
        .to(device)
        .to(dtype)
    )

    # Initialize weights
    with torch.no_grad():
        for layer in layers_no_flash:
            for name, param in layer.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.zero_()

    # Build flash model with same weights
    running_config_flash = RunningConfig(self_attn_backend="flash")
    layers_flash = (
        nn.ModuleList([SimplifiedDecoderLayer(model_config, running_config_flash, i) for i in range(num_layers)])
        .to(device)
        .to(dtype)
    )

    layers_flash.load_state_dict(layers_no_flash.state_dict())

    # Enable caches
    for layer in layers_no_flash:
        layer.self_attn.kcache = torch.empty(0, device=device)
        layer.self_attn.vcache = torch.empty(0, device=device)

    for layer in layers_flash:
        layer.self_attn.kcache = torch.empty(0, device=device)
        layer.self_attn.vcache = torch.empty(0, device=device)

    # Forward pass
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        with torch.no_grad():
            x_no_flash = input_tensor.clone()
            x_flash = input_tensor.clone()
            layer_diffs = []

            for i, (layer_nf, layer_f) in enumerate(zip(layers_no_flash, layers_flash)):
                x_no_flash = layer_nf(x_no_flash, attn_mask, step=0)
                x_flash = layer_f(x_flash, attn_mask, step=0)

                layer_diff = (x_no_flash - x_flash).abs()

                layer_diffs.append(
                    {
                        "layer": i + 1,
                        "mean_diff": layer_diff.mean().item(),
                        "max_diff": layer_diff.max().item(),
                    }
                )

                if i == 0 or (i + 1) % 8 == 0:
                    print(
                        f"  Layer {i+1:2d}: mean_diff={layer_diff.mean().item():.6e}, "
                        f"max_diff={layer_diff.max().item():.6e}"
                    )

    final_diff = (x_flash - x_no_flash).abs()

    print("\nFinal results:")
    print(f"  Mean absolute difference: {final_diff.mean().item():.6e}")
    print(f"  Max absolute difference:  {final_diff.max().item():.6e}")

    if layer_diffs and layer_diffs[0]["max_diff"] > 0:
        amp = layer_diffs[-1]["max_diff"] / layer_diffs[0]["max_diff"]
        print(f"  Amplification (layer 1 to {num_layers}): {amp:.1f}x")
    else:
        print("  Amplification: N/A (layer 1 diff is zero)")

    return {
        "config": config_name,
        "heads_kv": heads_kv,
        "dtype": dtype_name,
        "layer_diffs": layer_diffs,
        "final_mean_diff": final_diff.mean().item(),
        "final_max_diff": final_diff.max().item(),
    }


def main():
    print("=" * 80)
    print("GQA vs MHA Numerical Stability Comparison")
    print("Comparing BFloat16, Float16, and Float32")
    print("=" * 80)
    print("\nConfiguration:")
    print("  Number of layers: 32")
    print("  Batch size: 1")
    print("  Sequence length: 756")
    print("  Device: cuda")
    print("  Query heads: 16 (fixed)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("\nERROR: CUDA required for this test")
        return

    # Test different KV head configurations
    configs_to_test = [
        16,  # MHA: no KV expansion needed
        8,  # GQA: 2x expansion (your case)
    ]

    dtypes_to_test = [
        (torch.bfloat16, "bfloat16"),
        (torch.float16, "float16"),
        (torch.float32, "float32"),
    ]

    all_results = []

    for dtype, dtype_name in dtypes_to_test:
        print(f"\n{'#'*80}")
        print(f"# Testing with dtype: {dtype_name}")
        print(f"{'#'*80}")

        for heads_kv in configs_to_test:
            result = test_configuration(heads_kv, num_layers=32, dtype=dtype)
            all_results.append(result)

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    print(f"\n{'Configuration':<25} {'Dtype':<10} {'Layer 1 Max':<15} {'Layer 32 Max':<15} {'Amplification':<15}")
    print("-" * 95)

    for result in all_results:
        layer1_max = result["layer_diffs"][0]["max_diff"]
        layer32_max = result["layer_diffs"][-1]["max_diff"]
        amp = layer32_max / layer1_max if layer1_max > 0 else 0

        print(f"{result['config']:<25} {result['dtype']:<10} {layer1_max:<15.6e} {layer32_max:<15.6e} {amp:<15.1f}x")

    # Plot comparison - group by dtype
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Organize results by dtype
        results_by_dtype = {}
        for result in all_results:
            dtype = result["dtype"]
            if dtype not in results_by_dtype:
                results_by_dtype[dtype] = []
            results_by_dtype[dtype].append(result)

        dtype_names = ["bfloat16", "float16", "float32"]
        colors_mha = ["blue", "green", "purple"]
        colors_gqa = ["orange", "red", "brown"]

        # Plot 1: All dtypes, MHA comparison
        ax1 = axes[0, 0]
        for i, (dtype_name, color) in enumerate(zip(dtype_names, colors_mha)):
            if dtype_name in results_by_dtype:
                mha_result = [r for r in results_by_dtype[dtype_name] if r["heads_kv"] == 16][0]
                layers = [d["layer"] for d in mha_result["layer_diffs"]]
                max_diffs = [d["max_diff"] for d in mha_result["layer_diffs"]]
                ax1.semilogy(layers, max_diffs, "-o", color=color, label=f"MHA {dtype_name}", alpha=0.7, markersize=3)

        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Max Absolute Difference (log scale)")
        ax1.set_title("MHA: BFloat16 vs Float16 vs Float32")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: All dtypes, GQA comparison
        ax2 = axes[0, 1]
        for i, (dtype_name, color) in enumerate(zip(dtype_names, colors_gqa)):
            if dtype_name in results_by_dtype:
                gqa_result = [r for r in results_by_dtype[dtype_name] if r["heads_kv"] == 8][0]
                layers = [d["layer"] for d in gqa_result["layer_diffs"]]
                max_diffs = [d["max_diff"] for d in gqa_result["layer_diffs"]]
                ax2.semilogy(
                    layers, max_diffs, "-s", color=color, label=f"GQA(8) {dtype_name}", alpha=0.7, markersize=3
                )

        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Max Absolute Difference (log scale)")
        ax2.set_title("GQA (heads_kv=8): BFloat16 vs Float16 vs Float32")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: BFloat16 - MHA vs GQA
        ax3 = axes[1, 0]
        if "bfloat16" in results_by_dtype:
            for result, color, marker in zip(results_by_dtype["bfloat16"], ["blue", "orange"], ["o", "s"]):
                layers = [d["layer"] for d in result["layer_diffs"]]
                max_diffs = [d["max_diff"] for d in result["layer_diffs"]]
                label = "MHA" if result["heads_kv"] == 16 else "GQA(8)"
                ax3.semilogy(layers, max_diffs, f"-{marker}", color=color, label=label, alpha=0.7, markersize=3)

        ax3.set_xlabel("Layer")
        ax3.set_ylabel("Max Absolute Difference (log scale)")
        ax3.set_title("BFloat16: MHA vs GQA")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Bar chart - Final differences
        ax4 = axes[1, 1]
        x_positions = []
        heights = []
        labels = []
        bar_colors = []

        x = 0
        for dtype_name, base_color in zip(dtype_names, ["blue", "green", "purple"]):
            if dtype_name in results_by_dtype:
                for result in results_by_dtype[dtype_name]:
                    x_positions.append(x)
                    heights.append(result["final_max_diff"])
                    config_type = "MHA" if result["heads_kv"] == 16 else "GQA"
                    labels.append(f"{config_type}\n{dtype_name}")
                    # Lighter shade for MHA, darker for GQA
                    if result["heads_kv"] == 16:
                        bar_colors.append(base_color)
                    else:
                        bar_colors.append("dark" + base_color if base_color != "purple" else "indigo")
                    x += 1
                x += 0.5  # Gap between dtype groups

        bars = ax4.bar(x_positions, heights, color=bar_colors, alpha=0.7)
        ax4.set_xticks(x_positions)
        ax4.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax4.set_ylabel("Final Max Difference")
        ax4.set_title("Final Difference After 32 Layers")
        ax4.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars, heights):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        plt.savefig("dtype_comparison.png", dpi=150, bbox_inches="tight")
        print("\n✓ Plot saved to 'dtype_comparison.png'")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
        import traceback

        traceback.print_exc()

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Group results by dtype for analysis
    results_by_dtype = {}
    for result in all_results:
        dtype = result["dtype"]
        if dtype not in results_by_dtype:
            results_by_dtype[dtype] = {}
        results_by_dtype[dtype][result["heads_kv"]] = result

    for dtype_name in ["bfloat16", "float16", "float32"]:
        if dtype_name in results_by_dtype:
            print(f"\n{dtype_name.upper()}:")
            mha = results_by_dtype[dtype_name].get(16)
            gqa = results_by_dtype[dtype_name].get(8)

            if mha and gqa:
                print(f"  MHA final max diff:  {mha['final_max_diff']:.6e}")
                print(f"  GQA final max diff:  {gqa['final_max_diff']:.6e}")
                if mha["final_max_diff"] > 0:
                    ratio = gqa["final_max_diff"] / mha["final_max_diff"]
                    print(f"  GQA/MHA ratio:       {ratio:.2f}x")
                else:
                    print("  GQA/MHA ratio:       N/A (both are zero)")

    # Compare across dtypes for same config
    print("\n\nCOMPARISON ACROSS DTYPES (MHA):")
    for dtype_name in ["bfloat16", "float16", "float32"]:
        if dtype_name in results_by_dtype and 16 in results_by_dtype[dtype_name]:
            result = results_by_dtype[dtype_name][16]
            print(f"  {dtype_name:10s}: {result['final_max_diff']:.6e}")

    print("\n\nCOMPARISON ACROSS DTYPES (GQA heads_kv=8):")
    for dtype_name in ["bfloat16", "float16", "float32"]:
        if dtype_name in results_by_dtype and 8 in results_by_dtype[dtype_name]:
            result = results_by_dtype[dtype_name][8]
            print(f"  {dtype_name:10s}: {result['final_max_diff']:.6e}")

    # Key findings
    print("\n\nKEY FINDINGS:")

    bf16_mha = results_by_dtype.get("bfloat16", {}).get(16)
    fp32_mha = results_by_dtype.get("float32", {}).get(16)
    fp16_mha = results_by_dtype.get("float16", {}).get(16)

    if bf16_mha and fp32_mha:
        if fp32_mha["final_max_diff"] > 0:
            improvement = bf16_mha["final_max_diff"] / fp32_mha["final_max_diff"]
            print(f"  • FP32 vs BF16 (MHA): {improvement:.1f}x better")
        else:
            print("  • FP32 achieves ZERO divergence!")
            print(f"    BF16 has divergence: {bf16_mha['final_max_diff']:.6e}")

        if fp32_mha["final_max_diff"] < 1e-6:
            print(f"  • FP32 achieves near-zero divergence ({fp32_mha['final_max_diff']:.6e})")
            print("    → Precision is the MAIN issue!")
            print("    → Flash and non-flash implementations are numerically equivalent in FP32")
        elif fp32_mha["final_max_diff"] < 0.01:
            print(f"  • FP32 achieves very low divergence ({fp32_mha['final_max_diff']:.6e})")
            print("    → Precision is the main issue!")
        elif fp32_mha["final_max_diff"] > 0.1:
            print(f"  • FP32 still has significant divergence ({fp32_mha['final_max_diff']:.6e})")
            print("    → Different algorithms are the main issue, not just precision")
        else:
            print(f"  • FP32 has moderate divergence ({fp32_mha['final_max_diff']:.6e})")
            print("    → Both precision and algorithm differences contribute")

    if fp16_mha and bf16_mha:
        if fp16_mha["final_max_diff"] > 0 and bf16_mha["final_max_diff"] > 0:
            fp16_vs_bf16 = bf16_mha["final_max_diff"] / fp16_mha["final_max_diff"]
            print(f"  • BF16 vs FP16 (MHA): {fp16_vs_bf16:.1f}x {'worse' if fp16_vs_bf16 > 1 else 'better'}")
            if fp16_vs_bf16 > 5:
                print("    → BF16 is much less stable than FP16 for this workload")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
