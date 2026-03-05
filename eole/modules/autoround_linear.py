import gc
import threading
import torch.nn as nn
from torch.cuda import is_available as cuda_is_available

_marlin_preflight_done = False


def _preflight_marlin_import():
    """Import gptqmodel marlin and retroactively patch any pre-existing triton.Autotuner instances.

    gptqmodel's nogil_patcher replaces triton.Autotuner.__init__ and .run() with thread-safe
    versions.  The patched __init__ adds three instance attributes that the patched run()
    requires:

      _cache       – a dict alias for the existing ``cache`` dict (for lock-protected access)
      _cache_lock  – a threading.RLock() for cache serialisation across threads
      _cache_futures – a dict for in-flight autotuning work items

    Instances created before gptqmodel is imported (e.g. FLA's @triton.autotune kernels in
    fla/modules/convolution.py, or eole's own fused_moe.py kernels) are missing all three
    attributes.  When gptqmodel's background thread later calls the patched run(), it crashes
    on the first missing attribute.

    This function:
    1. Imports gptqmodel marlin (side effect: replaces Autotuner.__init__ and .run()).
    2. Walks all live Python objects via gc and fully back-fills the three attributes on any
       Autotuner instance that was created before the patch.

    Because the patching is retroactive and order-independent it works regardless of whether
    triton / FLA / eole's own triton kernels were imported before this function runs.

    A module-level flag ensures the gc scan executes at most once per process.

    Safe to call unconditionally: silently returns if gptqmodel is not installed or CUDA
    is unavailable.
    """
    global _marlin_preflight_done
    if _marlin_preflight_done:
        return
    if not cuda_is_available():
        return
    try:
        # side effect: patches triton.Autotuner
        from auto_round_extension.cuda.gptqmodel_marlin import get_marlin_layer  # noqa E401
    except ImportError:
        return

    # Retroactively apply the same attribute additions that gptqmodel's patched_init performs,
    # to every Autotuner instance that was created before the patch ran.
    try:
        from triton.runtime.autotuner import Autotuner

        for obj in gc.get_objects():
            if isinstance(obj, Autotuner) and not hasattr(obj, "_cache_lock"):
                # Mirror what gptqmodel's patched_init does for pre-existing instances:
                cache_map = getattr(obj, "cache", {})
                obj._cache = dict(cache_map)
                obj.cache = obj._cache
                obj._cache_lock = threading.RLock()
                obj._cache_futures = {}
    except (ImportError, AttributeError):
        pass

    _marlin_preflight_done = True


def replace_autoround_linear(
    model,
    module_to_convert=[],
    w_bit=4,
    group_size=128,
    packing_format="auto_round:auto_gptq",
    sym=True,
    module_to_not_convert=[],
):
    """Replace nn.Linear layers with AutoRound QuantLinear for quantized inference.

    The packing_format determines how qzeros are stored:
    - 'gptq' in packing_format: qzeros stored as (zero_point - 1) per GPTQ convention.
    - Otherwise: qzeros stored directly (direct zero-point).

    Backend preference order:
    1. Marlin CUDA kernels (requires CUDA + gptqmodel, sym=True only) — fastest
    2. Triton GPU kernels (requires CUDA + triton) — fast GPU kernels
    3. PyTorch fallback — works everywhere

    For Marlin layers, call post_init_autoround_linear(model) after loading the weights
    to repack them into the Marlin-optimized layout.

    Marlin validates layer dimensions at construction time and raises NotImplementedError
    for unsupported shapes (e.g. out_features not divisible by 64).  Such layers are
    replaced with the pure-PyTorch backend, which has no CUDA kernel shape requirements.

    module_to_not_convert: list of module names (direct children) whose entire subtree
        should be skipped.  Use this for parent modules that were kept in fp16 during
        quantization (e.g. ``shared_experts`` in MoE models).
    """
    use_gptq_zp = "gptq" in packing_format
    QuantLinear, use_marlin = _get_autoround_quant_linear_cls(use_gptq_zp, sym)

    # Lazily computed fallback class for layers that Marlin rejects.
    fallback_cls = None

    def _get_fallback():
        nonlocal fallback_cls
        if fallback_cls is None:
            fallback_cls, _ = _get_autoround_quant_linear_cls(use_gptq_zp, force_pytorch=True)
        return fallback_cls

    for name, module in model.named_children():
        if name in module_to_not_convert:
            continue  # skip entire subtree — this parent was kept in fp16
        if len(list(module.children())) > 0:
            replace_autoround_linear(
                module, module_to_convert, w_bit, group_size, packing_format, sym, module_to_not_convert
            )

        if isinstance(module, nn.Linear) and name in module_to_convert:
            if use_marlin:
                try:
                    new_module = QuantLinear(
                        bits=w_bit,
                        group_size=group_size,
                        desc_act=False,
                        sym=sym,
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                    )
                except NotImplementedError:
                    # Marlin rejected this layer's dimensions — use PyTorch fallback.
                    new_module = _get_fallback()(
                        bits=w_bit,
                        group_size=group_size,
                        infeatures=module.in_features,
                        outfeatures=module.out_features,
                        bias=module.bias is not None,
                    )
            else:
                new_module = QuantLinear(
                    bits=w_bit,
                    group_size=group_size,
                    infeatures=module.in_features,
                    outfeatures=module.out_features,
                    bias=module.bias is not None,
                )
            model._modules[name] = new_module
    return model


def post_init_autoround_linear(model):
    """Call post_init() on all AutoRound QuantLinear modules in the model.

    Required for Marlin layers: repacks GPTQ-format weights into Marlin's optimized
    memory layout and pre-allocates the workspace buffer.  No-op for other backends.
    Must be called after weights have been loaded and the model moved to CUDA.
    """
    for name, module in model.named_children():
        module_cls = type(module)
        module_pkg = getattr(module_cls, "__module__", "") or ""
        if module_pkg.startswith("auto_round_extension") and hasattr(module, "post_init"):
            module.post_init()
        else:
            # Recurse into containers that are not themselves QuantLinear modules.
            post_init_autoround_linear(module)


def _get_autoround_quant_linear_cls(use_gptq_zp: bool, sym: bool = True, force_pytorch: bool = False):
    """Return the best available QuantLinear class for AutoRound inference.

    Preference order:
    1. Marlin CUDA kernels (requires CUDA + gptqmodel, sym=True only) — fastest
    2. Triton kernels (requires CUDA + triton) — fast GPU kernels
    3. PyTorch fallback — works everywhere

    IMPORTANT: when Marlin is desired, ensure gptqmodel is imported before any
    triton-based library (e.g. FLA) creates triton.Autotuner instances.  In eole
    this is done by calling _preflight_marlin_import() in NNModel.build() before
    build_blocks() runs.  gptqmodel's nogil_patcher patches Autotuner at import
    time; if it runs after FLA has already created instances those instances will
    be missing the _cache_lock attribute the patched run() requires.

    Args:
        use_gptq_zp: use GPTQ-style zero-point packing.
        sym: when True, try Marlin first.
        force_pytorch: skip all CUDA-kernel backends; return PyTorch directly.

    Returns:
        (QuantLinear class, use_marlin: bool)
    """
    if not force_pytorch and cuda_is_available():
        if sym:
            try:
                from auto_round_extension.cuda.gptqmodel_marlin import get_marlin_layer

                return get_marlin_layer(), True
            except ImportError:
                pass
        try:
            if use_gptq_zp:
                from auto_round_extension.triton.qlinear_tritonv2_zp import QuantLinear
            else:
                from auto_round_extension.triton.qlinear_tritonv2 import QuantLinear
            return QuantLinear, False
        except ImportError:
            pass
    # Fallback to pure PyTorch kernels
    try:
        if use_gptq_zp:
            from auto_round_extension.torch.qlinear_torch_zp import QuantLinear
        else:
            from auto_round_extension.torch.qlinear_torch import QuantLinear
        return QuantLinear, False
    except ImportError:
        raise ImportError("Install auto-round to use autoround quantized models")
