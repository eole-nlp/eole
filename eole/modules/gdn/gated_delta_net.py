"""Gated Delta Network (GatedDeltaNet) – public compatibility entry point.

The implementation is split into:

- kernels.py: optional fast paths, PyTorch reference kernels, dispatch
  helpers and the gated RMSNorm variants;
- layer.py: the GatedDeltaNet nn.Module.

This module re-exports the full former namespace so existing
``eole.modules.gdn.gated_delta_net.X`` references (including tests) keep
working. New code should import from kernels.py / layer.py directly.
"""

from eole.modules.gdn import kernels
from eole.modules.gdn import gdn_ab, layer  # noqa: F401  (re-exported)
from eole.modules.gdn.gdn_ab import gdn_ab_projections  # noqa: F401
from eole.modules.gdn.layer import GatedDeltaNet  # noqa: F401

# Re-export the whole kernels namespace (including private helpers that tests
# and inference backends reference through this module).
globals().update({name: getattr(kernels, name) for name in dir(kernels) if not name.startswith("__")})
