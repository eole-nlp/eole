import os

__version__ = "0.6.4"
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
EOLE_TORCH_COMPILE = os.environ.get("EOLE_TORCH_COMPILE", "0") == "1"

if EOLE_TORCH_COMPILE:
    # FLA can use Triton's default cache before Inductor initializes its own.
    # Pin that default early: switching caches during warmup can select different
    # autotuned kernels for the first request and subsequent requests.
    # Mirror Triton's default without importing an optional CUDA dependency.
    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        os.path.join(os.environ.get("TRITON_HOME", os.path.expanduser("~/")), ".triton", "cache"),
    )

EOLE_COMPILE_MODE = os.environ.get("EOLE_COMPILE_MODE", "2")
# Mode = 0 : Decoder Level - cudagraphs True
# Mode = 1 : Decoder Level - cudagraphs False
# Mode = 2 : Decoder Layer Level - cudagraphs True
# Mode = 3 : Decoder Layer Level - cudagraphs False
