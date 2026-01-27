import os

__version__ = "0.4.4"
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
EOLE_TORCH_COMPILE = os.environ.get("EOLE_TORCH_COMPILE", "0") == "1"

EOLE_COMPILE_MODE = os.environ.get("EOLE_COMPILE_MODE", "2")
# Mode = 0 : Decoder Level - cudagraphs True
# Mode = 1 : Decoder Level - cudagraphs False
# Mode = 2 : Decoder Layer Level - cudagraphs True
# Mode = 3 : Decoder Layer Level - cudagraphs False
