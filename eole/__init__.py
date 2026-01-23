import os

__version__ = "0.4.4"
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
EOLE_TORCH_COMPILE = os.environ.get("EOLE_TORCH_COMPILE", "0") == "1"
