import eole.inputters
import eole.encoders
import eole.decoders
import eole.models
import eole.utils
import eole.modules
import sys
import eole.utils.optimizers

eole.utils.optimizers.Optim = eole.utils.optimizers.Optimizer
sys.modules["eole.Optim"] = eole.utils.optimizers

# For Flake
__all__ = [
    eole.inputters,
    eole.encoders,
    eole.decoders,
    eole.models,
    eole.utils,
    eole.modules,
]

__version__ = "3.5.1"
