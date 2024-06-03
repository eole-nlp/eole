"""Module defining various utilities."""
from eole.utils.misc import use_gpu, set_random_seed
from eole.utils.alignment import make_batch_align_matrix
from eole.utils.report_manager import ReportMgr, build_report_manager
from eole.utils.statistics import Statistics
from eole.utils.optimizers import MultipleOptimizer, Optimizer, AdaFactor, FusedAdam
from eole.utils.earlystopping import EarlyStopping, scorers_from_config

__all__ = [
    "use_gpu",
    "set_random_seed",
    "ReportMgr",
    "build_report_manager",
    "Statistics",
    "MultipleOptimizer",
    "Optimizer",
    "AdaFactor",
    "FusedAdam",
    "EarlyStopping",
    "scorers_from_config",
    "make_batch_align_matrix",
]
