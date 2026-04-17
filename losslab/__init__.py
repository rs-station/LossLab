"""LossLab: Modular coordinate refinement library."""

from losslab.losses.realspace import RealSpaceLoss
from losslab.refinement.config import RefinementConfig
from losslab.refinement.engine import RefinementEngine

__version__ = "0.1.0"

__all__ = [
    "RealSpaceLoss",
    "RefinementEngine",
    "RefinementConfig",
]
