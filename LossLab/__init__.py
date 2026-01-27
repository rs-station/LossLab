"""LossLab: Modular coordinate refinement library."""

from LossLab.losses.realspace import RealSpaceLoss
from LossLab.refinement.config import RefinementConfig
from LossLab.refinement.engine import RefinementEngine

__version__ = "0.1.0"

__all__ = [
    "RealSpaceLoss",
    "RefinementEngine",
    "RefinementConfig",
]
