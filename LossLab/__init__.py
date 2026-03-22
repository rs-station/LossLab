"""LossLab: Modular coordinate refinement library."""

from LossLab.losses.realspace import RealSpaceLoss
from LossLab.refinement.config import RefinementConfig
from LossLab.refinement.engine import RefinementEngine

__version__ = "0.1.0"


def __getattr__(name):
    if name == "CryoEMLLGLoss":
        from LossLab.losses.cryoLLGI import CryoEMLLGLoss

        return CryoEMLLGLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CryoEMLLGLoss",
    "RealSpaceLoss",
    "RefinementConfig",
    "RefinementEngine",
]
