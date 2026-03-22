"""Loss functions module for coordinate refinement."""

from LossLab.losses.base import BaseLoss
from LossLab.losses.mse import MSECoordinatesLoss
from LossLab.losses.realspace import RealSpaceLoss
from LossLab.losses.saxs import DebyeLoss, DebyeRawLoss, debye_intensity, load_saxs_data


def __getattr__(name):
    if name == "CryoEMLLGLoss":
        from LossLab.losses.cryoLLGI import CryoEMLLGLoss

        return CryoEMLLGLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseLoss",
    "CryoEMLLGLoss",
    "DebyeLoss",
    "DebyeRawLoss",
    "MSECoordinatesLoss",
    "RealSpaceLoss",
    "debye_intensity",
    "load_saxs_data",
]
