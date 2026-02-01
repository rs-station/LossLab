"""Loss functions module for coordinate refinement."""

from LossLab.losses.base import BaseLoss
from LossLab.losses.mse import MSECoordinatesLoss
from LossLab.losses.realspace import RealSpaceLoss

__all__ = ["RealSpaceLoss", "MSECoordinatesLoss", "BaseLoss"]
