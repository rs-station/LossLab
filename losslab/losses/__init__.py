"""Loss functions module for coordinate refinement."""

from losslab.losses.base import BaseLoss
from losslab.losses.mse import MSECoordinatesLoss, MSEPdbLoss
from losslab.losses.realspace import RealSpaceLoss

__all__ = ["RealSpaceLoss", "MSECoordinatesLoss", "MSEPdbLoss", "BaseLoss"]
