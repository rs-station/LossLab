"""Loss functions module for coordinate refinement."""

from losslab.losses.base import BaseLoss
from losslab.losses.mse import MSECoordinatesLoss

__all__ = ["MSECoordinatesLoss", "BaseLoss"]
