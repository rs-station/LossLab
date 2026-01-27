"""Loss functions module for coordinate refinement."""

from LossLab.losses.base import BaseLoss
from LossLab.losses.realspace import RealSpaceLoss

__all__ = ["RealSpaceLoss", "BaseLoss"]
