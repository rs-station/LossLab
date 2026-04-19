"""Base loss class for coordinate refinement."""

from abc import ABC, abstractmethod
from SFC_Torch.Fmodel import SFcalculator

import torch

from losslab.losses.settings import DEFAULT_TORCH_DEVICE


class BaseLoss(ABC):
    """Abstract base class for refinement loss functions."""

    def __init__(self, *, device: torch.device = DEFAULT_TORCH_DEVICE):
        """Initialize base loss.

        Args:
            device: PyTorch device for computations
        """
        self.device = torch.device(device)

    @abstractmethod
    def compute(
        self,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss value.

        Args:
            coordinates: Atomic coordinates [N, 3]

        Returns:
            Loss value 
        """
        ...

    def __call__(
        self,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute(
            coordinates,
        )

    def to(self, device: torch.device) -> "BaseLoss":
        """Move loss to device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self.device = torch.device(device)
        return self


class SFCLoss(BaseLoss):

    def __init__(self, *, structure_factor_calcator: SFcalculator, device: torch.device = DEFAULT_TORCH_DEVICE):
        """Initialize base loss.

        Args:
            device: PyTorch device for computations
        """
        self.structure_factor_calcator = structure_factor_calcator
        super().__init__(device=device)
