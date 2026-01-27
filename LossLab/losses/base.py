"""Base loss class for coordinate refinement."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseLoss(ABC):
    """Abstract base class for refinement loss functions."""

    def __init__(self, device: torch.device | str = "cuda:0"):
        """Initialize base loss.

        Args:
            device: PyTorch device for computations
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    @abstractmethod
    def compute(
        self,
        coordinates: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Compute loss value.

        Args:
            coordinates: Atomic coordinates [N, 3]
            **kwargs: Additional loss-specific arguments

        Returns:
            Loss value or tuple of (loss, metadata_dict)
        """
        pass

    def __call__(
        self,
        coordinates: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Compute loss (callable interface).

        Args:
            coordinates: Atomic coordinates [N, 3]
            **kwargs: Additional loss-specific arguments

        Returns:
            Loss value or tuple of (loss, metadata_dict)
        """
        return self.compute(coordinates, **kwargs)

    def to(self, device: torch.device | str) -> "BaseLoss":
        """Move loss to device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        return self
