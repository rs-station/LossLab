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
        structure_factor_calc: Any = None,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Compute loss value.

        Args:
            coordinates: Atomic coordinates [N, 3]
            structure_factor_calc: Optional structure factor calculator
            return_metadata: Whether to return additional metadata
            **kwargs: Additional loss-specific arguments

        Returns:
            Loss value or tuple of (loss, metadata_dict)
        """
        pass

    def __call__(
        self,
        coordinates: torch.Tensor,
        structure_factor_calc: Any = None,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Compute loss (callable interface).

        Args:
            coordinates: Atomic coordinates [N, 3]
            structure_factor_calc: Optional structure factor calculator
            return_metadata: Whether to return additional metadata
            **kwargs: Additional loss-specific arguments

        Returns:
            Loss value or tuple of (loss, metadata_dict)
        """
        return self.compute(
            coordinates,
            structure_factor_calc=structure_factor_calc,
            return_metadata=return_metadata,
            **kwargs,
        )

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
