"""Base loss class for coordinate refinement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from SFC_Torch.Fmodel import SFcalculator

from losslab.losses.settings import DEFAULT_TORCH_DEVICE


class BaseLoss(ABC):
    """Abstract base class for refinement loss functions."""

    def __init__(self, *, device: torch.device | str = DEFAULT_TORCH_DEVICE):
        """Initialize base loss.

        Args:
            device: PyTorch device for computations
        """
        self.device = torch.device(device)

    @abstractmethod
    def compute(
        self,
        coordinates: torch.Tensor,
        structure_factor_calc: Any = None,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss value.

        Args:
            coordinates: Atomic coordinates [N, 3]
            structure_factor_calc: Structure factor calculator (optional)
            return_metadata: Whether to return additional metadata

        Returns:
            Loss value or tuple of (loss, metadata_dict)
        """
        ...

    def __call__(
        self,
        coordinates: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        return self.compute(coordinates, *args, **kwargs)

    def to(self, device: torch.device | str) -> BaseLoss:
        """Move loss to device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self.device = torch.device(device)
        return self


class SFCLoss(BaseLoss):
    def __init__(
        self,
        *,
        structure_factor_calcator: SFcalculator,
        device: torch.device = DEFAULT_TORCH_DEVICE,
    ):
        """Initialize base loss.

        Args:
            device: PyTorch device for computations
        """
        self.structure_factor_calcator = structure_factor_calcator
        super().__init__(device=device)
