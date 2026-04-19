"""Base loss class for coordinate refinement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch
from SFC_Torch.Fmodel import SFcalculator

from losslab.losses.settings import DEFAULT_TORCH_DEVICE


class ValidLoss(Protocol):
    def compute(self, coordinates: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.compute(coordinates)


class BaseLoss(ABC):
    """Abstract base class for refinement loss functions."""

    def __init__(self, *, device: torch.device | str = DEFAULT_TORCH_DEVICE):
        """Initialize base loss.

        Args:
            device: PyTorch device for computations
        """
        self.device = torch.device(device)

    @abstractmethod
    def compute(self, coordinates: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.compute(coordinates)

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
        structure_factor_calculator: SFcalculator,
        device: torch.device = DEFAULT_TORCH_DEVICE,
    ):
        """Initialize base loss.

        Args:
            device: PyTorch device for computations
        """
        self.structure_factor_calculator = structure_factor_calculator
        super().__init__(device=device)
