"""Mean squared error loss for coordinate refinement."""

from __future__ import annotations

import torch

from LossLab.losses.base import BaseLoss


class MSECoordinatesLoss(BaseLoss):
    """MSE loss between predicted and reference coordinates."""

    def __init__(
        self,
        reference_coordinates: torch.Tensor,
        device: torch.device | str = "cuda:0",
        reduction: str = "mean",
    ) -> None:
        super().__init__(device)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self.reference_coordinates = reference_coordinates.to(self.device)

    def compute(
        self,
        coordinates: torch.Tensor,
        structure_factor_calc=None,
        return_metadata: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if coordinates.shape != self.reference_coordinates.shape:
            raise ValueError(
                "coordinates and reference_coordinates must have the same shape"
            )
        diff = coordinates - self.reference_coordinates
        if self.reduction == "sum":
            loss = torch.sum(diff**2)
        else:
            loss = torch.mean(diff**2)

        if return_metadata:
            rmse = torch.sqrt(torch.mean(diff**2)).item()
            return loss, {"rmse": rmse}
        return loss
