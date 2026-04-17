"""Mean squared error loss for coordinate refinement."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from losslab.losses.base import BaseLoss


class MSECoordinatesLoss(BaseLoss):
    """MSE loss between predicted and reference coordinates."""

    def __init__(
        self,
        reference_coordinates: torch.Tensor | None = None,
        device: torch.device | str = "cuda:0",
        reduction: str = "mean",
        align: bool = True,
        reference_pdb=None,
        moving_pdb=None,
        selection: str = "BB",
    ) -> None:
        super().__init__(device)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self.align = align
        self.selection = selection.upper()

        self.reference_cra = None
        if reference_coordinates is None and reference_pdb is not None:
            reference_coordinates = torch.tensor(
                reference_pdb.atom_pos,
                device=self.device,
                dtype=torch.float32,
            )
        if reference_coordinates is None:
            raise ValueError("reference_coordinates is required")

        self.reference_coordinates = reference_coordinates.to(self.device)

        if reference_pdb is not None:
            self.reference_cra = list(reference_pdb.cra_name)

        self.index_moving: np.ndarray | None = None
        self.index_reference: np.ndarray | None = None
        if reference_pdb is not None and moving_pdb is not None:
            self.set_moving_pdb(moving_pdb)

    def set_moving_pdb(self, moving_pdb) -> None:
        if self.reference_cra is None:
            raise ValueError("reference_pdb is required to set moving_pdb")
        self.index_moving, self.index_reference = self._compute_common_indices(
            moving_pdb.cra_name,
            self.reference_cra,
            self.selection,
        )

    @staticmethod
    def _compute_common_indices(
        moving_cra: list[str],
        reference_cra: list[str],
        selection: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if selection not in {"ALL", "CA", "BB"}:
            raise ValueError("selection must be one of: ALL, CA, BB")

        def _keep(name: str) -> bool:
            if selection == "ALL":
                return True
            if selection == "CA":
                return name.endswith("-CA")
            if selection == "BB":
                return (
                    name.endswith("-N") or name.endswith("-CA") or name.endswith("-C")
                )
            return True

        reference_lookup = {
            name: idx for idx, name in enumerate(reference_cra) if _keep(name)
        }
        index_moving = []
        index_reference = []
        for idx, name in enumerate(moving_cra):
            if not _keep(name):
                continue
            ref_idx = reference_lookup.get(name)
            if ref_idx is not None:
                index_moving.append(idx)
                index_reference.append(ref_idx)

        if not index_moving:
            raise ValueError("No overlapping atoms found between moving and reference")

        return np.array(index_moving), np.array(index_reference)

    def compute(
        self,
        coordinates: torch.Tensor,
        structure_factor_calc: Any = None,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if (
            self.index_moving is None or self.index_reference is None
        ) and coordinates.shape != self.reference_coordinates.shape:
            raise ValueError(
                "coordinates and reference_coordinates must have the same shape"
            )
        if self.align:
            from losslab.utils.geometry import kabsch_align

            aligned_coords = kabsch_align(
                coordinates,
                self.reference_coordinates,
                indices_moving=self.index_moving,
                indices_reference=self.index_reference,
            )
        else:
            aligned_coords = coordinates

        if self.index_moving is not None and self.index_reference is not None:
            aligned_coords = aligned_coords[self.index_moving]
            reference_coords = self.reference_coordinates[self.index_reference]
        else:
            reference_coords = self.reference_coordinates

        diff = aligned_coords - reference_coords
        if self.reduction == "sum":
            loss = torch.sum(diff**2)
        else:
            loss = torch.mean(diff**2)

        if return_metadata:
            rmse = torch.sqrt(torch.mean(diff**2)).item()
            return loss, {"rmse": rmse}
        return loss
