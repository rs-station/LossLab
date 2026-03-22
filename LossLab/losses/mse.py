"""Mean squared error loss for coordinate refinement."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from LossLab.losses.base import BaseLoss
from LossLab.utils.sequence import compute_common_indices

logger = logging.getLogger(__name__)


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
            self._reference_sequence = getattr(reference_pdb, "sequence", None)

        self.index_moving: np.ndarray | None = None
        self.index_reference: np.ndarray | None = None
        if reference_pdb is not None and moving_pdb is not None:
            self.set_moving_pdb(moving_pdb)

    def set_moving_pdb(self, moving_pdb) -> None:
        if self.reference_cra is None:
            raise ValueError("reference_pdb is required to set moving_pdb")
        self.index_moving, self.index_reference = compute_common_indices(
            list(moving_pdb.cra_name),
            self.reference_cra,
            self.selection,
            moving_sequence=getattr(moving_pdb, "sequence", None),
            reference_sequence=getattr(self, "_reference_sequence", None),
        )

    def compute(
        self,
        coordinates: torch.Tensor,
        structure_factor_calc=None,
        return_metadata: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if (
            self.index_moving is None or self.index_reference is None
        ) and coordinates.shape != self.reference_coordinates.shape:
            raise ValueError(
                "coordinates and reference_coordinates must have the same shape"
            )
        if self.align:
            from LossLab.utils.geometry import kabsch_align

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

    def set_matching_indices(
        self,
        query_idx: list[int],
        ref_idx: list[int],
    ) -> None:
        """Pre-compute matching atom indices for rollout -> reference mapping.

        Args:
            query_idx: Indices into the flat atom array of the OF3 prediction.
            ref_idx: Corresponding indices into ``self.reference_coordinates``.
        """
        self._query_idx = query_idx
        self._ref_idx = ref_idx

    def compute_from_rollout(
        self,
        rollout_output: dict,
        batch: dict,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Compute MSE loss directly from OF3 diffusion rollout output.

        Indexes into ``atom_positions_predicted`` with pre-stored query/ref
        indices (set via :meth:`set_matching_indices`), Kabsch-aligns, and
        returns the MSE.
        """
        if not hasattr(self, "_query_idx") or self._query_idx is None:
            raise ValueError(
                "Call set_matching_indices(query_idx, ref_idx) before "
                "compute_from_rollout"
            )

        xl_pred = rollout_output["atom_positions_predicted"]  # [1, 1, N_atom, 3]
        pred_coords = xl_pred[0, 0, self._query_idx]  # [N_match, 3]
        ref_coords = self.reference_coordinates[self._ref_idx].to(pred_coords.device)

        if self.align:
            from LossLab.utils.geometry import kabsch_align

            pred_coords = kabsch_align(pred_coords, ref_coords)

        diff = pred_coords - ref_coords
        if self.reduction == "sum":
            loss = torch.sum(diff**2)
        else:
            loss = torch.mean(diff**2)

        if return_metadata:
            rmse = torch.sqrt(torch.mean(diff**2)).item()
            return loss, {"rmse": rmse}
        return loss
