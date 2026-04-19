"""Mean squared error loss for coordinate refinement."""

from __future__ import annotations

import numpy as np
import torch

from SFC_Torch.io import PDBParser

from losslab.losses.settings import DEFAULT_TORCH_DEVICE
from losslab.losses.base import BaseLoss
from losslab.utils.geometry import (
    AlignmentSelection,
    compute_common_indices,
    kabsch_align,
)


class MSECoordinatesLoss(BaseLoss):
    """MSE loss between two coordinate tensors of identical shape."""

    def __init__(
        self,
        *,
        reference_coordinates: torch.Tensor,
        device: torch.device = DEFAULT_TORCH_DEVICE,
        align: bool = True,
    ) -> None:
        super().__init__(device=device)
        self.align = align
        self.reference_coordinates = reference_coordinates.to(self.device)

    def compute(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape != self.reference_coordinates.shape:
            raise ValueError(
                "coordinates and reference_coordinates must have the same shape"
            )
        aligned = (
            kabsch_align(coordinates, self.reference_coordinates)
            if self.align
            else coordinates
        )
        return torch.mean((aligned - self.reference_coordinates) ** 2)


class MSEPdbLoss(MSECoordinatesLoss):
    """MSE loss between two PDB structures, restricted to a common atom selection."""

    def __init__(
        self,
        *,
        reference_pdb: PDBParser,
        moving_pdb: PDBParser,
        device: torch.device = DEFAULT_TORCH_DEVICE,
        align: bool = True,
        alignment_selection: AlignmentSelection = AlignmentSelection.BB,
    ) -> None:
        reference_coordinates = torch.tensor(
            reference_pdb.atom_pos,
            device=device,
            dtype=torch.float32,
        )
        super().__init__(
            reference_coordinates=reference_coordinates,
            device=device,
            align=align,
        )
        self.alignment_selection = AlignmentSelection(alignment_selection)
        self.reference_cra = list(reference_pdb.cra_name)
        self._full_reference_coordinates = self.reference_coordinates
        
        self.index_moving: np.ndarray
        self.index_reference: np.ndarray

        if moving_pdb is not None:
            self.set_moving_pdb(moving_pdb)

    def set_moving_pdb(self, moving_pdb) -> None:
        self.index_moving, self.index_reference = compute_common_indices(
            moving_pdb.cra_name,
            self.reference_cra,
            self.alignment_selection,
        )
        self.reference_coordinates = self._full_reference_coordinates[
            self.index_reference
        ]

    def compute(self, coordinates: torch.Tensor) -> torch.Tensor:
        if self.index_moving is None:
            raise ValueError("moving_pdb must be set before compute() is called")
        return super().compute(coordinates[self.index_moving])
