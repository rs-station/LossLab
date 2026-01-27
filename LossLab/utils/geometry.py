"""Geometry utilities for coordinate manipulation."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def kabsch_align(
    moving: torch.Tensor,
    reference: torch.Tensor,
    indices_moving: np.ndarray | None = None,
    indices_reference: np.ndarray | None = None,
) -> torch.Tensor:
    """Align moving coordinates to reference using Kabsch algorithm.

    Args:
        moving: Coordinates to align [N, 3]
        reference: Reference coordinates [M, 3]
        indices_moving: Indices to use from moving (default: all)
        indices_reference: Indices to use from reference (default: all)

    Returns:
        Aligned coordinates [N, 3]
    """
    device = moving.device
    dtype = moving.dtype

    # Convert to numpy for Kabsch
    moving_np = moving.detach().cpu().numpy()
    reference_np = reference.detach().cpu().numpy()

    # Apply indices if provided
    if indices_moving is not None:
        moving_subset = moving_np[indices_moving]
    else:
        moving_subset = moving_np

    if indices_reference is not None:
        reference_subset = reference_np[indices_reference]
    else:
        reference_subset = reference_np

    # Compute centroids
    com_moving = np.mean(moving_subset, axis=0)
    com_reference = np.mean(reference_subset, axis=0)

    # Align vectors
    rotation, _ = Rotation.align_vectors(
        reference_subset - com_reference, moving_subset - com_moving
    )

    # Convert back to torch
    rotation_matrix = torch.tensor(rotation.as_matrix(), device=device, dtype=dtype)
    centroid_moving = torch.tensor(com_moving, device=device, dtype=dtype)
    centroid_reference = torch.tensor(com_reference, device=device, dtype=dtype)

    # Apply transformation to all coordinates
    aligned = (
        torch.matmul(moving - centroid_moving, rotation_matrix.T) + centroid_reference
    )

    return aligned


def compute_rmsd(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    indices: np.ndarray | None = None,
) -> float:
    """Compute RMSD between two sets of coordinates.

    Args:
        coords1: First set of coordinates [N, 3]
        coords2: Second set of coordinates [N, 3]
        indices: Indices to use for RMSD calculation (default: all)

    Returns:
        RMSD value in Angstroms
    """
    if indices is not None:
        coords1 = coords1[indices]
        coords2 = coords2[indices]

    diff = coords1 - coords2
    rmsd = torch.sqrt(torch.mean(torch.sum(diff**2, dim=-1)))
    return rmsd.item()


def apply_rigid_body_transform(
    coordinates: torch.Tensor,
    rotation_matrix: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Apply rigid body transformation to coordinates.

    Args:
        coordinates: Input coordinates [N, 3]
        rotation_matrix: Rotation matrix [3, 3]
        translation: Translation vector [3]

    Returns:
        Transformed coordinates [N, 3]
    """
    return torch.matmul(coordinates, rotation_matrix.T) + translation


def center_coordinates(coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Center coordinates at origin.

    Args:
        coordinates: Input coordinates [N, 3]

    Returns:
        Tuple of (centered_coordinates, centroid)
    """
    centroid = torch.mean(coordinates, dim=0)
    centered = coordinates - centroid
    return centered, centroid
