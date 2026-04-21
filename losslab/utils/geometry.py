"""Geometry utilities for coordinate manipulation."""

from __future__ import annotations

import contextlib
from enum import StrEnum
from typing import Any

import numpy as np
import torch


class AlignmentSelection(StrEnum):
    """Atom subset used when aligning two structures."""

    ALL = "ALL"
    CA = "CA"
    BB = "BB"


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def weighted_kabsch(
    P: Any,
    Q: Any,
    weights: np.ndarray | torch.Tensor | None = None,
    *,
    torch_backend: bool = False,
) -> tuple[Any, Any, Any]:
    """
    Weighted Kabsch alignment that computes the optimal rotation and translation
    aligning P -> Q according to given non-negative weights.

    Parameters
    - P, Q : (N,3) arrays (numpy or torch). P are source points, Q are target.
    - weights : (N,) non-negative weights. If None, uniform weights are used.
    - torch_backend : if True, use torch operations and return torch tensors.

    Returns
    - R : (3,3) rotation matrix
    - t : (3,) translation vector (applied as: aligned = (P - cP) @ R + cQ)
    - P_aligned : (N,3) aligned coordinates in same array type as chosen backend
    """
    if torch_backend:
        if not isinstance(P, torch.Tensor):
            P = torch.as_tensor(P, dtype=torch.float32)
        if not isinstance(Q, torch.Tensor):
            Q = torch.as_tensor(Q, dtype=torch.float32)
        dev = P.device

        n_points = P.shape[0]
        if weights is None:
            w = torch.ones(n_points, device=dev, dtype=torch.float32)
        else:
            w = torch.as_tensor(weights, device=dev, dtype=torch.float32)

        wsum = w.sum().clamp_min(1e-8)
        wn = (w / wsum).view(n_points, 1)

        cP = (wn * P).sum(dim=0, keepdim=True)
        cQ = (wn * Q).sum(dim=0, keepdim=True)

        X = P - cP
        Y = Q - cQ

        H = (X * wn).T @ Y
        ac: Any = (
            torch.cuda.amp.autocast(enabled=False)
            if dev.type == "cuda"
            else contextlib.nullcontext()
        )
        with ac, torch.no_grad():
            H32 = H.to(torch.float32)
            U, _, Vt = torch.linalg.svd(H32, full_matrices=False)
            detVU = torch.linalg.det(U @ Vt)
            D = torch.eye(3, dtype=torch.float32, device=dev)
            if detVU < 0:
                D[2, 2] = -1.0
            R = U @ D @ Vt

        P_aligned = (X @ R) + cQ
        t = cQ.view(3) - (cP.view(3) @ R)
        return R, t, P_aligned

    Pn = _as_numpy(P).astype(np.float64)
    Qn = _as_numpy(Q).astype(np.float64)
    n_points = Pn.shape[0]
    if weights is None:
        wn_arr = np.ones((n_points,), dtype=np.float64)
    else:
        wn_arr = np.asarray(weights, dtype=np.float64)

    wsum_f = max(float(wn_arr.sum()), 1e-8)
    wn_2d = (wn_arr / wsum_f).reshape(n_points, 1)

    cP = (wn_2d * Pn).sum(axis=0, keepdims=True)
    cQ = (wn_2d * Qn).sum(axis=0, keepdims=True)

    X = Pn - cP
    Y = Qn - cQ

    H = (X * wn_2d).T @ Y
    U, _, Vt = np.linalg.svd(H, full_matrices=False)
    detVU = np.linalg.det(U @ Vt)
    Dn = np.eye(3)
    if detVU < 0:
        Dn[2, 2] = -1.0
    R = U @ Dn @ Vt

    P_aligned = (X @ R) + cQ
    t = cQ.reshape(3) - (cP.reshape(3) @ R)
    return R, t, P_aligned


def kabsch_alignment(P, Q, *, torch_backend: bool = False):
    """Convenience wrapper performing a uniform-weight Kabsch alignment."""
    return weighted_kabsch(P, Q, weights=None, torch_backend=torch_backend)


def iterative_kabsch_alignment(
    P: Any,
    Q: Any,
    weights: np.ndarray | torch.Tensor | None = None,
    *,
    torch_backend: bool = False,
    max_iters: int = 5,
    tol: float = 1e-6,
) -> tuple[Any, Any, Any]:
    """Iteratively align P -> Q using weighted Kabsch and compose transforms."""
    if torch_backend:
        if not isinstance(P, torch.Tensor):
            P_current = torch.as_tensor(P, dtype=torch.float32)
        else:
            P_current = P
        if not isinstance(Q, torch.Tensor):
            Q_current = torch.as_tensor(Q, dtype=torch.float32, device=P_current.device)
        else:
            Q_current = Q

        w = None
        if weights is not None:
            w = torch.as_tensor(weights, dtype=torch.float32, device=P_current.device)

        R_total = torch.eye(3, dtype=torch.float32, device=P_current.device)
        t_total = torch.zeros(3, dtype=torch.float32, device=P_current.device)
        for _ in range(max(1, max_iters)):
            R, t, P_aligned = weighted_kabsch(
                P_current, Q_current, weights=w, torch_backend=True
            )
            t_total = t_total @ R + t
            R_total = R_total @ R
            shift = torch.sqrt(((P_aligned - P_current) ** 2).sum(dim=-1).mean())
            P_current = P_aligned
            if float(shift) <= tol:
                break
        return R_total, t_total, P_current

    Pn = _as_numpy(P).astype(np.float64)
    Qn = _as_numpy(Q).astype(np.float64)
    wn_np = None if weights is None else np.asarray(weights, dtype=np.float64)

    R_total_np: np.ndarray = np.eye(3, dtype=np.float64)
    t_total_np: np.ndarray = np.zeros(3, dtype=np.float64)
    P_current_np = Pn
    for _ in range(max(1, max_iters)):
        R, t, P_aligned = weighted_kabsch(
            P_current_np, Qn, weights=wn_np, torch_backend=False
        )
        t_total_np = t_total_np @ R + t
        R_total_np = R_total_np @ R
        shift = np.sqrt(((P_aligned - P_current_np) ** 2).sum(axis=-1).mean())
        P_current_np = P_aligned
        if shift <= tol:
            break
    return R_total_np, t_total_np, P_current_np


def align_pred_to_target(
    pred,
    target,
    weights: torch.Tensor | None = None,
    *,
    torch_backend: bool = True,
    iters: int = 1,
):
    """Align predicted coordinates `pred` to `target` using weighted Kabsch."""
    if torch_backend:
        if isinstance(pred, torch.Tensor):
            if pred.ndim == 3:
                P = pred[0]
                batched = True
            elif pred.ndim == 2:
                P = pred
                batched = False
            else:
                raise ValueError("`pred` must have shape [N,3] or [B,N,3]")
        else:
            P = torch.as_tensor(pred, dtype=torch.float32)
            batched = False

        if isinstance(target, torch.Tensor):
            Q = target
            if Q.ndim == 3:
                Q = Q[0]
            elif Q.ndim != 2:
                raise ValueError("`target` must have shape [N,3] or [B,N,3]")
        else:
            Q = torch.as_tensor(target, dtype=torch.float32)

        w = None
        if weights is not None:
            w = torch.as_tensor(weights, dtype=torch.float32, device=P.device)

        _, _, P_aligned = iterative_kabsch_alignment(
            P, Q, weights=w, torch_backend=True, max_iters=iters
        )
        if batched:
            return P_aligned.unsqueeze(0)
        return P_aligned

    Pn = _as_numpy(pred)
    Qn = _as_numpy(target)
    wn = None if weights is None else _as_numpy(weights)
    _, _, Pn_aligned = iterative_kabsch_alignment(
        Pn, Qn, weights=wn, torch_backend=False, max_iters=iters
    )
    return Pn_aligned


def kabsch_align(
    moving: torch.Tensor,
    reference: torch.Tensor,
    indices_moving: np.ndarray | None = None,
    indices_reference: np.ndarray | None = None,
) -> torch.Tensor:
    """Align moving coordinates to reference using iterative Kabsch alignment."""
    if indices_moving is None and indices_reference is not None:
        indices_moving = indices_reference
    if indices_reference is None and indices_moving is not None:
        indices_reference = indices_moving

    use_torch = torch.is_tensor(moving) or torch.is_tensor(reference)
    if use_torch:
        if not torch.is_tensor(moving):
            moving = torch.as_tensor(moving, dtype=torch.float32)
        if not torch.is_tensor(reference):
            reference = torch.as_tensor(
                reference, dtype=torch.float32, device=moving.device
            )
        P = moving[indices_moving] if indices_moving is not None else moving
        Q = reference[indices_reference] if indices_reference is not None else reference
        R, t, _ = iterative_kabsch_alignment(P, Q, torch_backend=True, max_iters=5)
        return moving @ R + t

    moving_np = _as_numpy(moving)
    reference_np = _as_numpy(reference)
    Pn = moving_np[indices_moving] if indices_moving is not None else moving_np
    Qn = (
        reference_np[indices_reference]
        if indices_reference is not None
        else reference_np
    )
    R, t, _ = iterative_kabsch_alignment(Pn, Qn, torch_backend=False, max_iters=5)
    return moving_np @ R + t


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


def compute_common_indices(
    moving_cra: list[str],
    reference_cra: list[str],
    alignment_selection: AlignmentSelection,
) -> tuple[np.ndarray, np.ndarray]:
    """Find indices of atoms shared between moving and reference, filtered by selection.

    Args:
        moving_cra: Atom names from the moving structure (chain-residue-atom strings).
        reference_cra: Atom names from the reference structure.
        alignment_selection: Which atom subset to keep (ALL, CA, or BB).

    Returns:
        Tuple of (index_moving, index_reference) into the respective atom lists.
    """
    alignment_selection = AlignmentSelection(alignment_selection)

    def _keep(name: str) -> bool:
        if alignment_selection is AlignmentSelection.ALL:
            return True
        if alignment_selection is AlignmentSelection.CA:
            return name.endswith("-CA")
        return name.endswith("-N") or name.endswith("-CA") or name.endswith("-C")

    reference_lookup = {
        name: idx for idx, name in enumerate(reference_cra) if _keep(name)
    }
    index_moving: list[int] = []
    index_reference: list[int] = []
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


__all__ = [
    "weighted_kabsch",
    "kabsch_alignment",
    "iterative_kabsch_alignment",
    "align_pred_to_target",
    "kabsch_align",
    "compute_rmsd",
    "apply_rigid_body_transform",
    "center_coordinates",
    "compute_common_indices",
    "AlignmentSelection",
]
