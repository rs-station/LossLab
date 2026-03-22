"""OF3/ROCKET atom extraction and alignment helpers."""

from __future__ import annotations

import logging

import numpy as np
import torch

try:
    from openfold3.core.data.resources.token_atom_constants import (
        TOKEN_NAME_TO_ATOM_NAMES,
        TOKEN_TYPES_WITH_GAP,
    )
except ModuleNotFoundError:
    from openfold3.core.np.token_atom_constants import (
        TOKEN_NAME_TO_ATOM_NAMES,
        TOKEN_TYPES_WITH_GAP,
    )
from rocket import coordinates as rk_coordinates
from rocket import utils as rk_utils

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# OF3 token/atom helpers
# ------------------------------------------------------------------


def _restype_to_token_idx(restype: np.ndarray, vocab_size: int) -> np.ndarray:
    """Convert restype array to 1-D int64 token indices."""
    rt = rk_utils.assert_numpy(restype)

    while rt.ndim > 2 and rt.shape[0] == 1:
        rt = rt[0]

    if rt.ndim == 1:
        idx = rt.astype(np.int64)
    elif rt.ndim == 2 and rt.shape[-1] == vocab_size:
        idx = rt.argmax(-1).astype(np.int64)
    else:
        raise ValueError(f"Unexpected restype shape after squeeze: {rt.shape}")

    return np.clip(idx.reshape(-1), 0, vocab_size - 1)


def get_res_names(restype: np.ndarray) -> np.ndarray:
    """Get 3-letter residue/token names from unified token set."""
    token_names = np.asarray(TOKEN_TYPES_WITH_GAP, dtype=object)
    token_idx = _restype_to_token_idx(restype, len(token_names))
    return token_names[token_idx]


# ------------------------------------------------------------------
# Atom extraction from OF3 rollout
# ------------------------------------------------------------------


def extract_allatoms(
    outputs: dict, feats: dict, cra_name_sfc: list
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract all-atom positions from OF3 rollout in SFC order.

    Returns (positions [N, 3], plddt [N]) with gradients preserved.
    """
    token_names = np.asarray(TOKEN_TYPES_WITH_GAP, dtype=object)

    token_idx = _restype_to_token_idx(feats["restype"], len(token_names))
    res_names = token_names[token_idx]
    n_res = int(res_names.shape[0])
    chain_resid = np.asarray([f"A-{i}-" for i in range(n_res)], dtype=object)

    atom_pos_pred = outputs["atom_positions_predicted"][0].squeeze(0)
    if atom_pos_pred.ndim != 2 or atom_pos_pred.shape[-1] != 3:
        raise ValueError(
            f"Unexpected atom_positions_predicted shape: {tuple(atom_pos_pred.shape)}"
        )
    n_atom = int(atom_pos_pred.shape[0])

    atom_mask = rk_utils.assert_numpy(feats["atom_mask"][0]).reshape(-1)
    if atom_mask.shape[0] != n_atom:
        raise ValueError(f"atom_mask {atom_mask.shape[0]} != n_atom {n_atom}")

    pl_logits = outputs["plddt_logits"][0].squeeze(0).to(torch.float32)
    probs = torch.softmax(pl_logits, dim=-1)
    bin_centers = torch.linspace(0.0, 1.0, steps=probs.shape[-1], device=probs.device)
    plddt_atom = ((probs * bin_centers).sum(dim=-1) * 100.0).detach().cpu().numpy()

    cra_names: list[str] = []
    atom_positions: list[torch.Tensor] = []
    plddts: list[float] = []

    global_atom_index = 0
    for i in range(n_res):
        rn = res_names[i]
        resname = rn.decode() if isinstance(rn, (bytes, np.bytes_)) else str(rn)
        atoms = TOKEN_NAME_TO_ATOM_NAMES.get(resname, [])
        for aname in atoms:
            if global_atom_index >= n_atom:
                raise IndexError("global_atom_index ran past atom_positions_predicted")
            if atom_mask[global_atom_index] == 0:
                global_atom_index += 1
                continue

            cra_names.append(f"{chain_resid[i]}{resname}-{aname}")
            atom_positions.append(atom_pos_pred[global_atom_index].to(torch.float32))
            plddts.append(float(plddt_atom[global_atom_index]))
            global_atom_index += 1

    if not atom_positions:
        raise RuntimeError("No atoms extracted (empty atom_positions).")

    positions_atom = torch.stack(atom_positions, dim=0)
    plddt_atom_t = torch.tensor(
        plddts, dtype=torch.float32, device=positions_atom.device
    )

    idx_map = {c: k for k, c in enumerate(cra_names)}
    missing = [c for c in cra_name_sfc if c not in idx_map]
    if missing:
        raise AssertionError(
            f"Topology mismatch; missing {len(missing)} CRA "
            f"(showing 10): {missing[:10]}"
        )

    reorder_index = [idx_map[c] for c in cra_name_sfc]
    return positions_atom[reorder_index], plddt_atom_t[reorder_index]


# ------------------------------------------------------------------
# Alignment
# ------------------------------------------------------------------


def position_alignment(
    rollout_output,
    batch,
    best_pos,
    exclude_res,
    cra_name,
    domain_segs=None,
    reference_bfactor=None,
):
    """Kabsch-align OF3 rollout coords to reference.

    Returns (aligned_xyz, plddts, pseudo_Bs).
    """
    coords, plddts = extract_allatoms(rollout_output, batch, cra_name)
    pseudo_bs = rk_utils.plddt2pseudoB_pt(plddts)

    if reference_bfactor is None:
        pseudo_b_np = rk_utils.assert_numpy(pseudo_bs)
        cutoff1 = np.quantile(pseudo_b_np, 0.3)
        cutoff2 = cutoff1 * 1.5
        weights = rk_utils.weighting(pseudo_b_np, cutoff1, cutoff2)
    else:
        assert reference_bfactor.shape == pseudo_bs.shape, (
            "reference_bfactor must match model bfactor shape"
        )
        ref_b_np = rk_utils.assert_numpy(reference_bfactor)
        cutoff1 = np.quantile(ref_b_np, 0.3)
        cutoff2 = cutoff1 * 1.5
        weights = rk_utils.weighting(ref_b_np, cutoff1, cutoff2)

    aligned_xyz = rk_coordinates.iterative_kabsch_alignment(
        coords,
        best_pos,
        cra_name,
        weights=weights,
        exclude_res=exclude_res,
        domain_segs=domain_segs,
    )
    mask = cra_exclude_mask(cra_name, exclude_res=exclude_res)

    rmsd_pre = weighted_rmsd(coords, best_pos, w=weights, mask=mask)
    rmsd_post = weighted_rmsd(aligned_xyz, best_pos, w=weights, mask=mask)

    logger.debug(
        "RMSD pre-align: %.3f  post-align: %.3f",
        rmsd_pre,
        rmsd_post,
    )
    if rmsd_post > rmsd_pre + 1e-3:
        logger.warning(
            "RMSD increased after alignment — check ordering / weights / mask."
        )

    return aligned_xyz, plddts, pseudo_bs.detach()


# ------------------------------------------------------------------
# RMSD and masking helpers
# ------------------------------------------------------------------


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def weighted_rmsd(xyz_a, xyz_b, w=None, mask=None):
    """Weighted RMSD between two [N,3] coordinate arrays."""
    a = _to_numpy(xyz_a).astype(np.float64, copy=False)
    b = _to_numpy(xyz_b).astype(np.float64, copy=False)
    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 3

    finite = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    if mask is not None:
        finite = mask & finite

    if w is None:
        diff2 = ((a[finite] - b[finite]) ** 2).sum(-1)
        return float(np.sqrt(diff2.mean()))

    ww = _to_numpy(w).reshape(-1).astype(np.float64, copy=False)
    assert ww.shape[0] == a.shape[0]
    ww = ww[finite]
    ww_sum = ww.sum()
    if ww_sum <= 0:
        raise ValueError(f"Non-positive weight sum after masking: {ww_sum}")
    ww = ww / ww_sum
    diff2 = ((a[finite] - b[finite]) ** 2).sum(-1)
    return float(np.sqrt((ww * diff2).sum()))


def cra_exclude_mask(cra_name, exclude_res=None):
    """Build boolean keep-mask from CRA names and exclusion list."""
    if not exclude_res:
        return None
    ex = {int(r) for r in exclude_res}
    keep = np.ones(len(cra_name), dtype=bool)
    for i, cra in enumerate(cra_name):
        parts = str(cra).split("-")
        if len(parts) >= 2:
            try:
                resid = int(parts[1])
                if resid in ex:
                    keep[i] = False
            except ValueError:
                pass
    return keep


__all__ = [
    "cra_exclude_mask",
    "extract_allatoms",
    "get_res_names",
    "position_alignment",
    "weighted_rmsd",
]
