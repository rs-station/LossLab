"""OF3/Rocket atom extraction and alignment helpers for cryo losses."""

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

def get_res_names(aatype: np.ndarray) -> np.ndarray:
    """Get 3-letter residue names from unified token set."""
    token_names = np.array(TOKEN_TYPES_WITH_GAP)  # ["ALA", ..., "G", ..., "DA", ..., "GAP"]
    return token_names[np.clip(aatype, 0, len(token_names) - 1)] 

def _restype_to_token_idx(restype, vocab_size: int) -> np.ndarray:
    """
    Accepts restype in any of:
      - [n_tok] (already indices)
      - [n_tok, vocab] (one-hot/logits)
      - [1, n_tok, vocab] (batched one-hot/logits)   <-- your case
    Returns: [n_tok] int64 token indices clipped into vocab.
    """
    rt = rk_utils.assert_numpy(restype)

    # peel leading singleton dims (batch, etc.)
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

def extract_allatoms(outputs, feats, cra_name_sfc: list):
    token_names = np.asarray(TOKEN_TYPES_WITH_GAP, dtype=object)
    atom_names_dict = TOKEN_NAME_TO_ATOM_NAMES

    # restype: (1, n_tok, vocab) in your case
    token_idx = _restype_to_token_idx(feats["restype"], len(token_names))
    res_names = token_names[token_idx]
    n_res = int(res_names.shape[0])
    chain_resid = np.asarray([f"A-{i}-" for i in range(n_res)], dtype=object)

    # atom-flat predicted coords WITH GRAD
    atom_pos_pred = outputs["atom_positions_predicted"][0].squeeze(0)  # [n_atom, 3]
    if atom_pos_pred.ndim != 2 or atom_pos_pred.shape[-1] != 3:
        raise ValueError(f"Unexpected atom_positions_predicted shape: {tuple(atom_pos_pred.shape)}")
    n_atom = int(atom_pos_pred.shape[0])

    # atom-flat mask
    atom_mask = rk_utils.assert_numpy(feats["atom_mask"][0]).reshape(-1)  # [n_atom]
    if atom_mask.shape[0] != n_atom:
        raise ValueError(f"atom_mask {atom_mask.shape[0]} != n_atom {n_atom}")

    # per-atom pLDDT (no grad needed)
    pl_logits = outputs["plddt_logits"][0].squeeze(0).to(torch.float32)  # [n_atom, n_bins]
    probs = torch.softmax(pl_logits, dim=-1)
    bin_centers = torch.linspace(0.0, 1.0, steps=probs.shape[-1], device=probs.device)
    plddt_atom = ((probs * bin_centers).sum(dim=-1) * 100.0).detach().cpu().numpy()  # [n_atom]

    cra_names = []
    atom_positions = []
    plddts = []

    global_atom_index = 0
    for i in range(n_res):
        rn = res_names[i]
        resname = rn.decode() if isinstance(rn, (bytes, np.bytes_)) else str(rn)

        atoms = atom_names_dict.get(resname, [])
        for aname in atoms:
            if global_atom_index >= n_atom:
                raise IndexError("global_atom_index ran past atom_positions_predicted")

            if atom_mask[global_atom_index] == 0:
                global_atom_index += 1
                continue

            cra_names.append(f"{chain_resid[i]}{resname}-{aname}")
            # IMPORTANT: keep grad!
            atom_positions.append(atom_pos_pred[global_atom_index].to(torch.float32))
            plddts.append(float(plddt_atom[global_atom_index]))
            global_atom_index += 1

    if not atom_positions:
        raise RuntimeError("No atoms extracted (empty atom_positions).")

    positions_atom = torch.stack(atom_positions, dim=0)               # [n_kept, 3], REQUIRES_GRAD if input does
    plddt_atom_t = torch.tensor(plddts, dtype=torch.float32, device=positions_atom.device)

    # reorder to SFC topology
    idx_map = {c: k for k, c in enumerate(cra_names)}
    missing = [c for c in cra_name_sfc if c not in idx_map]
    if missing:
        raise AssertionError(f"Topology mismatch; missing {len(missing)} CRA (showing 10): {missing[:10]}")

    reorder_index = [idx_map[c] for c in cra_name_sfc]

    return positions_atom[reorder_index], plddt_atom_t[reorder_index]

from LossLab.utils import geometry as geom
import numpy as np
import torch

def _anchor_idx_from_cra(cra_name):
    # prefer protein backbone if present; else RNA backbone; else all atoms
    atoms = np.array([str(c).split("-")[-1].replace("*", "'") for c in cra_name], dtype=object)

    prot = np.isin(atoms, ["N", "CA", "C"])
    if prot.any():
        return np.where(prot)[0]

    rna = np.isin(atoms, ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","C1'"])
    if rna.any():
        return np.where(rna)[0]

    return np.arange(len(cra_name), dtype=np.int64)


def position_alignment(
    rollout_output,
    batch,
    best_pos,
    exclude_res,
    cra_name,
    domain_segs=None,
    reference_bfactor=None,     # optional per-atom B-factors (Tensor)
):
    """
    Minimal alignment:
      - coords: Tensor [..., N, 3] or [N, 3]
      - best_pos: reference coords Tensor [..., N, 3] or [N, 3]
      - exclude_res: indices to mask out (list/array/Tensor), optional
      - plddt OR reference_bfactor provide weights; if neither, uniform weights
    Returns:
      aligned_xyz (torch.float32 [N,3]),
      plddts_res (np.float32 [N]),
      pseudo_Bs  (torch.float32 [N])
    """
    coords, plddts = extract_allatoms(rollout_output, batch, cra_name)
    pseudo_Bs = rk_utils.plddt2pseudoB_pt(plddts)
    if reference_bfactor is None:
        pseudoB_np = rk_utils.assert_numpy(pseudo_Bs)
        cutoff1 = np.quantile(pseudoB_np, 0.3)
        cutoff2 = cutoff1 * 1.5
        weights = rk_utils.weighting(pseudoB_np, cutoff1, cutoff2)
    else:
        assert reference_bfactor.shape == pseudo_Bs.shape, (
            "Reference bfactor should have same shape as model bfactor!"
        )
        reference_bfactor_np = rk_utils.assert_numpy(reference_bfactor)
        cutoff1 = np.quantile(reference_bfactor_np, 0.3)
        cutoff2 = cutoff1 * 1.5
        weights = rk_utils.weighting(reference_bfactor_np, cutoff1, cutoff2)

    # --- alignment ---
    aligned_xyz = rk_coordinates.iterative_kabsch_alignment(
        coords, best_pos, cra_name,
        weights=weights,
        exclude_res=exclude_res,
        domain_segs=domain_segs,
    )
    mask = cra_exclude_mask(cra_name, exclude_res=exclude_res)

    rmsd_pre  = weighted_rmsd(coords,      best_pos, w=weights, mask=mask)
    rmsd_post = weighted_rmsd(aligned_xyz, best_pos, w=weights, mask=mask)

    logger.debug("RMSD pre-align: %.3f  post-align: %.3f", rmsd_pre, rmsd_post)

    if rmsd_post > rmsd_pre + 1e-3:
        logger.warning("RMSD increased after alignment — check ordering / weights / mask.")

    return aligned_xyz, plddts, pseudo_Bs.detach()

#debug
def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def weighted_rmsd(xyz_a, xyz_b, w=None, mask=None):
    """
    xyz_*: [N,3]
    w:     [N] (optional) weights per atom (can be unnormalized)
    mask:  [N] bool (optional) atoms to include
    """
    a = _to_numpy(xyz_a).astype(np.float64, copy=False)
    b = _to_numpy(xyz_b).astype(np.float64, copy=False)
    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 3, (a.shape, b.shape)

    if mask is None:
        mask = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    else:
        mask = mask & np.isfinite(a).all(-1) & np.isfinite(b).all(-1)

    if w is None:
        diff2 = ((a[mask] - b[mask]) ** 2).sum(-1)   # [M]
        return float(np.sqrt(diff2.mean()))
    else:
        ww = _to_numpy(w).reshape(-1).astype(np.float64, copy=False)
        assert ww.shape[0] == a.shape[0], (ww.shape, a.shape)
        ww = ww[mask]
        ww_sum = ww.sum()
        if ww_sum <= 0:
            raise ValueError(f"Non-positive weight sum after masking: {ww_sum}")
        ww = ww / ww_sum
        diff2 = ((a[mask] - b[mask]) ** 2).sum(-1)
        return float(np.sqrt((ww * diff2).sum()))

def cra_exclude_mask(cra_name, exclude_res=None):
    """
    cra_name entries look like 'A-12-ALA-CA' (from your construction).
    exclude_res: iterable of residue indices (ints) to exclude.
    Returns: [N] bool mask (True = keep)
    """
    if not exclude_res:
        return None
    ex = set(int(r) for r in exclude_res)
    keep = np.ones(len(cra_name), dtype=bool)
    for i, cra in enumerate(cra_name):
        # 'A-12-ALA-CA' -> parts[1] == '12'
        parts = str(cra).split("-")
        if len(parts) >= 2:
            try:
                resid = int(parts[1])
                if resid in ex:
                    keep[i] = False
            except ValueError:
                pass
    return keep