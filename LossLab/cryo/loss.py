"""Cryo-EM LLG loss — mirrors ROCKET ``refinement_cryoem.py`` exactly.

Pipeline per step:
  1. Extract all-atom positions from OF3 rollout (with grad).
  2. Kabsch-align to reference pose (no_grad), apply STE for gradient flow.
  3. Set pseudo-B-factors on both SFCs from pLDDT.
  4. Compute per-atom RSCC → update B-factors on both SFCs.
  5. Rigid-body refinement (RBR) via quaternion LBFGS on the RBR LLGloss.
  6. Score with ``-LLG`` (main LLGloss) + B-factor-weighted L2 penalty.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

import gemmi
from SFC_Torch import PDBParser

from LossLab.cryo.alignment import (
    cra_exclude_mask,
    extract_allatoms,
    position_alignment,
    weighted_rmsd,
)
from LossLab.losses.base import BaseLoss
from rocket import coordinates as rk_coordinates
from rocket import utils as rk_utils
from rocket.cryo import structurefactors as cryo_sf
from rocket.cryo import targets as cryo_targets

logger = logging.getLogger(__name__)


class CryoEMLLGLoss(BaseLoss):
    """Cryo-EM negative-LLG loss matching ROCKET's cryo refinement loop.

    Two ``SFcalculator`` / ``LLGloss`` pairs are maintained — one for the
    main LLG scoring, one for rigid-body refinement — exactly as in
    ``rocket.refinement_cryoem.run_cryoem_refinement``.
    """

    def __init__(
        self,
        input_cif: str | Path,
        input_mtz: str | Path,
        device: torch.device | str = "cuda:0",
        n_bins: int = 20,
        e_label: str = "Emean",
        phie_label: str = "PHIEmean",
        l2_weight: float = 1e-10,
        num_batch: int = 1,
        sub_ratio: float = 1.0,
        sfc_scale: bool = False,
        rbr_lbfgs: bool = True,
        rbr_lbfgs_lr: float = 150.0,
        rbr_verbose: bool = False,
        domain_segs: list[int] | None = None,
        exclude_res: list[int] | None = None,
        fixed_b_factor: float | None = None,
    ) -> None:
        super().__init__(device)
        self.input_cif = str(input_cif)
        self.input_mtz = str(input_mtz)
        self.n_bins = int(n_bins)
        self.e_label = e_label
        self.phie_label = phie_label
        self.l2_weight = float(l2_weight)
        self.num_batch = int(num_batch)
        self.sub_ratio = float(sub_ratio)
        self.sfc_scale = bool(sfc_scale)
        self.rbr_lbfgs = bool(rbr_lbfgs)
        self.rbr_lbfgs_lr = float(rbr_lbfgs_lr)
        self.rbr_verbose = bool(rbr_verbose)
        self.domain_segs = domain_segs
        self.exclude_res = exclude_res
        self.fixed_b_factor = fixed_b_factor

        # --- main SFC + LLGloss (for final scoring) ---
        self.mtz = rk_utils.load_mtz(self.input_mtz)

        # Strip hydrogens: OF3 only predicts heavy atoms, so the SFC
        # topology must exclude H to match the model output.
        pdb_model = self._load_pdb_no_h(self.input_cif)

        self.sfc = cryo_sf.initial_cryoSFC(
            pdb_model, self.mtz,
            self.e_label, self.phie_label,
            self.device, self.n_bins,
        )
        if hasattr(self.sfc, "scales") and self.sfc.scales is not None:
            self.sfc.scales = self.sfc.scales.detach()
        self.target = cryo_targets.LLGloss(self.sfc, self.input_mtz)

        # --- RBR SFC + LLGloss (for rigid-body refinement) ---
        pdb_model_rbr = self._load_pdb_no_h(self.input_cif)
        self.sfc_rbr = cryo_sf.initial_cryoSFC(
            pdb_model_rbr, self.mtz,
            self.e_label, self.phie_label,
            self.device, self.n_bins,
        )
        if hasattr(self.sfc_rbr, "scales") and self.sfc_rbr.scales is not None:
            self.sfc_rbr.scales = self.sfc_rbr.scales.detach()
        self.target_rbr = cryo_targets.LLGloss(self.sfc_rbr, self.input_mtz)

        # Reference pose and B-factor weights
        self.reference_pos = self.sfc.atom_pos_orth.detach().clone()
        self._init_bfactors()

        # Override with fixed B-factor if requested
        if self.fixed_b_factor is not None:
            n_atoms = int(self.sfc.atom_pos_orth.shape[0])
            fixed_Bs = torch.full(
                (n_atoms,), self.fixed_b_factor,
                dtype=torch.float32, device=self.device,
            )
            self.sfc.atom_b_iso = fixed_Bs.clone()
            self.sfc_rbr.atom_b_iso = fixed_Bs.clone()
            self.reference_b_iso = fixed_Bs.clone()
            self.init_pos_bfactor = fixed_Bs.clone()
            logger.info("Using fixed B-factor = %.1f for all atoms", self.fixed_b_factor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pdb_no_h(cif_path: str) -> PDBParser:
        """Load a CIF/PDB, strip hydrogens and ligands/waters before parsing."""
        st = gemmi.read_structure(str(cif_path))
        st.remove_hydrogens()
        st.remove_ligands_and_waters()
        return PDBParser(st)

    def _init_bfactors(self) -> None:
        """Compute initial RSCC-derived B-factors and alignment weights."""
        self.gridsize = self.mtz.get_reciprocal_grid_size(sample_rate=3.0)
        self.rg = torch.tensor(
            rk_utils.g_function_np(2 * self.sfc.dmin, 1 / self.sfc.dHKL),
            device=self.device, dtype=torch.float32,
        )
        self.uc_volume = self.sfc.unit_cell.volume

        dobs_values = self.mtz["Dobs"].to_numpy()
        sf_np = self.mtz.to_structurefactor(
            sf_key=self.e_label, phase_key=self.phie_label,
        ).to_numpy()
        self.rscc_reference_fmap = torch.tensor(
            dobs_values * sf_np, device=self.device, dtype=torch.complex64,
        )

        fprotein = self.sfc.calc_fprotein(Return=True).to(torch.complex64)
        cc_full = rk_utils.get_rscc_from_Fmap(
            fprotein.detach(), self.rscc_reference_fmap,
            self.sfc.HKL_array, self.gridsize, self.rg, self.uc_volume,
        )
        atom_cc_full = rk_utils.interpolate_grid_points(
            cc_full,
            self.sfc.atom_pos_frac.detach().to(torch.float32).cpu().numpy(),
        )
        rscc_b = torch.tensor(
            rk_utils.get_b_from_CC(atom_cc_full, self.sfc.dmin),
            dtype=torch.float32, device=self.device,
        )
        self.sfc.atom_b_iso = rscc_b.detach().clone()
        self.sfc_rbr.atom_b_iso = rscc_b.detach().clone()
        self.reference_b_iso = rscc_b.detach().clone()
        self.init_pos_bfactor = rscc_b.detach().clone()

        cutoff1 = torch.quantile(self.reference_b_iso, 0.30)
        cutoff2 = cutoff1 * 1.5
        bw = rk_utils.weighting_torch(self.reference_b_iso, cutoff1, cutoff2)
        self.bfactor_weights = bw / bw.sum().clamp_min(1e-8)

    def _update_bfactors_rscc(self, aligned_xyz: torch.Tensor) -> None:
        """Recompute RSCC B-factors from aligned coords — updates both SFCs."""
        self.sfc_rbr.atom_pos_orth = aligned_xyz.detach().clone()
        fprotein = self.sfc_rbr.calc_fprotein(Return=True).to(torch.complex64)
        ccmap = rk_utils.get_rscc_from_Fmap(
            fprotein.detach(), self.rscc_reference_fmap,
            self.sfc_rbr.HKL_array, self.gridsize, self.rg, self.uc_volume,
        )
        atom_cc = rk_utils.interpolate_grid_points(
            ccmap,
            self.sfc_rbr.atom_pos_frac.detach().to(torch.float32).cpu().numpy(),
        )
        rscc_b = torch.tensor(
            rk_utils.get_b_from_CC(atom_cc, self.sfc_rbr.dmin),
            dtype=torch.float32, device=self.device,
        )
        self.sfc_rbr.atom_b_iso = rscc_b.detach().clone()
        self.sfc.atom_b_iso = rscc_b.detach().clone()

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def compute_from_rollout(
        self,
        rollout_output: dict,
        batch: dict,
        exclude_res=None,
        domain_segs=None,
        return_metadata: bool = False,
        return_coordinates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss from a diffusion rollout output dict.

        Follows ROCKET ``refinement_cryoem.py`` step-for-step:
        align → pseudo-B → RSCC B-update → RBR → -LLG + L2.
        """
        _exclude = exclude_res or self.exclude_res
        _domain = domain_segs or self.domain_segs

        # 1. Extract atom positions in SFC order (keeps grad)
        x0, _ = extract_allatoms(rollout_output, batch, self.sfc.cra_name)
        x0 = x0.to(self.device)

        # 2. Kabsch align (grad flows through rotation/translation)
        aligned_xyz, plddts_res, pseudo_Bs = position_alignment(
            rollout_output=rollout_output,
            batch=batch,
            best_pos=self.reference_pos,
            exclude_res=_exclude,
            cra_name=self.sfc.cra_name,
            domain_segs=_domain,
            reference_bfactor=self.init_pos_bfactor,
        )
        aligned_xyz = aligned_xyz.to(self.device)
        pseudo_Bs = pseudo_Bs.to(self.device)

        # 3. Set B-factors
        if self.fixed_b_factor is not None:
            n_atoms = int(self.sfc.atom_pos_orth.shape[0])
            fixed_Bs = torch.full(
                (n_atoms,), self.fixed_b_factor,
                dtype=torch.float32, device=self.device,
            )
            self.sfc.atom_b_iso = fixed_Bs.clone()
            self.sfc_rbr.atom_b_iso = fixed_Bs.clone()
        else:
            # Pseudo-B from pLDDT + RSCC update (matches ROCKET)
            self.sfc.atom_b_iso = pseudo_Bs.detach().clone()
            self.sfc_rbr.atom_b_iso = pseudo_Bs.detach().clone()
            self._update_bfactors_rscc(aligned_xyz)

        # 5. Optional SFC scale on RBR SFC (matches ROCKET config.sfc_scale)
        if self.sfc_scale:
            self.sfc_rbr.calc_fprotein()
            self.sfc_rbr.get_scales_adam(lr=0.01, n_steps=10, sub_ratio=0.7, initialize=False)

        # 6. Rigid-body refinement (RBR) via quaternion LBFGS
        optimized_xyz, _ = rk_coordinates.rigidbody_refine_quat(
            aligned_xyz,
            self.target_rbr,
            self.sfc_rbr.cra_name,
            domain_segs=_domain,
            lbfgs=self.rbr_lbfgs,
            lbfgs_lr=self.rbr_lbfgs_lr,
            verbose=self.rbr_verbose,
        )

        # 8. Update main SFC position
        self.sfc.atom_pos_orth = optimized_xyz.detach().clone()

        # 9. Score: -LLG (standard ROCKET LLGloss.forward)
        l_llg = -self.target(
            optimized_xyz.to(torch.float32),
            bin_labels=None,
            num_batch=self.num_batch,
            sub_ratio=self.sub_ratio,
            update_scales=self.sfc_scale,
        )

        # 10. B-factor weighted L2 penalty
        l2_loss = torch.sum(
            self.bfactor_weights.unsqueeze(-1)
            * (optimized_xyz - self.reference_pos) ** 2
        )
        loss = l_llg + self.l2_weight * l2_loss

        if not return_metadata:
            return loss

        # Metadata
        try:
            rmsd_mask = cra_exclude_mask(self.sfc.cra_name, exclude_res=_exclude)
            w = self.bfactor_weights.detach()
            ref = self.reference_pos.detach()
            rmsd_raw = float(weighted_rmsd(x0.detach(), ref, w=w, mask=rmsd_mask))
            rmsd_aligned = float(weighted_rmsd(aligned_xyz.detach(), ref, w=w, mask=rmsd_mask))
            rmsd_rbr = float(weighted_rmsd(optimized_xyz.detach(), ref, w=w, mask=rmsd_mask))
        except Exception:
            rmsd_raw = rmsd_aligned = rmsd_rbr = float("nan")

        md: dict[str, Any] = {
            "mean_plddt": float(plddts_res.mean()),
            "mean_pseudo_b": float(pseudo_Bs.mean()),
            "l_llg": float(l_llg.detach()),
            "l2_loss": float(l2_loss.detach()),
            "loss": float(loss.detach()),
            "rmsd_raw_to_ref": rmsd_raw,
            "rmsd_aligned_to_ref": rmsd_aligned,
            "rmsd_rbr_to_ref": rmsd_rbr,
        }
        if return_coordinates:
            md["coords_raw"] = x0.detach()
            md["coords_aligned"] = aligned_xyz.detach()
            md["coords_rbr"] = optimized_xyz.detach()
            md["cra_name_sfc"] = list(self.sfc.cra_name)
        return loss, md

    def compute(
        self,
        coordinates: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Score raw coordinates directly (no alignment/rollout)."""
        return_metadata = kwargs.pop("return_metadata", False)
        loss = -self.target(coordinates.to(torch.float32), **kwargs)
        if return_metadata:
            return loss, {}
        return loss
