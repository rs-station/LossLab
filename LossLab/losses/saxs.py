"""SAXS Debye scattering losses for coordinate refinement.

Form factor implementation follows the AUSAXS C++ source:
  include/core/form_factor/FormFactor.h
  include/core/form_factor/FormFactorTable.h
  include/core/form_factor/ExvTable.h

FormFactor::evaluate(q):
    f(q) = (Σ_k a_k * exp(-b_k * q²) + c) * q0
where b values are stored in q-space (s-space values / (16π²)) and
q0 is a normalization constant (default 1; or set to effective charge).

AUSAXS AX asymmetry note:
    AUSAXS FFExplicit iterates atom pairs (i,j) with i<j and stores
    2×count in p_aa[type_i][type_j].  The AX cross term multiplies by
    f_atomic(ff1)×f_exv(ff2) which is not symmetric, so each pair gets
    weighted by 2×f_atomic(type_i)×f_exv(type_j) instead
    f_atomic(type_i)×f_exv(type_j) + f_atomic(type_j)×f_exv(type_i).
    Set match_ausaxs=True to replicate this behavior for validation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from LossLab.losses.base import BaseLoss
from LossLab.utils.form_factors import (
    EXV_VOLUMES,
    FORM_FACTOR_COEFFS,
    N_IMPLICIT_H,
    compute_form_factors,
    get_exv_volumes,
)

_FALLBACK_FF = FORM_FACTOR_COEFFS.get("other") or FORM_FACTOR_COEFFS.get("OTH")


def _sinc_debye(coords: torch.Tensor, q_values: torch.Tensor) -> torch.Tensor:
    """Compute sinc(q·r_ij) for all pairs. Returns (Q, N, N)."""
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)        # (N, N, 3)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-30)      # (N, N)
    qr = q_values[:, None, None] * dist[None, :, :]         # (Q, N, N)
    return torch.where(
        qr.abs() < 1e-10,
        torch.ones_like(qr),
        torch.sin(qr) / (qr + 1e-30),
    )


def _sinc_cross(
    coords_a: torch.Tensor, coords_b: torch.Tensor, q_values: torch.Tensor,
) -> torch.Tensor:
    """Compute sinc(q·r_ij) for cross pairs. Returns (Q, Na, Nb)."""
    diff = coords_a[:, None, :] - coords_b[None, :, :]      # (Na, Nb, 3)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-30)      # (Na, Nb)
    qr = q_values[:, None, None] * dist[None, :, :]         # (Q, Na, Nb)
    return torch.where(
        qr.abs() < 1e-10,
        torch.ones_like(qr),
        torch.sin(qr) / (qr + 1e-30),
    )


# ---------------------------------------------------------------------------
# Q-chunked Debye helpers — avoid materialising the full (Q, N, N) tensor
# ---------------------------------------------------------------------------

def _debye_self_chunked(
    coords: torch.Tensor,
    q_values: torch.Tensor,
    ff: torch.Tensor,
    q_chunk_size: int,
) -> torch.Tensor:
    """Chunked I(q) = Σ_ij ff_i(q)·ff_j(q)·sinc(q·r_ij). Returns (Q,).

    Iterates over *q* in chunks so peak memory is O(chunk·N²) instead of
    O(Q·N²).  Uses torch.cat (not in-place assignment) so autograd is safe.
    """
    # Pairwise distances — materialised once, shape (N, N)
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)        # (N, N, 3)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-30)      # (N, N)
    Q = q_values.shape[0]
    chunks = []
    for start in range(0, Q, q_chunk_size):
        end = min(start + q_chunk_size, Q)
        q_c = q_values[start:end]                            # (C,)
        qr = q_c[:, None, None] * dist[None, :, :]          # (C, N, N)
        sinc = torch.where(
            qr.abs() < 1e-10,
            torch.ones_like(qr),
            torch.sin(qr) / (qr + 1e-30),
        )
        ff_c = ff[start:end]                                 # (C, N)
        ww = ff_c[:, :, None] * ff_c[:, None, :]            # (C, N, N)
        chunks.append((ww * sinc).sum(dim=(-2, -1)))         # (C,)
    return torch.cat(chunks, dim=0)


def _debye_cross_chunked(
    coords_a: torch.Tensor,
    coords_b: torch.Tensor,
    q_values: torch.Tensor,
    ff_a: torch.Tensor,
    ff_b: torch.Tensor,
    q_chunk_size: int,
) -> torch.Tensor:
    """Chunked cross-term I(q) = Σ_ij ff_a_i(q)·ff_b_j(q)·sinc(q·r_ij).

    Uses torch.cat (not in-place assignment) so autograd is safe.
    """
    diff = coords_a[:, None, :] - coords_b[None, :, :]      # (Na, Nb, 3)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-30)      # (Na, Nb)
    Q = q_values.shape[0]
    chunks = []
    for start in range(0, Q, q_chunk_size):
        end = min(start + q_chunk_size, Q)
        q_c = q_values[start:end]
        qr = q_c[:, None, None] * dist[None, :, :]          # (C, Na, Nb)
        sinc = torch.where(
            qr.abs() < 1e-10,
            torch.ones_like(qr),
            torch.sin(qr) / (qr + 1e-30),
        )
        ww = ff_a[start:end, :, None] * ff_b[start:end, None, :]
        chunks.append((ww * sinc).sum(dim=(-2, -1)))         # (C,)
    return torch.cat(chunks, dim=0)


def debye_intensity(
    coords: torch.Tensor,
    ff_types: list[str],
    q_values: torch.Tensor,
    *,
    use_exv: bool = True,
    cx: torch.Tensor | float = 1.0,
    rho_water: float = 0.334,
) -> torch.Tensor:
    """Differentiable Debye scattering intensity.

    Args:
        coords: (N, 3) atom coordinates (requires_grad OK).
        ff_types: N form-factor type strings.
        q_values: (Q,) q-values in Å⁻¹.
        use_exv: whether to apply Fraser excluded-volume correction.
        cx: excluded-volume scale factor (can be nn.Parameter).
        rho_water: bulk solvent electron density.

    Returns:
        (Q,) scattering intensity I(q).
    """
    ff = compute_form_factors(ff_types, q_values, effective_charge=not use_exv)

    if use_exv:
        vols = get_exv_volumes(ff_types, q_values.device, q_values.dtype)
        v_23 = vols ** (2.0 / 3.0)
        exv_exp = -v_23[None, :] / (4.0 * math.pi) * (q_values[:, None] ** 2)
        ff_exv = vols[None, :] * rho_water * torch.exp(exv_exp)
        ff = ff - cx * ff_exv

    sinc = _sinc_debye(coords, q_values)
    ww = ff[:, :, None] * ff[:, None, :]
    return (ww * sinc).sum(dim=(-2, -1))


def debye_hydration_intensity(
    coords_atom: torch.Tensor,
    ff_types_atom: list[str],
    coords_water: torch.Tensor,
    ff_types_water: list[str],
    q_values: torch.Tensor,
    *,
    cx: torch.Tensor | float = 1.0,
    cw: torch.Tensor | float = 1.0,
    rho_water: float = 0.334,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable Debye intensity with hydration shell.

    Computes:
        I(q) = I_aa(cx) + 2·cw·I_aw(cx) + cw²·I_ww

    where I_aa uses the Fraser ExV-corrected atom form factors,
    I_aw is the atom-water cross term, and I_ww is the water self term.

    Args:
        coords_atom: (Na, 3) protein atom coordinates.
        ff_types_atom: Na form-factor type strings.
        coords_water: (Nw, 3) hydration shell coordinates.
        ff_types_water: Nw form-factor types (typically all "OH").
        q_values: (Q,) q-values in Å⁻¹.
        cx: excluded-volume scale factor.
        cw: hydration shell contrast parameter.
        rho_water: bulk solvent electron density.

    Returns:
        I_total: (Q,) total scattering intensity.
        components: dict with keys "I_aa", "I_aw", "I_ww".
    """
    # Atom form factors with ExV correction
    ff_atom = compute_form_factors(ff_types_atom, q_values, effective_charge=False)
    vols = get_exv_volumes(ff_types_atom, q_values.device, q_values.dtype)
    v_23 = vols ** (2.0 / 3.0)
    exv_exp = -v_23[None, :] / (4.0 * math.pi) * (q_values[:, None] ** 2)
    ff_exv = vols[None, :] * rho_water * torch.exp(exv_exp)
    ff_atom_eff = ff_atom - cx * ff_exv                              # (Q, Na)

    # Water form factors (no ExV — shell is excess over bulk)
    ff_water = compute_form_factors(
        ff_types_water, q_values, effective_charge=False,
    )                                                                # (Q, Nw)

    # I_aa: atom-atom
    sinc_aa = _sinc_debye(coords_atom, q_values)                    # (Q, Na, Na)
    ww_aa = ff_atom_eff[:, :, None] * ff_atom_eff[:, None, :]
    I_aa = (ww_aa * sinc_aa).sum(dim=(-2, -1))                      # (Q,)

    # I_aw: atom-water cross term
    sinc_aw = _sinc_cross(coords_atom, coords_water, q_values)      # (Q, Na, Nw)
    ww_aw = ff_atom_eff[:, :, None] * ff_water[:, None, :]
    I_aw = (ww_aw * sinc_aw).sum(dim=(-2, -1))                      # (Q,)

    # I_ww: water-water
    sinc_ww = _sinc_debye(coords_water, q_values)                   # (Q, Nw, Nw)
    ww_ww = ff_water[:, :, None] * ff_water[:, None, :]
    I_ww = (ww_ww * sinc_ww).sum(dim=(-2, -1))                      # (Q,)

    I_total = I_aa + 2 * cw * I_aw + cw ** 2 * I_ww
    return I_total, {"I_aa": I_aa, "I_aw": I_aw, "I_ww": I_ww}


def load_saxs_data(
    path: str | Path,
    device: torch.device | str = "cpu",
    q_min: float | None = None,
    q_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Load experimental SAXS profile from a 2- or 3-column file.

    Accepts whitespace- or comma-delimited files with columns:
        q  I(q)  [sigma(q)]

    Lines starting with '#' are skipped.

    Returns:
        q_exp: (Q,) q-values.
        I_exp: (Q,) experimental intensities.
        sigma_exp: (Q,) errors, or None if not present.
    """
    path = Path(path)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(x) for x in parts[:3]])
            except ValueError:
                continue

    data = np.array(rows)
    q_exp = data[:, 0]
    I_exp = data[:, 1]
    sigma_exp = data[:, 2] if data.shape[1] >= 3 else None

    mask = np.ones(len(q_exp), dtype=bool)
    if q_min is not None:
        mask &= q_exp >= q_min
    if q_max is not None:
        mask &= q_exp <= q_max
    q_exp = q_exp[mask]
    I_exp = I_exp[mask]
    if sigma_exp is not None:
        sigma_exp = sigma_exp[mask]

    q_t = torch.tensor(q_exp, dtype=torch.float32, device=device)
    I_t = torch.tensor(I_exp, dtype=torch.float32, device=device)
    s_t = (
        torch.tensor(sigma_exp, dtype=torch.float32, device=device)
        if sigma_exp is not None
        else None
    )
    return q_t, I_t, s_t


class DebyeRawLoss(BaseLoss):
    """SAXS loss using the raw Debye equation with effective-charge weights.

    Computes I(q) = Σ_ij w_i w_j sin(q·r_ij)/(q·r_ij) where
    w_i = f_i(0) + n_implicit_H_i  (the effective charge).

    Matches pyAUSAXS ``mol.debye_raw()`` when ff_types are provided.
    Falls back to uniform weights (w=1) if ff_types is None.

    Args:
        q_values: (Q,) scattering vector magnitudes in Å⁻¹
        I_target: (Q,) target scattering intensities
        ff_types: list of N form factor type strings; None for uniform weights
        sigma: (Q,) per-point uncertainties; if None uses uniform weights
        device: PyTorch device
        scale_invariant: if True, fit a free log-scale offset before computing χ²
    """

    def __init__(
        self,
        q_values: torch.Tensor | np.ndarray,
        I_target: torch.Tensor | np.ndarray,
        ff_types: list[str] | None = None,
        sigma: torch.Tensor | np.ndarray | None = None,
        device: torch.device | str = "cuda:0",
        scale_invariant: bool = False,
    ) -> None:
        super().__init__(device)
        self.q_values = torch.as_tensor(q_values, dtype=torch.float64).to(self.device)
        self.I_target = torch.as_tensor(I_target, dtype=torch.float64).to(self.device)
        self.sigma = (
            torch.as_tensor(sigma, dtype=torch.float64).to(self.device)
            if sigma is not None else None
        )
        self.scale_invariant = scale_invariant

        # Effective-charge weights: w_i = f_i(0) + n_implicit_H_i
        if ff_types is not None:
            w = []
            for t in ff_types:
                entry = FORM_FACTOR_COEFFS.get(t, _FALLBACK_FF)
                f0 = sum(entry["a"]) + entry["c"]
                n_h = N_IMPLICIT_H.get(t, 0)
                w.append(f0 + n_h)
            self._weights = torch.tensor(w, dtype=torch.float64, device=self.device)
        else:
            self._weights = None

    def _I_pred(self, coordinates: torch.Tensor) -> torch.Tensor:
        sinc = _sinc_debye(coordinates.to(self.device), self.q_values)  # (Q, N, N)
        if self._weights is not None:
            ww = self._weights[:, None] * self._weights[None, :]       # (N, N)
            return (ww[None, :, :] * sinc).sum(dim=(-2, -1))           # (Q,)
        return sinc.sum(dim=(-2, -1))

    def compute(
        self,
        coordinates: torch.Tensor,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        I_pred = self._I_pred(coordinates)

        log_pred = torch.log(I_pred.clamp(min=1e-30))
        log_tgt  = torch.log(self.I_target.clamp(min=1e-30))

        if self.scale_invariant:
            log_pred = log_pred + (log_tgt - log_pred).mean().detach()

        residuals = log_pred - log_tgt
        if self.sigma is not None:
            residuals = residuals / (self.sigma + 1e-30)

        loss = (residuals ** 2).mean()

        if return_metadata:
            return loss, {"chi2_log": loss.item()}
        return loss

    def to(self, device: torch.device | str) -> "DebyeRawLoss":
        super().to(device)
        self.q_values = self.q_values.to(self.device)
        self.I_target = self.I_target.to(self.device)
        if self.sigma is not None:
            self.sigma = self.sigma.to(self.device)
        if self._weights is not None:
            self._weights = self._weights.to(self.device)
        return self


# ---------------------------------------------------------------------------
# Shared chi2 computation helpers
# ---------------------------------------------------------------------------

def _linear_chi2(
    I_pred: torch.Tensor,
    I_target: torch.Tensor,
    sigma: torch.Tensor | None,
    detach_scale: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Standard SAXS chi2 in linear space with optimal scale factor.

    chi2 = (1/N) Σ_i ((I_exp_i - c * I_calc_i) / σ_i)²

    The optimal scale factor c is computed analytically:
        c = Σ(I_exp * I_calc / σ²) / Σ(I_calc² / σ²)

    When ``detach_scale=True``, gradients do not flow through c.  This
    prevents the optimizer from gaming the scale factor — it can only
    reduce chi2 by making I_calc/c closer to I_exp.  This mirrors
    standard SAXS refinement practice (CRYSOL, FoXS) where c is re-fitted
    at each step but not differentiated through.
    """
    if sigma is not None:
        w = 1.0 / (sigma ** 2 + 1e-30)  # inverse variance weights
    else:
        w = torch.ones_like(I_target)

    # Optimal scale: c = (I_exp · I_calc · w) / (I_calc² · w)
    c = (I_target * I_pred * w).sum() / ((I_pred ** 2 * w).sum() + 1e-30)

    if detach_scale:
        c = c.detach()

    residuals = (I_target - c * I_pred) * torch.sqrt(w)
    chi2 = (residuals ** 2).mean()

    return chi2, {
        "chi2": chi2.item(),
        "scale_factor": c.item(),
    }


def _log_chi2(
    I_pred: torch.Tensor,
    I_target: torch.Tensor,
    sigma: torch.Tensor | None,
    scale_invariant: bool,
) -> tuple[torch.Tensor, dict]:
    """Log-space chi2 (original LossLab metric).

    *sigma* is expected in **log-space** units (σ_log = σ/I).
    """
    log_pred = torch.log(I_pred.clamp(min=1e-30))
    log_tgt = torch.log(I_target.clamp(min=1e-30))

    if scale_invariant:
        log_pred = log_pred + (log_tgt - log_pred).mean().detach()

    residuals = log_pred - log_tgt
    if sigma is not None:
        residuals = residuals / (sigma + 1e-30)

    loss = (residuals ** 2).mean()
    return loss, {"chi2_log": loss.item()}


class DebyeLoss(BaseLoss):
    """SAXS loss using the full Debye equation with q-dependent form factors.

    Computes:
        I(q) = Σ_ij f_i^eff(q) f_j^eff(q) sin(q·r_ij)/(q·r_ij)

    where f_i(q) uses the 5-Gaussian parameterisation from FormFactorTable.h:
        f(q) = (Σ_k a_k exp(-b_k q²) + c) * q0
    with b_k in q-space (= s-space value / 16π²) and q0 set to the effective
    charge when ``use_exv=False`` or q0=1 when ``use_exv=True``.

    The Fraser excluded volume correction (ExvTable.h):
        f_i^exv(q) = V_i · ρ₀ · exp(-V_i^(2/3) / (4π) · q²)
    is applied when ``use_exv=True``, giving f_i^eff = f_i - f_i^exv.

    Differentiable equivalent of pyAUSAXS ``mol.debye()``.

    Args:
        q_values: (Q,) scattering vector magnitudes in Å⁻¹
        I_target: (Q,) target scattering intensities
        ff_types: list of N form factor type strings (e.g. ["C", "N", "OH"])
        sigma: (Q,) per-point uncertainties; if None uses uniform weights
        use_exv: if True apply Fraser excluded volume correction (default True)
        match_ausaxs: if True replicate the AUSAXS AX histogram asymmetry
            (see module docstring); default False uses the correct symmetric
            formula
        rho_water: solvent electron density in e/Å³ (default 0.334)
        device: PyTorch device
        scale_invariant: if True, fit a free log-scale offset before computing χ²
        chi2_mode: ``"log"`` (default) for log-space chi2, or ``"linear"``
            for the standard SAXS chi2 = (1/N) Σ ((I_exp - c·I_calc)/σ)²
            with analytically-optimal scale factor c.  The ``"linear"`` mode
            matches pyAUSAXS ``mol.fit()`` and is the accepted SAXS metric.
    """

    def __init__(
        self,
        q_values: torch.Tensor | np.ndarray,
        I_target: torch.Tensor | np.ndarray,
        ff_types: list[str],
        sigma: torch.Tensor | np.ndarray | None = None,
        use_exv: bool = True,
        match_ausaxs: bool = False,
        rho_water: float = 0.334,
        device: torch.device | str = "cuda:0",
        scale_invariant: bool = False,
        q_chunk_size: int | None = None,
        chi2_mode: str = "log",
        detach_scale: bool = False,
    ) -> None:
        super().__init__(device)
        self.q_values = torch.as_tensor(q_values, dtype=torch.float64).to(self.device)
        self.I_target = torch.as_tensor(I_target, dtype=torch.float64).to(self.device)
        self.sigma = (
            torch.as_tensor(sigma, dtype=torch.float64).to(self.device)
            if sigma is not None else None
        )
        self.ff_types = list(ff_types)
        self.use_exv = use_exv
        self.match_ausaxs = match_ausaxs
        self.rho_water = rho_water
        self.scale_invariant = scale_invariant
        self.q_chunk_size = q_chunk_size
        if chi2_mode not in ("log", "linear"):
            raise ValueError(f"chi2_mode must be 'log' or 'linear', got {chi2_mode!r}")
        self.chi2_mode = chi2_mode
        self.detach_scale = detach_scale

        if use_exv:
            self._exv_volumes = get_exv_volumes(
                self.ff_types, self.device, torch.float64
            )
        else:
            self._exv_volumes = None

    def _I_pred(self, coordinates: torch.Tensor) -> torch.Tensor:
        coords = coordinates.to(self.device)
        N = coords.shape[0]

        # effective_charge=False for Fraser (raw 5-Gaussian);
        # effective_charge=True for no-ExV mode
        effective_charge = not self.use_exv
        ff = compute_form_factors(self.ff_types, self.q_values, effective_charge)  # (Q, N)

        if not self.use_exv:
            if self.q_chunk_size is not None:
                return _debye_self_chunked(coords, self.q_values, ff, self.q_chunk_size)
            sinc = _sinc_debye(coords, self.q_values)                    # (Q, N, N)
            ww = ff[:, :, None] * ff[:, None, :]                         # (Q, N, N)
            return (ww * sinc).sum(dim=(-2, -1))                         # (Q,)

        V = self._exv_volumes                                            # (N,)
        V_23 = V ** (2.0 / 3.0)
        exv_exp = -V_23[None, :] / (4.0 * math.pi) * (self.q_values[:, None] ** 2)
        ff_exv = V[None, :] * self.rho_water * torch.exp(exv_exp)       # (Q, N)

        if not self.match_ausaxs:
            # Correct symmetric formula: I = Σ (f-f_exv)_i (f-f_exv)_j sinc
            ff_eff = ff - ff_exv
            if self.q_chunk_size is not None:
                return _debye_self_chunked(coords, self.q_values, ff_eff, self.q_chunk_size)
            sinc = _sinc_debye(coords, self.q_values)                    # (Q, N, N)
            ww = ff_eff[:, :, None] * ff_eff[:, None, :]                 # (Q, N, N)
            return (ww * sinc).sum(dim=(-2, -1))                         # (Q,)

        # AUSAXS-matching formula: I = AA - AX_asym + XX
        # AA and XX use symmetric products (correct).
        # AX uses the asymmetric product from the i<j histogram ordering.
        sinc = _sinc_debye(coords, self.q_values)                       # (Q, N, N)
        AA = (ff[:, :, None] * ff[:, None, :] * sinc).sum(dim=(-2, -1))

        # XX (symmetric)
        XX = (ff_exv[:, :, None] * ff_exv[:, None, :] * sinc).sum(dim=(-2, -1))

        # AX (asymmetric): for i<j pairs, weight is 2*f_atom_i*f_exv_j
        # Equivalent to: 4 * Σ_{i<j} f_atom_i * f_exv_j * sinc_ij
        triu_mask = torch.triu(torch.ones(N, N, device=self.device, dtype=sinc.dtype), diagonal=1)
        ax_upper = ff[:, :, None] * ff_exv[:, None, :] * sinc * triu_mask[None, :, :]
        AX = 4 * ax_upper.sum(dim=(-2, -1))

        return AA - AX + XX

    def compute(
        self,
        coordinates: torch.Tensor,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        I_pred = self._I_pred(coordinates)
        self._last_I_pred = I_pred.detach()

        if self.chi2_mode == "linear":
            loss, meta = _linear_chi2(I_pred, self.I_target, self.sigma, self.detach_scale)
        else:
            loss, meta = _log_chi2(
                I_pred, self.I_target, self.sigma, self.scale_invariant,
            )

        if return_metadata:
            return loss, meta
        return loss

    def to(self, device: torch.device | str) -> "DebyeLoss":
        super().to(device)
        self.q_values = self.q_values.to(self.device)
        self.I_target = self.I_target.to(self.device)
        if self.sigma is not None:
            self.sigma = self.sigma.to(self.device)
        if self._exv_volumes is not None:
            self._exv_volumes = self._exv_volumes.to(self.device)
        return self


class DebyeHydrationLoss(BaseLoss):
    """SAXS loss with hydration shell contribution.

    Computes the full three-component Debye intensity:

        I(q) = I_aa(cx) + 2·cw·I_aw(cx) + cw²·I_ww

    Water positions are generated externally (e.g. by pyAUSAXS ``hydrate()``)
    and can operate in two modes:

    - ``"fixed"``: water coords are constants, no gradient through them.
      Gradient still flows through I_aa and the atom side of I_aw.
    - ``"attached"``: each water is anchored to its nearest protein atom
      via a frozen offset vector.  When atom i moves, its waters move
      with it, giving additional gradient flow through the water positions.

    Args:
        q_values: (Q,) scattering vector magnitudes in Å⁻¹
        I_target: (Q,) target scattering intensities
        ff_types_atom: list of Na form factor type strings
        sigma: (Q,) per-point uncertainties; None for uniform weights
        cx: excluded-volume scale; set ``learn_cx=True`` to optimise
        cw: hydration shell contrast; set ``learn_cw=True`` to optimise
        learn_cx: if True, cx becomes a learnable parameter
        learn_cw: if True, cw becomes a learnable parameter
        water_mode: ``"fixed"`` or ``"attached"``
        rho_water: solvent electron density in e/Å³
        device: PyTorch device
        scale_invariant: fit a log-scale offset before computing χ²
        q_chunk_size: if set, iterate over q in chunks of this size to
            reduce peak GPU memory from O(Q·N²) to O(chunk·N²)
        chi2_mode: ``"log"`` (default) or ``"linear"`` — see DebyeLoss.
    """

    def __init__(
        self,
        q_values: torch.Tensor | np.ndarray,
        I_target: torch.Tensor | np.ndarray,
        ff_types_atom: list[str],
        sigma: torch.Tensor | np.ndarray | None = None,
        cx: float = 1.0,
        cw: float = 1.0,
        learn_cx: bool = False,
        learn_cw: bool = False,
        water_mode: str = "fixed",
        rho_water: float = 0.334,
        device: torch.device | str = "cuda:0",
        scale_invariant: bool = False,
        q_chunk_size: int | None = None,
        chi2_mode: str = "log",
        detach_scale: bool = False,
    ) -> None:
        super().__init__(device)
        self.q_values = torch.as_tensor(q_values, dtype=torch.float64).to(self.device)
        self.I_target = torch.as_tensor(I_target, dtype=torch.float64).to(self.device)
        self.sigma = (
            torch.as_tensor(sigma, dtype=torch.float64).to(self.device)
            if sigma is not None else None
        )
        self.ff_types_atom = list(ff_types_atom)
        self.rho_water = rho_water
        self.scale_invariant = scale_invariant
        self.water_mode = water_mode
        self.q_chunk_size = q_chunk_size
        if chi2_mode not in ("log", "linear"):
            raise ValueError(f"chi2_mode must be 'log' or 'linear', got {chi2_mode!r}")
        self.chi2_mode = chi2_mode
        self.detach_scale = detach_scale

        # Learnable or fixed scaling parameters
        if learn_cx:
            self.cx = torch.nn.Parameter(
                torch.tensor(cx, dtype=torch.float64, device=self.device)
            )
        else:
            self.cx = torch.tensor(cx, dtype=torch.float64, device=self.device)

        if learn_cw:
            self.cw = torch.nn.Parameter(
                torch.tensor(cw, dtype=torch.float64, device=self.device)
            )
        else:
            self.cw = torch.tensor(cw, dtype=torch.float64, device=self.device)

        # ExV volumes for atoms (precomputed)
        self._exv_volumes = get_exv_volumes(
            self.ff_types_atom, self.device, torch.float64
        )

        # Water state (populated by set_water)
        self._coords_water: torch.Tensor | None = None
        self._ff_types_water: list[str] | None = None
        self._parent_idx: torch.Tensor | None = None
        self._offsets: torch.Tensor | None = None
        self._I_ww_cached: torch.Tensor | None = None

    def set_water(
        self,
        coords_water: torch.Tensor | np.ndarray,
        coords_atom_ref: torch.Tensor | np.ndarray | None = None,
        ff_types_water: list[str] | None = None,
    ) -> None:
        """Set hydration shell positions.

        Args:
            coords_water: (Nw, 3) water coordinates.
            coords_atom_ref: (Na, 3) atom coords at the time of hydration.
                Required for ``water_mode="attached"`` to compute offsets.
            ff_types_water: Nw type strings; defaults to all "OH".
        """
        cw_t = torch.as_tensor(coords_water, dtype=torch.float64).to(self.device)
        Nw = cw_t.shape[0]
        self._ff_types_water = ff_types_water or (["OH"] * Nw)

        if self.water_mode == "attached":
            if coords_atom_ref is None:
                raise ValueError(
                    "coords_atom_ref is required for water_mode='attached'"
                )
            ca_ref = torch.as_tensor(
                coords_atom_ref, dtype=torch.float64,
            ).to(self.device)
            # Assign each water to its nearest atom
            dists = torch.cdist(cw_t, ca_ref)                       # (Nw, Na)
            self._parent_idx = dists.argmin(dim=1)                   # (Nw,)
            self._offsets = cw_t - ca_ref[self._parent_idx]          # (Nw, 3)
            self._coords_water = None  # derived at forward time
        else:
            self._coords_water = cw_t
            self._parent_idx = None
            self._offsets = None

        # Invalidate cached I_ww
        self._I_ww_cached = None

    def _get_water_coords(self, coords_atom: torch.Tensor) -> torch.Tensor:
        """Return water coords — fixed or derived from atom positions."""
        if self.water_mode == "attached":
            return coords_atom[self._parent_idx] + self._offsets
        return self._coords_water

    def _I_pred(
        self, coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coords_a = coordinates.to(self.device)
        coords_w = self._get_water_coords(coords_a)

        # Atom effective form factors (with ExV)
        ff_atom = compute_form_factors(
            self.ff_types_atom, self.q_values, effective_charge=False,
        )
        V = self._exv_volumes
        V_23 = V ** (2.0 / 3.0)
        exv_exp = -V_23[None, :] / (4.0 * math.pi) * (self.q_values[:, None] ** 2)
        ff_exv = V[None, :] * self.rho_water * torch.exp(exv_exp)
        ff_eff = ff_atom - self.cx * ff_exv                          # (Q, Na)

        # Water form factors (no ExV)
        ff_w = compute_form_factors(
            self._ff_types_water, self.q_values, effective_charge=False,
        )                                                            # (Q, Nw)

        chunk = self.q_chunk_size

        # I_aa
        if chunk is not None:
            I_aa = _debye_self_chunked(coords_a, self.q_values, ff_eff, chunk)
        else:
            sinc_aa = _sinc_debye(coords_a, self.q_values)
            I_aa = (ff_eff[:, :, None] * ff_eff[:, None, :] * sinc_aa).sum(dim=(-2, -1))

        # I_aw (cross)
        if chunk is not None:
            I_aw = _debye_cross_chunked(coords_a, coords_w, self.q_values, ff_eff, ff_w, chunk)
        else:
            sinc_aw = _sinc_cross(coords_a, coords_w, self.q_values)
            I_aw = (ff_eff[:, :, None] * ff_w[:, None, :] * sinc_aw).sum(dim=(-2, -1))

        # I_ww (cache when water is fixed — it's constant)
        if self.water_mode == "fixed" and self._I_ww_cached is not None:
            I_ww = self._I_ww_cached
        else:
            if chunk is not None:
                I_ww = _debye_self_chunked(coords_w, self.q_values, ff_w, chunk)
            else:
                sinc_ww = _sinc_debye(coords_w, self.q_values)
                I_ww = (ff_w[:, :, None] * ff_w[:, None, :] * sinc_ww).sum(dim=(-2, -1))
            if self.water_mode == "fixed":
                self._I_ww_cached = I_ww.detach()

        I_total = I_aa + 2 * self.cw * I_aw + self.cw ** 2 * I_ww
        return I_total, {"I_aa": I_aa, "I_aw": I_aw, "I_ww": I_ww}

    def compute(
        self,
        coordinates: torch.Tensor,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if self._coords_water is None and self._offsets is None:
            raise RuntimeError("Call set_water() before compute()")

        I_pred, components = self._I_pred(coordinates)
        self._last_I_pred = I_pred.detach()

        if self.chi2_mode == "linear":
            loss, meta = _linear_chi2(I_pred, self.I_target, self.sigma, self.detach_scale)
        else:
            loss, meta = _log_chi2(
                I_pred, self.I_target, self.sigma, self.scale_invariant,
            )

        if return_metadata:
            meta["I_aa_frac"] = (
                components["I_aa"].abs().sum() / I_pred.abs().sum()
            ).item()
            return loss, meta
        return loss

    def parameters(self):
        """Yield learnable parameters (cx, cw if set to learn)."""
        if isinstance(self.cx, torch.nn.Parameter):
            yield self.cx
        if isinstance(self.cw, torch.nn.Parameter):
            yield self.cw

    def to(self, device: torch.device | str) -> "DebyeHydrationLoss":
        super().to(device)
        self.q_values = self.q_values.to(self.device)
        self.I_target = self.I_target.to(self.device)
        if self.sigma is not None:
            self.sigma = self.sigma.to(self.device)
        self._exv_volumes = self._exv_volumes.to(self.device)
        if not isinstance(self.cx, torch.nn.Parameter):
            self.cx = self.cx.to(self.device)
        if not isinstance(self.cw, torch.nn.Parameter):
            self.cw = self.cw.to(self.device)
        if self._coords_water is not None:
            self._coords_water = self._coords_water.to(self.device)
        if self._parent_idx is not None:
            self._parent_idx = self._parent_idx.to(self.device)
        if self._offsets is not None:
            self._offsets = self._offsets.to(self.device)
        if self._I_ww_cached is not None:
            self._I_ww_cached = self._I_ww_cached.to(self.device)
        return self
