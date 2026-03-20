"""SAXS Debye scattering losses for coordinate refinement.

Form factor implementation follows the AUSAXS C++ source:
  include/core/form_factor/FormFactor.h
  include/core/form_factor/FormFactorTable.h
  include/core/form_factor/ExvTable.h

FormFactor::evaluate(q):
    f(q) = (Σ_k a_k * exp(-b_k * q²) + c) * q0
where b values are stored in q-space (s-space values / (16π²)) and
q0 is a normalization constant (default 1; or set to effective charge).
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
            rows.append([float(x) for x in parts[:3]])

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
        rho_water: solvent electron density in e/Å³ (default 0.334)
        device: PyTorch device
        scale_invariant: if True, fit a free log-scale offset before computing χ²
    """

    def __init__(
        self,
        q_values: torch.Tensor | np.ndarray,
        I_target: torch.Tensor | np.ndarray,
        ff_types: list[str],
        sigma: torch.Tensor | np.ndarray | None = None,
        use_exv: bool = True,
        rho_water: float = 0.334,
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
        self.ff_types = list(ff_types)
        self.use_exv = use_exv
        self.rho_water = rho_water
        self.scale_invariant = scale_invariant

        if use_exv:
            self._exv_volumes = get_exv_volumes(
                self.ff_types, self.device, torch.float64
            )
        else:
            self._exv_volumes = None

    def _I_pred(self, coordinates: torch.Tensor) -> torch.Tensor:
        coords = coordinates.to(self.device)

        # effective_charge=False for Fraser (raw 5-Gaussian);
        # effective_charge=True for no-ExV mode
        effective_charge = not self.use_exv
        ff = compute_form_factors(self.ff_types, self.q_values, effective_charge)  # (Q, N)

        if self.use_exv:
            V = self._exv_volumes                                         # (N,)
            V_23 = V ** (2.0 / 3.0)
            exv_exp = -V_23[None, :] / (4.0 * math.pi) * (self.q_values[:, None] ** 2)
            ff_exv = V[None, :] * self.rho_water * torch.exp(exv_exp)    # (Q, N)
            ff_eff = ff - ff_exv
        else:
            ff_eff = ff

        sinc = _sinc_debye(coords, self.q_values)                        # (Q, N, N)
        ww = ff_eff[:, :, None] * ff_eff[:, None, :]                     # (Q, N, N)
        return (ww * sinc).sum(dim=(-2, -1))                             # (Q,)

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

    def to(self, device: torch.device | str) -> "DebyeLoss":
        super().to(device)
        self.q_values = self.q_values.to(self.device)
        self.I_target = self.I_target.to(self.device)
        if self.sigma is not None:
            self.sigma = self.sigma.to(self.device)
        if self._exv_volumes is not None:
            self._exv_volumes = self._exv_volumes.to(self.device)
        return self
