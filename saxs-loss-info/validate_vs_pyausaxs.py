"""Validate LossLab SAXS losses against pyAUSAXS on the NFU1 protein (SASDPA8).

Generates comparison plots for:
  1. Atom-only (no ExV): LossLab debye_intensity vs pyAUSAXS mol.debye(ExvModel.none)
  2. With hydration (no ExV): manual Debye+water vs pyAUSAXS mol.debye(ExvModel.none, hydrated)
  3. Atom-only (Fraser ExV): LossLab DebyeLoss vs pyAUSAXS mol.debye(ExvModel.fraser)
     — shows the AX asymmetry bug effect
  4. Hydrated vs experimental SASDPA8.dat: LossLab +water vs pyAUSAXS fit
"""

import os, sys, math
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyausaxs as ausaxs
from pyausaxs.wrapper.Models import ExvModel
from pyausaxs.wrapper.settings import settings

from LossLab.losses.saxs import (
    debye_intensity,
    DebyeLoss,
    DebyeHydrationLoss,
    _sinc_debye,
    _sinc_cross,
)
from LossLab.utils.form_factors import compute_form_factors

# ── Config ──────────────────────────────────────────────────────────────────
GT_PDB = "/pscratch/sd/c/ckalicki/cryo-of3/of3-compass/apo_nfu1/SASDPA8_fit2_model1.pdb"
SAXS_FILE = "/pscratch/sd/c/ckalicki/cryo-of3/of3-compass/apo_nfu1/SASDPA8.dat"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

Q_GRID = np.linspace(0.01, 0.5, 200)


def load_pyausaxs(exv_model, hydrate=False):
    """Load molecule with pyAUSAXS, return types/coords/waters and I(q)."""
    # Hydrate first with default ExV, then set the desired ExV for Debye
    mol = ausaxs.create_molecule(GT_PDB)
    if hydrate:
        mol.hydrate()
    else:
        mol.clear_hydration()
    settings.exv(exv_model)
    a = mol.atoms()
    types = [str(t) for t in a[4]]
    coords = np.column_stack([a[0], a[1], a[2]])
    q, I = mol.debye(Q_GRID.tolist())
    waters = None
    if hydrate:
        wx, wy, wz, _ = mol.waters()
        waters = np.stack([wx, wy, wz], axis=1)
    return types, coords, waters, np.array(q), np.array(I)


def load_experimental():
    """Load SASDPA8.dat and return q (A^-1), I, sigma."""
    rows = []
    with open(SAXS_FILE) as f:
        for line in f:
            parts = line.strip().split()
            try:
                rows.append([float(x) for x in parts[:3]])
            except ValueError:
                continue
    data = np.array(rows)
    q = data[:, 0] / 10.0  # nm^-1 -> A^-1
    mask = q <= 0.5
    return q[mask], data[mask, 1], data[mask, 2]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Atom-only, no ExV
# ══════════════════════════════════════════════════════════════════════════════
print("Test 1: Atom-only, no ExV ...")
types_none, coords_none, _, q_none, I_pya_none = load_pyausaxs(ExvModel.none, hydrate=False)
coords_t = torch.tensor(coords_none, dtype=torch.float64)
q_t = torch.tensor(q_none, dtype=torch.float64)

with torch.no_grad():
    I_ll_none = debye_intensity(coords_t, types_none, q_t, use_exv=False).numpy()

dev_none = np.abs(I_ll_none / I_pya_none - 1)
print(f"  I(0) ratio: {I_ll_none[0]/I_pya_none[0]:.6f}")
print(f"  max|dev|: {dev_none.max():.2e}  mean: {dev_none.mean():.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. With hydration, no ExV
# ══════════════════════════════════════════════════════════════════════════════
print("\nTest 2: With hydration, no ExV ...")
types_h, coords_h, waters_h, q_h, I_pya_hyd = load_pyausaxs(ExvModel.none, hydrate=True)
n_w = len(waters_h)
print(f"  {n_w} waters")

coords_a = torch.tensor(coords_h, dtype=torch.float64)
coords_w = torch.tensor(waters_h, dtype=torch.float64)

with torch.no_grad():
    # Atom-only I(q) — effective_charge=True to match ExvModel.none
    I_aa_t = debye_intensity(coords_a, types_h, q_t, use_exv=False)

    # Water-only I(q) — same convention
    I_ww_t = debye_intensity(coords_w, ["OH"] * n_w, q_t, use_exv=False)

    # Cross term — compute with effective-charge ff
    ff_atom = compute_form_factors(types_h, q_t, effective_charge=True)   # (Q, Na)
    ff_w_one = compute_form_factors(["OH"], q_t, effective_charge=True)   # (Q, 1)
    ff_water = ff_w_one.expand(-1, n_w)                                  # (Q, Nw)
    sinc_aw = _sinc_cross(coords_a, coords_w, q_t)                      # (Q, Na, Nw)
    I_aw_t = (ff_atom[:, :, None] * ff_water[:, None, :] * sinc_aw).sum(dim=(-2, -1))

    I_aa = I_aa_t.numpy()
    I_aw = I_aw_t.numpy()
    I_ww = I_ww_t.numpy()
    cw = 1.0
    I_ll_hyd = I_aa + 2 * cw * I_aw + cw**2 * I_ww

dev_hyd = np.abs(I_ll_hyd / I_pya_hyd - 1)
print(f"  I(0) ratio: {I_ll_hyd[0]/I_pya_hyd[0]:.6f}")
print(f"  max|dev|: {dev_hyd.max():.2e}  mean: {dev_hyd.mean():.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Atom-only, Fraser ExV — shows AX asymmetry bug
# ══════════════════════════════════════════════════════════════════════════════
print("\nTest 3: Atom-only, Fraser ExV (AX bug comparison) ...")
types_f, coords_f, _, q_f, I_pya_fraser = load_pyausaxs(ExvModel.fraser, hydrate=False)
coords_ft = torch.tensor(coords_f, dtype=torch.float64)

# Correct formula
loss_correct = DebyeLoss(
    q_values=q_t, I_target=torch.ones_like(q_t),
    ff_types=types_f, use_exv=True, match_ausaxs=False, device="cpu",
)
# Bug-matching formula
loss_buggy = DebyeLoss(
    q_values=q_t, I_target=torch.ones_like(q_t),
    ff_types=types_f, use_exv=True, match_ausaxs=True, device="cpu",
)
with torch.no_grad():
    I_ll_correct = loss_correct._I_pred(coords_ft).numpy()
    I_ll_buggy = loss_buggy._I_pred(coords_ft).numpy()

dev_correct = np.abs(I_ll_correct / I_pya_fraser - 1)
dev_buggy = np.abs(I_ll_buggy / I_pya_fraser - 1)
print(f"  match_ausaxs=True:  I(0) ratio={I_ll_buggy[0]/I_pya_fraser[0]:.6f}, mean|dev|={dev_buggy.mean():.2e}")
print(f"  match_ausaxs=False: I(0) ratio={I_ll_correct[0]/I_pya_fraser[0]:.6f}, mean|dev|={dev_correct.mean():.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ---------- Plot 1: Atom-only no ExV ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
ax.semilogy(q_none, I_pya_none, "k-", lw=2, label="pyAUSAXS (ExvModel.none)")
ax.semilogy(q_none, I_ll_none, "r--", lw=1.5, label="LossLab debye_intensity(use_exv=False)")
ax.set_ylabel("I(q)")
ax.set_title(f"Atom-only, no ExV — NFU1 ({len(types_none)} atoms)\nI(0) ratio = {I_ll_none[0]/I_pya_none[0]:.6f}")
ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[1]
ax.plot(q_none, (I_ll_none / I_pya_none - 1) * 100, "b-", lw=1.5)
ax.axhline(0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("q (A^-1)"); ax.set_ylabel("Deviation (%)")
ax.set_title(f"max = {dev_none.max()*100:.1f}%, mean = {dev_none.mean()*100:.1f}% (pyAUSAXS histogram binning)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "val_atom_only_no_exv.png"), dpi=150)
plt.close(fig)
print(f"\nSaved: {OUT_DIR}/val_atom_only_no_exv.png")

# ---------- Plot 2: With hydration no ExV ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
ax.semilogy(q_none, I_pya_hyd, "k-", lw=2, label=f"pyAUSAXS hydrated ({n_w} waters)")
ax.semilogy(q_none, I_ll_hyd, "r--", lw=1.5, label="LossLab (manual Debye + water)")
ax.set_ylabel("I(q)")
ax.set_title(f"With hydration, no ExV — NFU1\nI(0) ratio = {I_ll_hyd[0]/I_pya_hyd[0]:.6f}")
ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[1]
ax.plot(q_none, (I_ll_hyd / I_pya_hyd - 1) * 100, "b-", lw=1.5)
ax.axhline(0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("q (A^-1)"); ax.set_ylabel("Deviation (%)")
ax.set_title(f"max = {dev_hyd.max()*100:.1f}%, mean = {dev_hyd.mean()*100:.1f}%")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "val_hydrated_no_exv.png"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT_DIR}/val_hydrated_no_exv.png")

# ---------- Plot 3: Fraser ExV — AX bug ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
ax.semilogy(q_none, np.abs(I_pya_fraser), "k-", lw=2, label="pyAUSAXS (ExvModel.fraser)")
ax.semilogy(q_none, np.abs(I_ll_buggy), "r--", lw=1.5, label="LossLab match_ausaxs=True")
ax.semilogy(q_none, np.abs(I_ll_correct), "b:", lw=1.5, label="LossLab match_ausaxs=False (correct)")
ax.set_ylabel("|I(q)|")
ax.set_title("Fraser ExV: AX asymmetry bug comparison")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax = axes[1]
ax.plot(q_none, (I_ll_buggy / I_pya_fraser - 1) * 100, "r-", lw=1.5, label="match_ausaxs=True")
ax.plot(q_none, (I_ll_correct / I_pya_fraser - 1) * 100, "b-", lw=1.5, label="match_ausaxs=False")
ax.axhline(0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("q (A^-1)"); ax.set_ylabel("Deviation from pyAUSAXS (%)")
ax.set_title("match_ausaxs=True replicates bug; False uses correct symmetric formula")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "val_fraser_exv_ax_bug.png"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT_DIR}/val_fraser_exv_ax_bug.png")

# ---------- Plot 4: Hydration ratio ----------
ratio_pya = I_pya_hyd / (I_pya_none + 1e-30)
ratio_ll = I_ll_hyd / (I_ll_none + 1e-30)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
ax.plot(q_none, ratio_pya, "k-", lw=2, label="pyAUSAXS I_hyd/I_dry")
ax.plot(q_none, ratio_ll, "r--", lw=1.5, label="LossLab I_hyd/I_dry")
ax.axhline(1.0, color="gray", ls=":", alpha=0.5)
ax.set_ylabel("I_hydrated / I_dry")
ax.set_title("Hydration shell effect (ratio)")
ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[1]
ratio_dev = (ratio_ll / ratio_pya - 1) * 100
ax.plot(q_none, ratio_dev, "b-", lw=1.5)
ax.axhline(0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("q (A^-1)"); ax.set_ylabel("Deviation (%)")
ax.set_title(f"Ratio deviation: max = {np.abs(ratio_dev).max():.1f}%, mean = {np.abs(ratio_dev).mean():.1f}%")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "val_hydration_ratio.png"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT_DIR}/val_hydration_ratio.png")

# ---------- Plot 5: Components breakdown ----------
fig, ax = plt.subplots(figsize=(10, 5))
I_aa_np = I_aa
I_aw_np = I_aw
I_ww_np = I_ww
ax.semilogy(q_none, np.abs(I_aa_np), "b-", lw=1.5, label="|I_aa| (atom-atom)")
ax.semilogy(q_none, np.abs(2 * I_aw_np), "g-", lw=1.5, label="|2*I_aw| (cross)")
ax.semilogy(q_none, np.abs(I_ww_np), "r-", lw=1.5, label="|I_ww| (water-water)")
ax.semilogy(q_none, np.abs(I_ll_hyd), "k--", lw=2, alpha=0.5, label="|I_total|")
ax.set_xlabel("q (A^-1)"); ax.set_ylabel("|I(q)|")
ax.set_title(f"Three-component Debye decomposition — {len(types_h)} atoms, {n_w} waters, cw=1.0")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "val_components.png"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT_DIR}/val_components.png")

print("\nAll validation plots saved to", OUT_DIR)
