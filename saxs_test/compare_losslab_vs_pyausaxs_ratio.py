"""Apples-to-apples comparison of LossLab SAXS vs pyAUSAXS.

Every test compares implementations that compute the *same* formula.
Differences come only from (a) histogram binning or (b) compiled vs parsed
form-factor coefficients — never from comparing different physics.

pyAUSAXS API semantics (from C++ source):
  debye_exact(mol, q)  : O(N^2) Σ w_i w_j sinc(q r_ij)  [no form factors]
  debye_raw()          : histogram-binned version of the above
  debye(ExvModel.none) : debye_raw() * exp(-q^2)  [global avg form factor]
  debye(ExvModel.fraser): per-atom FF product tables + Fraser ExV, binned

AUSAXS FFExplicit AX asymmetry (bug/limitation):
  AUSAXS iterates atom pairs (i,j) with i<j, storing 2×count in
  p_aa[type_i][type_j].  The factor=2 trick is correct for AA and XX
  (symmetric products), but the AX cross term uses f_atomic(A)×f_exv(B)
  which is NOT symmetric.  This means pair (i,j) gets weighted by
  2×f_atomic(type_i)×f_exv(type_j) instead of the correct
  [f_atomic(type_i)×f_exv(type_j) + f_atomic(type_j)×f_exv(type_i)].
  The error vanishes for same-type pairs and scales with molecular diversity.

Tests:
  1. DebyeRawLoss vs debye_exact()       — both exact O(N^2), no FF, no binning
  2. debye_raw() vs debye_exact()        — isolates histogram binning error
  3. debye(none) == debye_raw()*exp(-q^2) — verifies global FF identity
  4. DebyeLoss(fraser) vs numpy exact     — both exact O(N^2), per-atom FF + ExV
  5. DebyeLoss(match_ausaxs) vs pyAUSAXS  — exact O(N^2) with AUSAXS AX formula
  6. debye(fraser) vs AUSAXS formula      — isolates binning error for Fraser mode
  7. AUSAXS formula vs exact formula      — documents the AX asymmetry magnitude
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import pyausaxs as ausaxs
from pyausaxs.wrapper.Models import ExvModel
from pyausaxs.wrapper.settings import settings

from LossLab.utils.form_factors import (
    FORM_FACTOR_COEFFS,
    EXV_VOLUMES,
    N_IMPLICIT_H,
    compute_form_factors,
    get_exv_volumes,
)
from LossLab.losses.saxs import DebyeLoss, DebyeRawLoss

DATA_DIR = os.path.join(os.path.dirname(__file__), "pyAUSAXS", "tests", "files")
PDB_PATH = os.path.join(DATA_DIR, "2epe.pdb")
OUT_DIR = os.path.dirname(__file__)

ATOL = 1e-5   # absolute tolerance for exact O(N^2) comparisons
BTOL = 0.02   # 2% tolerance for binned comparisons


# ── helpers ──────────────────────────────────────────────────────────────────

def load_molecule():
    """Load 2epe via pyAUSAXS and extract coords, types, weights."""
    settings.exv(ExvModel.none)
    mol = ausaxs.create_molecule(PDB_PATH)
    mol.clear_hydration()
    a = mol.atoms()
    coords = np.column_stack([a[0], a[1], a[2]])
    types = [str(t) for t in a[4]]
    weights = np.array(a[3])
    return coords, types, weights


def get_q_values(bin_width=0.1):
    """Get the q grid that pyAUSAXS uses for a given bin width."""
    ausaxs.settings.histogram(bin_width=bin_width)
    mol = ausaxs.create_molecule(PDB_PATH)
    mol.clear_hydration()
    q, _ = mol.debye_raw()
    return np.array(q)


def numpy_exact_raw(coords, weights, q_arr):
    """Exact O(N^2) weighted sinc sum: Σ w_i w_j sinc(q r_ij).

    Matches pyAUSAXS debye_exact() and LossLab DebyeRawLoss.
    """
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    ww = np.outer(weights, weights)
    I_out = np.zeros(len(q_arr))
    for qi, q in enumerate(q_arr):
        sinc = np.where(dists == 0, 1.0, np.sin(q * dists) / (q * dists + 1e-30))
        I_out[qi] = (ww * sinc).sum()
    return I_out


def numpy_exact_fraser(coords, types, q_arr, rho_water=0.334):
    """Exact O(N^2) Debye with raw form factors and Fraser ExV.

    I(q) = Σ_ij (f_i(q) - f_exv_i(q)) * (f_j(q) - f_exv_j(q)) * sinc(q r_ij)

    Matches LossLab DebyeLoss(use_exv=True) and pyAUSAXS debye(fraser)
    (up to compiled-vs-parsed coefficient differences).
    """
    N = len(types)
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    entries = [FORM_FACTOR_COEFFS[t] for t in types]

    I_out = np.zeros(len(q_arr))
    for qi, q in enumerate(q_arr):
        q2 = q * q
        sinc = np.where(dists == 0, 1.0, np.sin(q * dists) / (q * dists + 1e-30))

        ff = np.zeros(N)
        for i, (t, e) in enumerate(zip(types, entries)):
            # raw 5-Gaussian: f(q) = Σ a_k exp(-b_k q^2) + c
            # b values in FORM_FACTOR_COEFFS are already in q-space
            fq = sum(a * math.exp(-b * q2) for a, b in zip(e["a"], e["b"])) + e["c"]
            # Fraser excluded volume
            V = EXV_VOLUMES.get(t, 0)
            f_exv = V * rho_water * math.exp(-V ** (2.0 / 3.0) / (4 * math.pi) * q2) if V > 0 else 0
            ff[i] = fq - f_exv

        I_out[qi] = np.einsum("i,j,ij->", ff, ff, sinc)
    return I_out


def numpy_ausaxs_fraser(coords, types, q_arr, rho_water=0.334):
    """AUSAXS-style Fraser: replicates the FFExplicit histogram asymmetry.

    Builds per-type-pair sinqd exactly as AUSAXS does (factor=2 for i<j,
    stored in p_aa[type_i][type_j]), then multiplies by form factor product
    tables.  The AX term's asymmetric indexing is the key difference from
    the exact symmetric Debye formula.

    Returns I(q) matching pyAUSAXS debye(fraser) up to binning error.
    """
    N = len(types)
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    entries = [FORM_FACTOR_COEFFS[t] for t in types]

    # Map atom types to contiguous indices
    unique_types = sorted(set(types))
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    type_idx = np.array([type_to_idx[t] for t in types])
    n_types = len(unique_types)

    # Count atoms per type (for self-correlation)
    type_counts = np.zeros(n_types)
    for i in range(N):
        type_counts[type_idx[i]] += 1

    I_out = np.zeros(len(q_arr))
    for qi, q in enumerate(q_arr):
        q2 = q * q
        sinc = np.where(dists == 0, 1.0, np.sin(q * dists) / (q * dists + 1e-30))

        # Build per-type-pair sinqd (AUSAXS histogram: factor=2, i<j only)
        sinqd = np.zeros((n_types, n_types))
        for i in range(N):
            ti = type_idx[i]
            sinqd[ti, ti] += 1  # self-correlation at d=0
            for j in range(i + 1, N):
                tj = type_idx[j]
                sinqd[ti, tj] += 2 * sinc[i, j]

        # Compute form factors per type
        ff_atom = np.zeros(n_types)
        ff_exv = np.zeros(n_types)
        for idx, t in enumerate(unique_types):
            e = FORM_FACTOR_COEFFS[t]
            ff_atom[idx] = sum(a * math.exp(-b * q2) for a, b in zip(e["a"], e["b"])) + e["c"]
            V = EXV_VOLUMES.get(t, 0)
            if V > 0:
                ff_exv[idx] = V * rho_water * math.exp(-V ** (2.0 / 3.0) / (4 * math.pi) * q2)

        # AA = Σ sinqd[A][B] * f_atom[A] * f_atom[B]
        AA = sum(sinqd[a, b] * ff_atom[a] * ff_atom[b]
                 for a in range(n_types) for b in range(n_types))

        # AX = 2 * Σ (sinqd[A][B] - δ(A,B)*N(A)) * f_atom[A] * f_exv[B]
        # NOTE: f_atom[A]*f_exv[B] is ASYMMETRIC — this is the AUSAXS behavior
        AX = 0
        for a in range(n_types):
            for b in range(n_types):
                sc = type_counts[a] if a == b else 0
                AX += 2 * (sinqd[a, b] - sc) * ff_atom[a] * ff_exv[b]

        # XX = Σ sinqd[A][B] * f_exv[A] * f_exv[B]
        XX = sum(sinqd[a, b] * ff_exv[a] * ff_exv[b]
                 for a in range(n_types) for b in range(n_types))

        I_out[qi] = AA - AX + XX
    return I_out


def rel_dev(a, b):
    """Element-wise relative deviation a/b - 1, safe for zero division."""
    return a / (b + 1e-30) - 1


def report(name, dev, tol, extra=""):
    """Print a PASS/FAIL line."""
    max_dev = np.max(np.abs(dev))
    rms_dev = np.sqrt(np.mean(dev ** 2))
    ok = max_dev < tol
    tag = "PASS" if ok else "FAIL"
    print(f"  {name:40s}: max|rel| = {max_dev:.2e}  rms|rel| = {rms_dev:.2e}  {tag}{extra}")
    return ok


# ── tests ────────────────────────────────────────────────────────────────────

def test_raw_exact(coords, types, weights, q_arr):
    """Test 1: DebyeRawLoss vs debye_exact() — both exact O(N^2), same formula."""
    print("\n" + "=" * 80)
    print("TEST 1: DebyeRawLoss vs debye_exact()  [both exact O(N^2), no FF]")
    print("=" * 80)

    # pyAUSAXS exact (O(N^2), no binning)
    settings.exv(ExvModel.none)
    mol = ausaxs.create_molecule(PDB_PATH)
    mol.clear_hydration()
    _, I_exact = ausaxs.unoptimized.debye_exact(mol, q_arr)
    I_exact = np.array(I_exact)

    # LossLab DebyeRawLoss
    coords_t = torch.tensor(coords, dtype=torch.float64)
    q_t = torch.tensor(q_arr, dtype=torch.float64)
    raw_loss = DebyeRawLoss(q_values=q_t, I_target=torch.ones_like(q_t),
                            ff_types=types, device="cpu")
    I_ll = raw_loss._I_pred(coords_t).detach().numpy()

    # numpy exact (our reference implementation)
    I_np = numpy_exact_raw(coords, weights, q_arr)

    dev_ll_exact = rel_dev(I_ll, I_exact)
    dev_np_exact = rel_dev(I_np, I_exact)
    dev_ll_np = rel_dev(I_ll, I_np)

    p1 = report("LossLab vs debye_exact()", dev_ll_exact, ATOL)
    p2 = report("numpy_exact vs debye_exact()", dev_np_exact, ATOL)
    p3 = report("LossLab vs numpy_exact", dev_ll_np, ATOL)

    return {
        "q": q_arr, "I_exact": I_exact, "I_ll": I_ll, "I_np": I_np,
        "pass": p1 and p2 and p3,
    }


def test_binning_error(q_arr, I_exact, bin_width=0.1):
    """Test 2: debye_raw() vs debye_exact() — isolates histogram binning error."""
    print("\n" + "=" * 80)
    print(f"TEST 2: debye_raw() vs debye_exact()  [binning error, bin_width={bin_width}]")
    print("=" * 80)

    settings.exv(ExvModel.none)
    ausaxs.settings.histogram(bin_width=bin_width)
    mol = ausaxs.create_molecule(PDB_PATH)
    mol.clear_hydration()
    q_raw, I_raw = mol.debye_raw()
    q_raw, I_raw = np.array(q_raw), np.array(I_raw)

    # Interpolate exact onto debye_raw's q grid if they differ
    if not np.allclose(q_raw, q_arr, atol=1e-12):
        I_exact_interp = np.interp(q_raw, q_arr, I_exact)
    else:
        I_exact_interp = I_exact

    dev = rel_dev(I_raw, I_exact_interp)
    ok = report("debye_raw() vs debye_exact()", dev, BTOL)

    return {
        "q": q_raw, "I_raw": I_raw, "I_exact": I_exact_interp,
        "dev": dev, "pass": ok,
    }


def test_none_identity(bin_width=0.1):
    """Test 3: Verify debye(none) == debye_raw() * exp(-q^2)."""
    print("\n" + "=" * 80)
    print("TEST 3: debye(none) == debye_raw() * exp(-q^2)  [global FF identity]")
    print("=" * 80)

    ausaxs.settings.histogram(bin_width=bin_width)

    # debye(none)
    settings.exv(ExvModel.none)
    mol_none = ausaxs.create_molecule(PDB_PATH)
    mol_none.clear_hydration()
    q_none, I_none = mol_none.debye()
    q_none, I_none = np.array(q_none), np.array(I_none)

    # debye_raw()
    mol_raw = ausaxs.create_molecule(PDB_PATH)
    mol_raw.clear_hydration()
    q_raw, I_raw = mol_raw.debye_raw()
    q_raw, I_raw = np.array(q_raw), np.array(I_raw)

    # They should use the same q grid
    if not np.allclose(q_none, q_raw, atol=1e-12):
        I_raw_interp = np.interp(q_none, q_raw, I_raw)
    else:
        I_raw_interp = I_raw

    # Check identity: debye(none) = debye_raw() * exp(-q^2)
    I_expected = I_raw_interp * np.exp(-q_none ** 2)
    dev = rel_dev(I_none, I_expected)
    ok = report("debye(none) vs debye_raw()*exp(-q^2)", dev, ATOL)

    return {
        "q": q_none, "I_none": I_none, "I_raw": I_raw_interp,
        "I_expected": I_expected, "dev": dev, "pass": ok,
    }


def test_fraser_exact(coords, types, q_arr):
    """Test 4: DebyeLoss(fraser) vs numpy exact — both exact O(N^2), per-atom FF."""
    print("\n" + "=" * 80)
    print("TEST 4: DebyeLoss(fraser) vs numpy exact  [both exact O(N^2), per-atom FF+ExV]")
    print("=" * 80)

    # numpy exact
    I_np = numpy_exact_fraser(coords, types, q_arr)

    # LossLab DebyeLoss
    coords_t = torch.tensor(coords, dtype=torch.float64)
    q_t = torch.tensor(q_arr, dtype=torch.float64)
    loss = DebyeLoss(q_values=q_t, I_target=torch.ones_like(q_t),
                     ff_types=types, use_exv=True, device="cpu")
    I_ll = loss._I_pred(coords_t).detach().numpy()

    dev = rel_dev(I_ll, I_np)
    ok = report("DebyeLoss(fraser) vs numpy_exact", dev, ATOL)

    return {
        "q": q_arr, "I_ll": I_ll, "I_np": I_np, "dev": dev, "pass": ok,
    }


def test_fraser_ausaxs_match(coords, types, q_arr, bin_width=0.1):
    """Test 5: DebyeLoss(match_ausaxs) vs pyAUSAXS — exact O(N^2) with AUSAXS AX."""
    print("\n" + "=" * 80)
    print("TEST 5: DebyeLoss(match_ausaxs) vs pyAUSAXS  [exact O(N^2), AUSAXS formula]")
    print("=" * 80)

    # pyAUSAXS debye(fraser)
    settings.exv(ExvModel.fraser)
    ausaxs.settings.histogram(bin_width=bin_width)
    mol = ausaxs.create_molecule(PDB_PATH)
    mol.clear_hydration()
    q_pya, I_pya = mol.debye()
    q_pya, I_pya = np.array(q_pya), np.array(I_pya)

    # LossLab DebyeLoss with match_ausaxs=True
    coords_t = torch.tensor(coords, dtype=torch.float64)
    q_t = torch.tensor(q_pya, dtype=torch.float64)
    loss = DebyeLoss(q_values=q_t, I_target=torch.ones_like(q_t),
                     ff_types=types, use_exv=True, match_ausaxs=True, device="cpu")
    I_ll = loss._I_pred(coords_t).detach().numpy()

    dev = rel_dev(I_ll, I_pya)
    mask = np.abs(I_pya) > 1
    dev_masked = dev[mask] if np.any(mask) else dev

    ok = report("DebyeLoss(match_ausaxs) vs debye(fraser) (|I|>1)", dev_masked, BTOL)

    return {
        "q": q_pya, "I_ll": I_ll, "I_pya": I_pya,
        "dev": dev, "mask": mask, "pass": ok,
    }


def test_fraser_binned(coords, types, q_arr, I_np_fraser, bin_width=0.1):
    """Test 6: pyAUSAXS debye(fraser) vs AUSAXS formula — isolates binning error."""
    print("\n" + "=" * 80)
    print(f"TEST 6: debye(fraser) vs AUSAXS formula  [binning error only, bw={bin_width}]")
    print("=" * 80)

    settings.exv(ExvModel.fraser)
    ausaxs.settings.histogram(bin_width=bin_width)
    mol = ausaxs.create_molecule(PDB_PATH)
    mol.clear_hydration()
    q_pya, I_pya = mol.debye()
    q_pya, I_pya = np.array(q_pya), np.array(I_pya)

    # Compute AUSAXS-style formula (with AX asymmetry) at pyAUSAXS q grid
    I_ausaxs = numpy_ausaxs_fraser(coords, types, q_pya)

    # Also interpolate exact formula for comparison
    if not np.allclose(q_pya, q_arr, atol=1e-12):
        I_exact_interp = np.interp(q_pya, q_arr, I_np_fraser)
    else:
        I_exact_interp = I_np_fraser

    # Mask near-zero values
    mask_ausaxs = np.abs(I_ausaxs) > 1
    mask_exact = np.abs(I_exact_interp) > 1

    dev_ausaxs = rel_dev(I_pya, I_ausaxs)
    dev_exact = rel_dev(I_pya, I_exact_interp)
    dev_ausaxs_m = dev_ausaxs[mask_ausaxs] if np.any(mask_ausaxs) else dev_ausaxs
    dev_exact_m = dev_exact[mask_exact] if np.any(mask_exact) else dev_exact

    ok = report("debye(fraser) vs AUSAXS formula (|I|>1)", dev_ausaxs_m, BTOL)
    report("debye(fraser) vs exact formula (|I|>1)", dev_exact_m, BTOL, extra="  (info: shows AX asymmetry)")

    return {
        "q": q_pya, "I_pya": I_pya, "I_ausaxs": I_ausaxs, "I_exact": I_exact_interp,
        "dev_ausaxs": dev_ausaxs, "dev_exact": dev_exact,
        "mask_ausaxs": mask_ausaxs, "mask_exact": mask_exact, "pass": ok,
    }


def test_ax_asymmetry(coords, types, q_arr, I_np_fraser):
    """Test 7: Quantify the AUSAXS AX asymmetry — exact formula vs AUSAXS formula."""
    print("\n" + "=" * 80)
    print("TEST 7: AUSAXS formula vs exact formula  [AX asymmetry magnitude]")
    print("=" * 80)

    I_ausaxs = numpy_ausaxs_fraser(coords, types, q_arr)

    mask = np.abs(I_np_fraser) > 1
    dev = rel_dev(I_ausaxs, I_np_fraser)
    dev_masked = dev[mask] if np.any(mask) else dev

    # This is informational — the asymmetry is a known AUSAXS limitation
    max_dev = np.max(np.abs(dev_masked))
    rms_dev = np.sqrt(np.mean(dev_masked ** 2))
    print(f"  {'AUSAXS formula vs exact (|I|>1)':40s}: max|rel| = {max_dev:.2e}  rms|rel| = {rms_dev:.2e}  INFO")
    print(f"  This is caused by the AX histogram asymmetry in AUSAXS FFExplicit.")
    print(f"  LossLab uses the correct symmetric formula (Test 4 passes).")

    return {
        "q": q_arr, "I_ausaxs": I_ausaxs, "I_exact": I_np_fraser,
        "dev": dev, "mask": mask,
    }


def verify_form_factors(types, weights):
    """Check pyAUSAXS weights match LossLab f(0) + n_implicit_H."""
    print("\n" + "=" * 80)
    print("FORM FACTOR VERIFICATION")
    print("=" * 80)

    unique_types = sorted(set(types))
    missing = [t for t in unique_types if t not in FORM_FACTOR_COEFFS]
    if missing:
        print(f"  MISSING types in LossLab: {missing}")
        return False

    max_diff = 0
    for t in unique_types:
        mask = np.array([tt == t for tt in types])
        pya_w = weights[mask][0]
        entry = FORM_FACTOR_COEFFS[t]
        f0 = sum(entry["a"]) + entry["c"]
        n_h = N_IMPLICIT_H.get(t, 0)
        ll_w = f0 + n_h
        diff = abs(pya_w - ll_w)
        max_diff = max(max_diff, diff)
        status = "OK" if diff < 1e-4 else "MISMATCH"
        print(f"  {t:5s}: pyAUSAXS={pya_w:10.6f}  LossLab f(0)+nH={ll_w:10.6f}  diff={diff:.2e}  {status}")

    ok = max_diff < 1e-4
    print(f"\n  Weight verification: {'PASS' if ok else 'FAIL'} (max diff = {max_diff:.2e})")
    return ok


# ── plots ────────────────────────────────────────────────────────────────────

def plot_test1(res):
    """Plot Test 1: Raw Debye exact comparison."""
    q = res["q"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.semilogy(q, res["I_exact"], "k-", lw=2, label="pyAUSAXS debye_exact()")
    ax.semilogy(q, res["I_ll"], "r--", lw=1.5, label="LossLab DebyeRawLoss")
    ax.set_ylabel("I(q)")
    ax.set_title("Test 1: Raw Debye — exact O(N^2) comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(q, rel_dev(res["I_ll"], res["I_exact"]) * 100, "r-", lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("q (1/A)")
    ax.set_ylabel("Relative deviation (%)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "test1_raw_exact.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_test2(res):
    """Plot Test 2: Binning error."""
    q = res["q"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    pct = res["dev"] * 100
    ax.plot(q, pct, "g-", lw=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.fill_between(q, pct, 0, alpha=0.15, color="green")
    ax.set_xlabel("q (1/A)")
    ax.set_ylabel("Relative deviation (%)")
    ax.set_title("Test 2: Histogram binning error — debye_raw() vs debye_exact()")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "test2_binning_error.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_test3(res):
    """Plot Test 3: debye(none) identity check."""
    q = res["q"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.semilogy(q, res["I_none"], "b-", lw=2, label="debye(none)")
    ax.semilogy(q, res["I_expected"], "r--", lw=1.5, label="debye_raw() * exp(-q^2)")
    ax.set_ylabel("I(q)")
    ax.set_title("Test 3: debye(none) identity — should be debye_raw() * exp(-q^2)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(q, res["dev"] * 100, "b-", lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("q (1/A)")
    ax.set_ylabel("Relative deviation (%)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "test3_none_identity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_test4(res):
    """Plot Test 4: Fraser exact comparison."""
    q = res["q"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.semilogy(q, np.abs(res["I_np"]), "k-", lw=2, label="numpy exact (Fraser)")
    ax.semilogy(q, np.abs(res["I_ll"]), "r--", lw=1.5, label="LossLab DebyeLoss(fraser)")
    ax.set_ylabel("|I(q)|")
    ax.set_title("Test 4: Fraser Debye — exact O(N^2) comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(q, res["dev"] * 100, "r-", lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("q (1/A)")
    ax.set_ylabel("Relative deviation (%)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "test4_fraser_exact.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_test5(res):
    """Plot Test 5: DebyeLoss(match_ausaxs) vs pyAUSAXS."""
    q = res["q"]
    mask = res["mask"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.semilogy(q, np.abs(res["I_pya"]), "g-", lw=2, label="pyAUSAXS debye(fraser)")
    ax.semilogy(q, np.abs(res["I_ll"]), "r--", lw=1.5, label="LossLab DebyeLoss(match_ausaxs)")
    ax.set_ylabel("|I(q)|")
    ax.set_title("Test 5: DebyeLoss(match_ausaxs) vs pyAUSAXS debye(fraser)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if np.any(mask):
        ax.plot(q[mask], res["dev"][mask] * 100, "r-", lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("q (1/A)")
    ax.set_ylabel("Relative deviation (%)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "test5_fraser_ausaxs_match.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_test6(res):
    """Plot Test 6: Fraser binned — pyAUSAXS vs AUSAXS formula and exact."""
    q = res["q"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.semilogy(q, np.abs(res["I_exact"]), "k-", lw=2, label="exact symmetric (Fraser)")
    ax.semilogy(q, np.abs(res["I_ausaxs"]), "b--", lw=1.5, label="AUSAXS formula (asymmetric AX)")
    ax.semilogy(q, np.abs(res["I_pya"]), "g-.", lw=1.5, label="pyAUSAXS debye(fraser)")
    ax.set_ylabel("|I(q)|")
    ax.set_title("Test 6: Fraser Debye — pyAUSAXS vs AUSAXS formula vs exact")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    m1 = res["mask_ausaxs"]
    m2 = res["mask_exact"]
    if np.any(m1):
        ax.plot(q[m1], res["dev_ausaxs"][m1] * 100, "b-", lw=1.5, label="pya vs AUSAXS formula")
    if np.any(m2):
        ax.plot(q[m2], res["dev_exact"][m2] * 100, "r--", lw=1, alpha=0.7, label="pya vs exact (AX asymmetry)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("q (1/A)")
    ax.set_ylabel("Relative deviation (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "test6_fraser_binned.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_test7(res):
    """Plot Test 7: AX asymmetry magnitude."""
    q = res["q"]
    mask = res["mask"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.semilogy(q, np.abs(res["I_exact"]), "k-", lw=2, label="exact symmetric")
    ax.semilogy(q, np.abs(res["I_ausaxs"]), "r--", lw=1.5, label="AUSAXS formula (asymmetric AX)")
    ax.set_ylabel("|I(q)|")
    ax.set_title("Test 7: AUSAXS AX asymmetry — correct formula vs AUSAXS formula")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if np.any(mask):
        ax.plot(q[mask], res["dev"][mask] * 100, "r-", lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("q (1/A)")
    ax.set_ylabel("Relative deviation (%)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "test7_ax_asymmetry.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BIN_WIDTH = 0.1

    print("=" * 80)
    print("LossLab SAXS vs pyAUSAXS — apples-to-apples comparison")
    print(f"PDB: {PDB_PATH}")
    print(f"bin_width: {BIN_WIDTH}")
    print("=" * 80)

    # Setup: load molecule and get q values
    coords, types, weights = load_molecule()
    q_arr = get_q_values(bin_width=BIN_WIDTH)
    print(f"Atoms: {len(types)}, q-points: {len(q_arr)}, "
          f"q range: [{q_arr[0]:.4f}, {q_arr[-1]:.4f}] 1/A")

    # Form factor verification
    p_ff = verify_form_factors(types, weights)

    # Test 1: Raw Debye exact O(N^2)
    res1 = test_raw_exact(coords, types, weights, q_arr)

    # Test 2: Binning error
    res2 = test_binning_error(q_arr, res1["I_exact"], bin_width=BIN_WIDTH)

    # Test 3: debye(none) identity
    res3 = test_none_identity(bin_width=BIN_WIDTH)

    # Test 4: Fraser exact O(N^2)
    res4 = test_fraser_exact(coords, types, q_arr)

    # Test 5: DebyeLoss(match_ausaxs) vs pyAUSAXS
    res5 = test_fraser_ausaxs_match(coords, types, q_arr, bin_width=BIN_WIDTH)

    # Test 6: Fraser binned (pyAUSAXS vs AUSAXS formula)
    res6 = test_fraser_binned(coords, types, q_arr, res4["I_np"], bin_width=BIN_WIDTH)

    # Test 7: AX asymmetry magnitude
    res7 = test_ax_asymmetry(coords, types, q_arr, res4["I_np"])

    # Plots
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)
    plot_test1(res1)
    plot_test2(res2)
    plot_test3(res3)
    plot_test4(res4)
    plot_test5(res5)
    plot_test6(res6)
    plot_test7(res7)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    results = {
        "form factors":            p_ff,
        "raw exact (test 1)":      res1["pass"],
        "binning error (test 2)":  res2["pass"],
        "none identity (test 3)":  res3["pass"],
        "fraser exact (test 4)":   res4["pass"],
        "fraser ausaxs (test 5)":  res5["pass"],
        "fraser binned (test 6)":  res6["pass"],
    }
    all_pass = True
    for name, passed in results.items():
        print(f"  {name:30s}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
    sys.exit(0 if all_pass else 1)
