"""SAXS form factor tables and differentiable evaluation.

FORM_FACTOR_COEFFS is populated at import time by parsing the AUSAXS C++ header:
  include/core/form_factor/FormFactorTable.h

The header is located via (in order):
  1. Environment variable  AUSAXS_ROOT  (path to the AUSAXS repo root)
  2. The path stored in    AUSAXS_HEADER (direct path to FormFactorTable.h)

Convention: b values are stored in q-space (s-space values / 16π²).
  evaluate(q) = (Σ_k a_k * exp(-b_k * q²) + c) * q0

References:
  H:                  International Tables for Crystallography
  C, N, O, S, OTH:   Waasmeier & Kirfel (1995) doi:10.1107/S0108767394013292
  CH/NH/OH/SH groups: Grudinin et al. (2017) doi:10.1107/s2059798317005745
  ExV volumes:        Schaefer et al. (2001) doi:10.1002/JCC.1137
"""

from __future__ import annotations

import math
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# s_to_q_factor = 1/(16π²), matching C++: 1/(4*4*pi*pi)
_S_TO_Q: float = 1.0 / (16.0 * math.pi ** 2)

_HEADER_RELPATH = os.path.join("include", "core", "form_factor", "FormFactorTable.h")


def _find_header() -> str:
    """Return the path to FormFactorTable.h or raise FileNotFoundError."""
    # 1. Direct override
    direct = os.environ.get("AUSAXS_HEADER")
    if direct and os.path.isfile(direct):
        return direct

    # 2. Repo root override
    root = os.environ.get("AUSAXS_ROOT")
    if root:
        candidate = os.path.join(root, _HEADER_RELPATH)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "FormFactorTable.h not found.\n"
        "Set one of these environment variables:\n"
        "  AUSAXS_ROOT=/path/to/AUSAXS   (repo root)\n"
        "  AUSAXS_HEADER=/path/to/FormFactorTable.h   (direct)"
    )


def _parse_array(text: str) -> list[float]:
    """Extract floats from a C++ brace-initialiser, e.g. {1.0, -2.3, 0}."""
    inner = re.search(r"\{([^}]*)\}", text)
    if not inner:
        return []
    return [float(v.strip()) for v in inner.group(1).split(",") if v.strip()]


def _extract_namespace_body(src: str, start: int) -> tuple[str, int]:
    """Return the body inside the braces starting at `start` and the end index."""
    depth = 0
    i = start
    body_start = None
    while i < len(src):
        if src[i] == "{":
            depth += 1
            if body_start is None:
                body_start = i + 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[body_start:i], i + 1
        i += 1
    return "", len(src)


def _load_form_factor_coeffs(header_path: str) -> dict[str, dict]:
    """Parse FormFactorTable.h and return a dict keyed by atom-type name.

    Each entry: {"a": [5 floats], "b": [5 floats in q-space], "c": float}
    """
    with open(header_path) as fh:
        src = fh.read()

    # Strip C++ comments
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)

    coeffs: dict[str, dict] = {}

    # Find all "namespace <Name>" occurrences, extract body, look for a/b/c
    for m in re.finditer(r"\bnamespace\s+(\w+)\s*\{", src):
        name = m.group(1)
        body, _ = _extract_namespace_body(src, m.end() - 1)

        a_m = re.search(r"\ba\s*=\s*(\{[^}]*\})", body)
        b_m = re.search(r"\bb\s*=\s*s_to_q\s*\((\{[^}]*\})\)", body)
        c_m = re.search(r"\bc\s*=\s*([-\d.e+]+)", body)

        if not (a_m and b_m and c_m):
            continue

        a = _parse_array(a_m.group(1))
        b_s = _parse_array(b_m.group(1))
        b_q = [v * _S_TO_Q for v in b_s]
        c = float(c_m.group(1))

        coeffs[name] = {"a": a, "b": b_q, "c": c}

    return coeffs


def _build_coeffs_from_header() -> dict[str, dict]:
    """Load from header and add pyAUSAXS enum aliases."""
    path = _find_header()
    raw = _load_form_factor_coeffs(path)

    # FormFactor.h enum mappings: CH→CH_sp3, CH2→CH2_sp3, CH3→CH3_sp3,
    #                             NH3→NH3_plus, OH→OH_alc, OTH→other
    aliases = {
        "CH":  "CH_sp3",
        "CH2": "CH2_sp3",
        "CH3": "CH3_sp3",
        "NH3": "NH3_plus",
        "OH":  "OH_alc",
        "OTH": "other",
    }
    for short, full in aliases.items():
        if full in raw:
            raw[short] = raw[full]

    return raw


# Hardcoded fallback coefficients (s-space b values from Waasmaier & Kirfel /
# Grudinin). These are converted to q-space at the bottom of this block.
_FALLBACK_COEFFS_S_SPACE: dict[str, dict] = {
    "H": {"a": [0.489918, 0.262003, 0.196767, 0.049879, 0.0],
           "b": [20.6593, 7.74039, 49.5519, 2.20159, 0.0], "c": 0.001305},
    "C": {"a": [2.657506, 1.078079, 1.490909, -4.241070, 0.713791],
           "b": [14.780758, 0.776775, 42.086843, -0.000294, 0.239535], "c": 4.297983},
    "N": {"a": [11.893780, 3.277479, 1.858092, 0.858927, 0.912985],
           "b": [0.000158, 10.232723, 30.344690, 0.656065, 0.217287], "c": -11.804902},
    "O": {"a": [2.960427, 2.508818, 0.637853, 0.722838, 1.142756],
           "b": [14.182259, 5.936858, 0.112726, 34.958481, 0.390240], "c": 0.027014},
    "S": {"a": [6.362157, 5.154568, 1.473732, 1.635073, 1.209372],
           "b": [1.514347, 22.092528, 0.061373, 55.445176, 0.646925], "c": 0.154722},
    "CH": {"a": [2.909530, 0.485267, 1.516151, 0.206905, 1.541626],
            "b": [13.933084, 23.221524, 41.990403, 4.974183, 0.679266], "c": 0.337670},
    "CH2": {"a": [3.275723, 0.870037, 1.534606, 0.395078, 1.544562],
             "b": [13.408502, 23.785175, 41.922444, 5.019072, 0.724439], "c": 0.377096},
    "CH3": {"a": [3.681341, 1.228691, 1.549320, 0.574033, 1.554377],
             "b": [13.026207, 24.131974, 41.869426, 4.984373, 0.765769], "c": 0.409294},
    "NH": {"a": [1.650531, 0.429639, 2.144736, 1.851894, 1.408921],
            "b": [10.603730, 6.987283, 29.939901, 10.573859, 0.611678], "c": 0.510589},
    "NH2": {"a": [1.904157, 1.942536, 2.435585, 0.730512, 1.379728],
             "b": [10.803702, 10.792421, 29.610479, 6.847755, 0.709687], "c": 0.603738},
    "NH3": {"a": [1.882162, 1.933200, 2.465843, 0.927311, 1.190889],
             "b": [10.975157, 10.956008, 29.208572, 6.663555, 0.843650], "c": 0.597322},
    "OH": {"a": [0.456221, 3.219608, 0.812773, 2.666928, 1.380927],
            "b": [21.503498, 13.397134, 34.547137, 5.826620, 0.412902], "c": 0.463202},
    "SH": {"a": [0.570042, 6.337416, 1.641643, 5.398549, 1.527982],
            "b": [11.447986, 1.197657, 55.401032, 22.420955, 2.356552], "c": 1.523944},
    "OTH": {"a": [7.188004, 6.638454, 0.454180, 1.929593, 1.523654],
             "b": [0.956221, 15.339877, 15.339862, 39.043824, 0.062409], "c": 0.265954},
}


def _build_fallback_coeffs() -> dict[str, dict]:
    """Convert hardcoded s-space coefficients to q-space."""
    result = {}
    for name, entry in _FALLBACK_COEFFS_S_SPACE.items():
        result[name] = {
            "a": list(entry["a"]),
            "b": [v * _S_TO_Q for v in entry["b"]],
            "c": entry["c"],
        }
    return result


def _build_coeffs() -> dict[str, dict]:
    """Try parsing AUSAXS header; fall back to hardcoded tables."""
    try:
        return _build_coeffs_from_header()
    except FileNotFoundError:
        return _build_fallback_coeffs()


# ---------------------------------------------------------------------------
# Populated at import time — prefers AUSAXS header, falls back to hardcoded
# ---------------------------------------------------------------------------
FORM_FACTOR_COEFFS: dict[str, dict] = _build_coeffs()

# ---------------------------------------------------------------------------
# Number of implicit hydrogens per type (for effective charge correction)
# w = f(0) + n_H matches pyAUSAXS ExvModel.none convention.
# ---------------------------------------------------------------------------
N_IMPLICIT_H: dict[str, int] = {
    "H": 0, "C": 0, "N": 0, "O": 0, "S": 0,
    "CH": 1, "CH_sp3": 1, "CH_sp2": 1, "CH_arom": 1,
    "CH2": 2, "CH2_sp3": 2,
    "CH3": 3, "CH3_sp3": 3,
    "NH": 1, "NH_plus": 1, "NH_guanine": 1,
    "NH2": 2, "NH2_plus": 2, "NH2_guanine": 2,
    "NH3": 3, "NH3_plus": 3,
    "OH": 1, "OH_alc": 1, "OH_acid": 1,
    "O_res": 0,
    "SH": 1,
    "OTH": 0, "other": 0,
}

# ---------------------------------------------------------------------------
# Excluded volumes (Å³) — MinimumFluctuation_implicit_H (Schaefer et al. 2001)
# AUSAXS default: constants::exv::standard = MinimumFluctuation_implicit_H
# ---------------------------------------------------------------------------
EXV_VOLUMES: dict[str, float] = {
    "H": 0.0,
    "C": 12.352,
    "N": 0.027,
    "O": 14.238,
    "S": 15.413,
    "CH": 11.640,    "CH_sp3": 11.640,  "CH_sp2": 11.640,  "CH_arom": 11.640,
    "CH2": 34.583,   "CH2_sp3": 34.583,
    "CH3": 41.851,   "CH3_sp3": 41.851,
    "NH": 2.181,     "NH_plus": 2.181,   "NH_guanine": 2.181,
    "NH2": 20.562,   "NH2_plus": 20.562, "NH2_guanine": 20.562,
    "NH3": 20.722,   "NH3_plus": 20.722,
    "OH": 20.911,    "OH_alc": 20.911,   "OH_acid": 20.911,
    "O_res": 14.238,
    "SH": 28.529,
    "OTH": 20.0,     "other": 20.0,
}

_FALLBACK_FF = FORM_FACTOR_COEFFS.get("other") or FORM_FACTOR_COEFFS.get("OTH")


# ---------------------------------------------------------------------------
# Differentiable form factor evaluation
# ---------------------------------------------------------------------------

def compute_form_factors(
    ff_types: list[str],
    q_values: "torch.Tensor",
    effective_charge: bool = True,
) -> "torch.Tensor":
    """Compute q-dependent atomic form factors (Q, N).

    Implements AUSAXS FormFactor::evaluate(q):
        f(q) = (Σ_k a_k * exp(-b_k * q²) + c) * q0

    where b_k are in q-space (s-space / 16π²).

    Args:
        ff_types: N atom type strings (e.g. ["C", "N", "OH"])
        q_values: (Q,) scattering vector magnitudes in Å⁻¹
        effective_charge: if True, set q0 = (f(0) + n_implicit_H) / f(0)
            to match pyAUSAXS ExvModel.none.
            If False, q0 = 1 (raw) matching ExvModel.fraser.

    Returns:
        ff: (Q, N) form factors
    """
    import torch

    q_sq = q_values ** 2

    a_list, b_list, c_list = [], [], []
    for ff in ff_types:
        entry = FORM_FACTOR_COEFFS.get(ff, _FALLBACK_FF)
        a_list.append(entry["a"])
        b_list.append(entry["b"])
        c_list.append(entry["c"])

    a = torch.tensor(a_list, dtype=q_values.dtype, device=q_values.device)  # (N, 5)
    b = torch.tensor(b_list, dtype=q_values.dtype, device=q_values.device)  # (N, 5)
    c = torch.tensor(c_list, dtype=q_values.dtype, device=q_values.device)  # (N,)

    exponent = -b[None, :, :] * q_sq[:, None, None]                         # (Q, N, 5)
    ff = (a[None, :, :] * torch.exp(exponent)).sum(dim=-1) + c[None, :]     # (Q, N)

    if effective_charge:
        f0 = a.sum(dim=-1) + c
        n_h = torch.tensor(
            [N_IMPLICIT_H.get(str(t), 0) for t in ff_types],
            dtype=q_values.dtype, device=q_values.device,
        )
        q0 = (f0 + n_h) / (f0 + 1e-30)
        ff = ff * q0[None, :]

    return ff


def get_exv_volumes(
    ff_types: list[str],
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    """Return a tensor of excluded volumes (Å³) for the given atom types."""
    import torch

    return torch.tensor(
        [EXV_VOLUMES.get(str(t), EXV_VOLUMES.get("other", 20.0)) for t in ff_types],
        dtype=dtype, device=device,
    )
