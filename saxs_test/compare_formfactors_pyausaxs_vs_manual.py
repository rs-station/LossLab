"""
Compare the manually written form factors with those from pyausaxs.
"""

from LossLab.utils.form_factors import _build_coeffs_from_pyausaxs, _build_fallback_coeffs
import pyausaxs as ausaxs

def test_form_factors():
    pyausaxs = _build_coeffs_from_pyausaxs()
    fallback = _build_fallback_coeffs()

    for key in fallback.keys():
        print(f"Checking form factor type '{key}'...")
        if key not in pyausaxs:
            print(f"Warning: pyAUSAXS missing form factor type '{key}'")
            continue
        for param in ["a", "b", "c"]:
            p_val = pyausaxs[key][param]
            f_val = fallback[key][param]
            if (p_val != f_val):
                print(f"Warning: pyAUSAXS {key} {param} = {p_val} differs from header value {f_val}")

    for key in pyausaxs.keys():
        if key not in fallback:
            print(f"Extra key in pyAUSAXS: '{key}'")

def test_excluded_volumes():
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

    def _build_exv_from_pyausaxs() -> dict[str, float]:
        exv = {}
        for key in ausaxs.form_factor.valid_types():
            if key == "EXV":
                continue
            exv[str(key)] = ausaxs.form_factor.get_current_exv_volume(key)
        return exv

    pyausaxs = _build_exv_from_pyausaxs()
    print(pyausaxs)

test_form_factors()
test_excluded_volumes()