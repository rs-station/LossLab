"""
Compare the manually written form factors with those from pyausaxs.
"""

from LossLab.utils.form_factors import _build_coeffs_from_pyausaxs, _build_fallback_coeffs

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