# saxs-loss-info

Documentation and validation for the LossLab SAXS loss functions.

- **[LOSSES.md](LOSSES.md)** — API reference, design choices, form factor types, chi2 modes, memory/performance
- **[VALIDATION.md](VALIDATION.md)** — Validation results: LossLab vs pyAUSAXS on NFU1 (atom-only, hydrated, ExV modes)
- **[validate_vs_pyausaxs.py](validate_vs_pyausaxs.py)** — Script to regenerate all validation plots
- **[plots/](plots/)** — Generated comparison figures

Regenerate plots: `conda run -n confornet python validate_vs_pyausaxs.py`
