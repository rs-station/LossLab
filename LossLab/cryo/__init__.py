"""Cryo-EM specific helpers (re-exported from canonical locations)."""


def __getattr__(name):
    if name == "CryoEMLLGLoss":
        from LossLab.losses.cryoLLGI import CryoEMLLGLoss

        return CryoEMLLGLoss
    if name in ("extract_allatoms", "get_res_names", "position_alignment"):
        from LossLab.utils import alignment

        return getattr(alignment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CryoEMLLGLoss",
    "extract_allatoms",
    "get_res_names",
    "position_alignment",
]
