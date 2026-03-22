"""Smith-Waterman based common-index discovery for CRA name lists."""

from __future__ import annotations

import re

import numpy as np

# 3-letter -> 1-letter mapping for standard amino acids
_AA3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def _get_identical_indices(aligned_a: str, aligned_b: str):
    """Find indices of identical residues in a pairwise alignment."""
    ind_a: list[int] = []
    ind_b: list[int] = []
    ai = 0
    bi = 0
    for a, b in zip(aligned_a, aligned_b, strict=False):
        if a == "-":
            bi += 1
            continue
        if b == "-":
            ai += 1
            continue
        if a == b:
            ind_a.append(ai)
            ind_b.append(bi)
        ai += 1
        bi += 1
    return np.array(ind_a), np.array(ind_b)


def _get_pattern_index(str_list, pattern: str):
    """Return first index in str_list matching regex pattern."""
    for i, s in enumerate(str_list):
        if re.match(pattern, str(s)):
            return i
    return None


def _sequence_from_cra(cra_name: list[str]) -> str:
    """Extract 1-letter sequence from CRA name list (CA atoms only)."""
    seq_parts: list[tuple[int, str]] = []
    seen: set[int] = set()
    for name in cra_name:
        parts = str(name).split("-")
        if len(parts) < 4:
            continue
        if parts[-1] != "CA":
            continue
        try:
            resid = int(parts[1])
        except ValueError:
            continue
        if resid in seen:
            continue
        seen.add(resid)
        resname = parts[2]
        one_letter = _AA3TO1.get(resname, "X")
        seq_parts.append((resid, one_letter))
    seq_parts.sort()
    return "".join(c for _, c in seq_parts)


def _atom_suffixes_for_selection(selection: str) -> list[str]:
    """Return atom name suffixes for the given selection mode."""
    sel = selection.upper()
    if sel == "CA":
        return ["CA"]
    if sel == "BB":
        return ["N", "CA", "C"]
    if sel == "ALL":
        return [".*"]
    raise ValueError(f"selection must be one of: ALL, CA, BB; got {sel}")


def compute_common_indices(
    moving_cra: list[str],
    reference_cra: list[str],
    selection: str = "CA",
    moving_sequence: str | None = None,
    reference_sequence: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find matching atom indices using Smith-Waterman alignment.

    Uses ``skbio.alignment.StripedSmithWaterman`` to handle sequences
    with missing residues, truncations, or insertions.

    Parameters
    ----------
    moving_cra, reference_cra : list[str]
        CRA name lists (e.g. ``['A-0-ALA-CA', ...]``).
    selection : {'ALL', 'CA', 'BB'}
        Which atom types to match.
    moving_sequence, reference_sequence : str, optional
        1-letter amino acid sequences. If not provided, they are
        inferred from ``cra_name`` entries.

    Returns
    -------
    index_moving, index_reference : np.ndarray
        Matched atom indices into the respective CRA lists.
    """
    from skbio.alignment import StripedSmithWaterman

    if moving_sequence is None:
        moving_sequence = _sequence_from_cra(moving_cra)
    if reference_sequence is None:
        reference_sequence = _sequence_from_cra(reference_cra)

    query = StripedSmithWaterman(reference_sequence)
    alignment = query(moving_sequence)

    sub_ref = np.arange(alignment.query_begin, alignment.query_end + 1)
    sub_mov = np.arange(
        alignment.target_begin,
        alignment.target_end_optimal + 1,
    )
    subsub_ref, subsub_mov = _get_identical_indices(
        alignment.aligned_query_sequence,
        alignment.aligned_target_sequence,
    )
    common_ref_res = sub_ref[subsub_ref]
    common_mov_res = sub_mov[subsub_mov]

    atom_suffixes = _atom_suffixes_for_selection(selection)

    index_moving: list[int] = []
    index_reference: list[int] = []

    for res_ref, res_mov in zip(common_ref_res, common_mov_res, strict=False):
        for suffix in atom_suffixes:
            pattern_ref = rf".*-{res_ref}-.*-{suffix}$"
            pattern_mov = rf".*-{res_mov}-.*-{suffix}$"
            idx_ref = _get_pattern_index(reference_cra, pattern_ref)
            idx_mov = _get_pattern_index(moving_cra, pattern_mov)
            if idx_ref is not None and idx_mov is not None:
                index_reference.append(idx_ref)
                index_moving.append(idx_mov)

    if not index_moving:
        raise ValueError("No overlapping atoms found between moving and reference")

    return np.array(index_moving), np.array(index_reference)


__all__ = [
    "compute_common_indices",
]
