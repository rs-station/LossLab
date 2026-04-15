"""Mean squared error loss for coordinate refinement."""

from __future__ import annotations

import re
import warnings

import numpy as np
import torch

from LossLab.losses.base import BaseLoss

# ---------------------------------------------------------------------------
# Smith-Waterman helpers (using biotite)
# ---------------------------------------------------------------------------


def _get_pattern_index(
    str_list: list[str] | np.ndarray,
    pattern: str,
) -> int | None:
    """Return first index whose element matches *pattern*, or ``None``."""
    for i, s in enumerate(str_list):
        if re.match(pattern, s):
            return i
    return None


def _align_sequences_sw(
    seq_moving: str,
    seq_reference: str,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Smith-Waterman alignment → residue-level index pairs.

    Uses ``biotite.sequence.align.align_optimal`` with ``local=True``
    (Smith-Waterman algorithm) for local pairwise alignment.

    Returns:
        (common_moving_resids, common_reference_resids, alignment_object)
        The first two are 0-based residue indices in the *original*
        (ungapped) sequences that are identical.
    """
    import biotite.sequence as bseq
    import biotite.sequence.align as balign

    s_ref = bseq.ProteinSequence(seq_reference)
    s_mov = bseq.ProteinSequence(seq_moving)
    matrix = balign.SubstitutionMatrix.std_protein_matrix()

    alignments = balign.align_optimal(s_ref, s_mov, matrix, local=True, max_number=1)
    aln = alignments[0]
    trace = aln.trace  # (L, 2) array; -1 = gap

    # Extract identical, non-gap residue pairs
    valid = (trace[:, 0] != -1) & (trace[:, 1] != -1)
    ref_indices = trace[valid, 0]
    mov_indices = trace[valid, 1]

    # Keep only positions where the residues are identical
    code_ref = aln.sequences[0].code
    code_mov = aln.sequences[1].code
    identical_mask = code_ref[ref_indices] == code_mov[mov_indices]

    return (
        mov_indices[identical_mask],
        ref_indices[identical_mask],
        aln,
    )


def _print_sw_alignment(alignment, len_moving: int, len_reference: int) -> None:
    """Pretty-print a biotite alignment in blocks of 60."""
    gapped = alignment.get_gapped_sequences()
    qseq = str(gapped[0])  # reference (first sequence)
    tseq = str(gapped[1])  # moving    (second sequence)
    score = alignment.score
    trace = alignment.trace

    # Determine start positions from the trace
    valid_ref = trace[:, 0] != -1
    valid_mov = trace[:, 1] != -1
    q_start = int(trace[valid_ref, 0][0]) if valid_ref.any() else 0
    t_start = int(trace[valid_mov, 1][0]) if valid_mov.any() else 0
    q_end = int(trace[valid_ref, 0][-1]) if valid_ref.any() else 0
    t_end = int(trace[valid_mov, 1][-1]) if valid_mov.any() else 0

    # Build identity line
    mid = []
    n_ident = 0
    for a, b in zip(qseq, tseq, strict=False):
        if a == b and a != "-":
            mid.append("|")
            n_ident += 1
        elif a == "-" or b == "-":
            mid.append(" ")
        else:
            mid.append(".")
    midline = "".join(mid)
    aln_len = len(qseq)

    header = (
        f"\n{'=' * 70}\n"
        f"Smith-Waterman alignment  (score {score})\n"
        f"  Reference : {len_reference} residues  "
        f"(aligned region {q_start}..{q_end})\n"
        f"  Moving    : {len_moving} residues  "
        f"(aligned region {t_start}..{t_end})\n"
        f"  Identical : {n_ident}/{aln_len} "
        f"({n_ident / aln_len * 100:.1f}%)\n"
        f"{'=' * 70}"
    )
    print(header)

    block = 60
    qi, ti = q_start, t_start  # running residue counters
    for start in range(0, aln_len, block):
        end = min(start + block, aln_len)
        q_chunk = qseq[start:end]
        t_chunk = tseq[start:end]
        m_chunk = midline[start:end]

        # Count non-gap residues in this block to advance counters
        q_nongap = sum(1 for c in q_chunk if c != "-")
        t_nongap = sum(1 for c in t_chunk if c != "-")

        print(f"  Ref  {qi:>5d}  {q_chunk}  {qi + q_nongap - 1}")
        print(f"              {m_chunk}")
        print(f"  Mov  {ti:>5d}  {t_chunk}  {ti + t_nongap - 1}")
        print()

        qi += q_nongap
        ti += t_nongap

    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# MSE loss
# ---------------------------------------------------------------------------


class MSECoordinatesLoss(BaseLoss):
    """MSE loss between predicted and reference coordinates."""

    def __init__(
        self,
        reference_coordinates: torch.Tensor | None = None,
        device: torch.device | str = "cuda:0",
        reduction: str = "mean",
        align: bool = True,
        reference_pdb=None,
        moving_pdb=None,
        selection: str = "BB",
    ) -> None:
        super().__init__(device)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self.align = align
        self.selection = selection.upper()

        self._reference_pdb = reference_pdb
        self.reference_cra = None
        if reference_coordinates is None and reference_pdb is not None:
            reference_coordinates = torch.tensor(
                reference_pdb.atom_pos,
                device=self.device,
                dtype=torch.float32,
            )
        if reference_coordinates is None:
            raise ValueError("reference_coordinates is required")

        self.reference_coordinates = reference_coordinates.to(self.device)

        if reference_pdb is not None:
            self.reference_cra = list(reference_pdb.cra_name)

        self.index_moving: np.ndarray | None = None
        self.index_reference: np.ndarray | None = None
        self._alignment_weights: np.ndarray | None = None
        if reference_pdb is not None and moving_pdb is not None:
            self.set_moving_pdb(moving_pdb)

    # ----- public helpers ---------------------------------------------------

    def set_alignment_weights(self, weights: np.ndarray | torch.Tensor | None) -> None:
        """Set per-atom weights for Kabsch alignment (e.g. derived from
        pseudo-B factors).  Accepts numpy or torch tensors."""
        self._alignment_weights = weights

    def set_moving_pdb(self, moving_pdb) -> None:
        """Compute atom-level index pairs via Smith-Waterman alignment.

        Uses the ``.sequence`` attribute on both PDB objects to run a
        Smith-Waterman alignment, then expands the residue-level matches
        to atom-level indices filtered by *self.selection*.
        """
        if self._reference_pdb is None or self.reference_cra is None:
            raise ValueError("reference_pdb is required to set moving_pdb")

        self.index_moving, self.index_reference = self._compute_common_indices(
            moving_pdb,
            self._reference_pdb,
            self.selection,
        )

    # ----- core matching ----------------------------------------------------

    @staticmethod
    def _compute_common_indices(
        moving_pdb,
        reference_pdb,
        selection: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find overlapping atom indices via Smith-Waterman sequence alignment.

        Steps:
        1. Align sequences with Smith-Waterman to get residue-level
           correspondences (handles truncations, insertions, mutations).
        2. For each aligned residue pair, find the atom-level indices in
           both CRA lists filtered by *selection* (BB / CA / ALL).
        3. Report statistics — how many residues & atoms matched / dropped.

        Args:
            moving_pdb: PDB object with ``.sequence`` and ``.cra_name``
            reference_pdb: PDB object with ``.sequence`` and ``.cra_name``
            selection: ``"BB"`` | ``"CA"`` | ``"ALL"``

        Returns:
            ``(index_moving, index_reference)`` — parallel integer arrays
            of atom indices into the respective ``cra_name`` lists.
        """
        if selection not in {"ALL", "CA", "BB"}:
            raise ValueError("selection must be one of: ALL, CA, BB")

        seq_mov = moving_pdb.sequence
        seq_ref = reference_pdb.sequence
        cra_mov = list(moving_pdb.cra_name)
        cra_ref = list(reference_pdb.cra_name)

        # 1. Sequence alignment ⟹ residue-level index pairs
        common_mov_resids, common_ref_resids, aln = _align_sequences_sw(
            seq_mov,
            seq_ref,
        )
        n_aligned_residues = len(common_mov_resids)

        # ---- pretty-print the gapped alignment ----
        _print_sw_alignment(aln, len(seq_mov), len(seq_ref))

        if n_aligned_residues == 0:
            raise ValueError(
                "Smith-Waterman alignment found zero identical residues "
                f"between moving (len={len(seq_mov)}) and reference "
                f"(len={len(seq_ref)}) sequences."
            )

        # 2. Decide which atom types to keep per residue
        if selection == "CA":
            atom_suffixes = ["CA"]
        elif selection == "BB":
            atom_suffixes = ["N", "CA", "C"]
        else:  # ALL
            atom_suffixes = None  # keep every atom in matched residues

        # 3. Expand to atom-level indices
        index_moving: list[int] = []
        index_reference: list[int] = []
        n_atoms_missing = 0

        for res_m, res_r in zip(common_mov_resids, common_ref_resids, strict=True):
            if atom_suffixes is not None:
                for atom in atom_suffixes:
                    im = _get_pattern_index(
                        cra_mov,
                        rf".*-{res_m}-.*-{atom}$",
                    )
                    ir = _get_pattern_index(
                        cra_ref,
                        rf".*-{res_r}-.*-{atom}$",
                    )
                    if im is not None and ir is not None:
                        index_moving.append(im)
                        index_reference.append(ir)
                    else:
                        n_atoms_missing += 1
            else:
                # ALL: collect every atom at this residue position that
                # shares the same atom name in both structures.
                mov_at_res = {
                    cra_mov[i].split("-")[-1]: i
                    for i, name in enumerate(cra_mov)
                    if re.match(rf".*-{res_m}-.*", name)
                }
                ref_at_res = {
                    cra_ref[i].split("-")[-1]: i
                    for i, name in enumerate(cra_ref)
                    if re.match(rf".*-{res_r}-.*", name)
                }
                common_atoms = set(mov_at_res) & set(ref_at_res)
                for atom_name in sorted(common_atoms):
                    index_moving.append(mov_at_res[atom_name])
                    index_reference.append(ref_at_res[atom_name])
                n_atoms_missing += len(set(mov_at_res) ^ set(ref_at_res))

        if not index_moving:
            raise ValueError(
                "No overlapping atoms found after Smith-Waterman alignment. "
                f"Aligned {n_aligned_residues} residues but no {selection} "
                f"atoms matched in both structures."
            )

        n_matched = len(index_moving)
        total_mov = len(seq_mov)
        total_ref = len(seq_ref)

        warnings.warn(
            f"MSECoordinatesLoss SW alignment ({selection}): "
            f"{n_aligned_residues}/{total_mov} moving residues aligned "
            f"to {n_aligned_residues}/{total_ref} reference residues → "
            f"{n_matched} atom pairs matched"
            + (
                f" ({n_atoms_missing} atoms missing in one side)"
                if n_atoms_missing
                else ""
            )
            + ".",
            stacklevel=3,
        )

        return np.array(index_moving, dtype=int), np.array(index_reference, dtype=int)

    # ----- loss computation -------------------------------------------------

    def compute(
        self,
        coordinates: torch.Tensor,
        structure_factor_calc=None,
        return_metadata: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if (
            self.index_moving is None or self.index_reference is None
        ) and coordinates.shape != self.reference_coordinates.shape:
            raise ValueError(
                "coordinates and reference_coordinates must have the same shape"
            )
        if self.align:
            from LossLab.utils.geometry import kabsch_align, weighted_kabsch

            if self._alignment_weights is not None:
                P = (
                    coordinates[self.index_moving]
                    if self.index_moving is not None
                    else coordinates
                )
                Q = (
                    self.reference_coordinates[self.index_reference]
                    if self.index_reference is not None
                    else self.reference_coordinates
                )
                w = (
                    self._alignment_weights[self.index_moving]
                    if self.index_moving is not None
                    else self._alignment_weights
                )
                R, t, _ = weighted_kabsch(P, Q, weights=w, torch_backend=True)
                aligned_coords = coordinates @ R + t
            else:
                aligned_coords = kabsch_align(
                    coordinates,
                    self.reference_coordinates,
                    indices_moving=self.index_moving,
                    indices_reference=self.index_reference,
                )
        else:
            aligned_coords = coordinates

        if self.index_moving is not None and self.index_reference is not None:
            aligned_coords = aligned_coords[self.index_moving]
            reference_coords = self.reference_coordinates[self.index_reference]
        else:
            reference_coords = self.reference_coordinates

        diff = aligned_coords - reference_coords
        if self.reduction == "sum":
            loss = torch.sum(diff**2)
        else:
            loss = torch.mean(diff**2)

        if return_metadata:
            rmse = torch.sqrt(torch.mean(diff**2)).item()
            return loss, {"rmse": rmse}
        return loss
