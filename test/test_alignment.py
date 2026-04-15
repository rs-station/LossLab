"""Tests for Smith-Waterman alignment in MSECoordinatesLoss.

Uses the 4LZT / 4LZT_truncated PDB pair from issue #2:
https://github.com/rs-station/LossLab/issues/2

The truncated structure is missing the first 4 N-terminal residues
(KVFG), so the alignment should map residues 0..124 of the truncated
structure to residues 4..128 of the full structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

DATA_DIR = Path(__file__).resolve().parent / "data"
PDB_FULL = DATA_DIR / "4LZT.pdb"
PDB_TRUNC = DATA_DIR / "4LZT_truncated.pdb"

# Skip the entire module if the PDB fixtures or SFC_Torch are missing.
pytestmark = pytest.mark.skipif(
    not PDB_FULL.exists() or not PDB_TRUNC.exists(),
    reason="4LZT test PDB files not found in test/data/",
)

try:
    from SFC_Torch import PDBParser
except ImportError:
    PDBParser = None

needs_sfc = pytest.mark.skipif(PDBParser is None, reason="SFC_Torch not installed")


# ---- Helpers ---------------------------------------------------------------


@pytest.fixture()
def pdb_full():
    return PDBParser(str(PDB_FULL))


@pytest.fixture()
def pdb_trunc():
    return PDBParser(str(PDB_TRUNC))


SEQ_FULL = (
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINS"
    "RWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQ"
    "AWIRGCRL"
)
SEQ_TRUNC = SEQ_FULL[4:]  # missing first 4 residues (KVFG)


# ---- _align_sequences_sw --------------------------------------------------


class TestAlignSequencesSW:
    """Tests for the low-level biotite-based SW alignment helper."""

    def test_identical_sequences(self):
        from LossLab.losses.mse import _align_sequences_sw

        mov, ref, _aln = _align_sequences_sw(SEQ_FULL, SEQ_FULL)
        # Every residue should match itself
        assert len(mov) == len(SEQ_FULL)
        np.testing.assert_array_equal(mov, ref)

    def test_truncated_n_terminal(self):
        from LossLab.losses.mse import _align_sequences_sw

        mov, ref, _aln = _align_sequences_sw(SEQ_TRUNC, SEQ_FULL)
        # All 125 truncated residues should align
        assert len(mov) == len(SEQ_TRUNC) == 125
        # Moving indices should be 0..124
        np.testing.assert_array_equal(mov, np.arange(125))
        # Reference indices should be 4..128 (offset by 4)
        np.testing.assert_array_equal(ref, np.arange(4, 129))

    def test_truncated_c_terminal(self):
        from LossLab.losses.mse import _align_sequences_sw

        seq_c_trunc = SEQ_FULL[:-5]  # drop last 5 residues
        mov, ref, _aln = _align_sequences_sw(seq_c_trunc, SEQ_FULL)
        assert len(mov) == len(seq_c_trunc)
        np.testing.assert_array_equal(mov, np.arange(len(seq_c_trunc)))
        np.testing.assert_array_equal(ref, np.arange(len(seq_c_trunc)))

    def test_single_mutation(self):
        from LossLab.losses.mse import _align_sequences_sw

        # Mutate position 10 (A→W)
        seq_mut = SEQ_FULL[:10] + "W" + SEQ_FULL[11:]
        mov, ref, _aln = _align_sequences_sw(seq_mut, SEQ_FULL)
        # The mutated position should be excluded (not identical)
        assert 10 not in mov
        assert 10 not in ref
        # All other positions should be present
        assert len(mov) == len(SEQ_FULL) - 1

    def test_internal_deletion(self):
        from LossLab.losses.mse import _align_sequences_sw

        # Delete residues 50..54
        seq_del = SEQ_FULL[:50] + SEQ_FULL[55:]
        mov, ref, _aln = _align_sequences_sw(seq_del, SEQ_FULL)
        # Moving should not contain anything mapping to deleted region
        # and all matched ref indices should be outside 50..54
        for r in ref:
            assert r < 50 or r >= 55
        # Total matches = full length - 5 deleted
        assert len(mov) == len(SEQ_FULL) - 5

    def test_returns_alignment_object(self):
        from LossLab.losses.mse import _align_sequences_sw

        _mov, _ref, aln = _align_sequences_sw(SEQ_TRUNC, SEQ_FULL)
        # biotite Alignment should have score, trace, sequences
        assert hasattr(aln, "score")
        assert hasattr(aln, "trace")
        assert aln.score > 0


# ---- _compute_common_indices (with real PDB objects) -----------------------


@needs_sfc
class TestComputeCommonIndices:
    """Tests for atom-level index matching via SW alignment on PDB objects."""

    def test_ca_selection(self, pdb_full, pdb_trunc):
        from LossLab.losses.mse import MSECoordinatesLoss

        idx_mov, idx_ref = MSECoordinatesLoss._compute_common_indices(
            pdb_trunc, pdb_full, "CA"
        )
        # 125 aligned residues → 125 CA atoms
        assert len(idx_mov) == 125
        assert len(idx_ref) == 125
        # Indices should be valid for their respective CRA lists
        assert idx_mov.max() < len(pdb_trunc.cra_name)
        assert idx_ref.max() < len(pdb_full.cra_name)
        # Every matched CRA should end in "-CA"
        for im, ir in zip(idx_mov, idx_ref, strict=True):
            assert pdb_trunc.cra_name[im].endswith("-CA")
            assert pdb_full.cra_name[ir].endswith("-CA")

    def test_bb_selection(self, pdb_full, pdb_trunc):
        from LossLab.losses.mse import MSECoordinatesLoss

        idx_mov, idx_ref = MSECoordinatesLoss._compute_common_indices(
            pdb_trunc, pdb_full, "BB"
        )
        # 125 residues × 3 backbone atoms (N, CA, C)
        assert len(idx_mov) == 125 * 3
        assert len(idx_ref) == 125 * 3

    def test_all_selection(self, pdb_full, pdb_trunc):
        from LossLab.losses.mse import MSECoordinatesLoss

        idx_mov, idx_ref = MSECoordinatesLoss._compute_common_indices(
            pdb_trunc, pdb_full, "ALL"
        )
        # Should match ≥ BB atoms
        assert len(idx_mov) >= 125 * 3
        assert len(idx_ref) >= 125 * 3
        # Should not exceed the smaller CRA list
        assert len(idx_mov) <= len(pdb_trunc.cra_name)

    def test_invalid_selection_raises(self, pdb_full, pdb_trunc):
        from LossLab.losses.mse import MSECoordinatesLoss

        with pytest.raises(ValueError, match="selection must be one of"):
            MSECoordinatesLoss._compute_common_indices(pdb_trunc, pdb_full, "INVALID")

    def test_same_structure(self, pdb_full):
        from LossLab.losses.mse import MSECoordinatesLoss

        idx_mov, idx_ref = MSECoordinatesLoss._compute_common_indices(
            pdb_full, pdb_full, "CA"
        )
        # Same structure → all 129 CA atoms should match
        assert len(idx_mov) == 129
        np.testing.assert_array_equal(idx_mov, idx_ref)


# ---- End-to-end MSECoordinatesLoss ----------------------------------------


@needs_sfc
class TestMSECoordinatesLossE2E:
    """End-to-end tests for MSECoordinatesLoss with the 4LZT pair."""

    def test_loss_near_zero_for_same_coords(self, pdb_full, pdb_trunc):
        """When moving coords come from the same structure, loss ≈ 0."""
        from LossLab.losses.mse import MSECoordinatesLoss

        loss_fn = MSECoordinatesLoss(
            reference_pdb=pdb_full,
            moving_pdb=pdb_trunc,
            selection="CA",
            device="cpu",
        )
        coords = torch.tensor(pdb_trunc.atom_pos, dtype=torch.float32)
        loss = loss_fn.compute(coords)
        assert loss.item() < 1e-6

    def test_loss_positive_for_displaced_coords(self, pdb_full, pdb_trunc):
        """Non-rigid displacement should produce positive loss."""
        from LossLab.losses.mse import MSECoordinatesLoss

        loss_fn = MSECoordinatesLoss(
            reference_pdb=pdb_full,
            moving_pdb=pdb_trunc,
            selection="BB",
            device="cpu",
        )
        coords = torch.tensor(pdb_trunc.atom_pos, dtype=torch.float32)
        # Add per-atom random noise (non-rigid → Kabsch can't remove it)
        torch.manual_seed(0)
        coords = coords + torch.randn_like(coords) * 2.0
        loss = loss_fn.compute(coords)
        assert loss.item() > 1.0

    def test_loss_with_metadata(self, pdb_full, pdb_trunc):
        """compute(return_metadata=True) should return RMSD in metadata."""
        from LossLab.losses.mse import MSECoordinatesLoss

        loss_fn = MSECoordinatesLoss(
            reference_pdb=pdb_full,
            moving_pdb=pdb_trunc,
            selection="CA",
            device="cpu",
        )
        coords = torch.tensor(pdb_trunc.atom_pos, dtype=torch.float32)
        loss, metadata = loss_fn.compute(coords, return_metadata=True)
        assert "rmse" in metadata
        assert metadata["rmse"] < 0.01  # near-zero for same coords

    def test_set_moving_pdb_updates_indices(self, pdb_full, pdb_trunc):
        """Calling set_moving_pdb should update index arrays."""
        from LossLab.losses.mse import MSECoordinatesLoss

        loss_fn = MSECoordinatesLoss(
            reference_pdb=pdb_full,
            selection="CA",
            device="cpu",
        )
        # Before set_moving_pdb, indices should be None
        assert loss_fn.index_moving is None
        assert loss_fn.index_reference is None

        loss_fn.set_moving_pdb(pdb_trunc)
        assert loss_fn.index_moving is not None
        assert loss_fn.index_reference is not None
        assert len(loss_fn.index_moving) == 125

    def test_alignment_weights(self, pdb_full, pdb_trunc):
        """Setting alignment weights should not break loss computation."""
        from LossLab.losses.mse import MSECoordinatesLoss

        loss_fn = MSECoordinatesLoss(
            reference_pdb=pdb_full,
            moving_pdb=pdb_trunc,
            selection="CA",
            device="cpu",
        )
        coords = torch.tensor(pdb_trunc.atom_pos, dtype=torch.float32)
        n_atoms = coords.shape[0]
        weights = torch.ones(n_atoms)
        loss_fn.set_alignment_weights(weights)
        loss = loss_fn.compute(coords)
        assert loss.item() < 1e-6

    def test_no_align_mode(self, pdb_full):
        """align=False should skip Kabsch and compute raw MSE."""
        from LossLab.losses.mse import MSECoordinatesLoss

        loss_fn = MSECoordinatesLoss(
            reference_pdb=pdb_full,
            moving_pdb=pdb_full,
            selection="CA",
            align=False,
            device="cpu",
        )
        coords = torch.tensor(pdb_full.atom_pos, dtype=torch.float32)
        loss = loss_fn.compute(coords)
        assert loss.item() < 1e-6
