"""Tests for checkpoint manager."""

import tempfile
from pathlib import Path

import pytest
import torch

from LossLab.refinement.checkpoint import CheckpointManager


def test_checkpoint_manager_initialization():
    """Test checkpoint manager initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, save_best_only=False)

        assert manager.output_dir == Path(tmpdir)
        assert manager.best_loss == float("inf")


def test_save_checkpoint():
    """Test saving a checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, save_best_only=False)

        # Create test tensor
        test_tensor = torch.randn(10, 20)

        # Save checkpoint
        saved = manager.save_checkpoint(
            iteration=5,
            run_id="A",
            loss=1.23,
            msa_bias=test_tensor,
        )

        assert saved is True

        # Check file exists
        tensor_path = Path(tmpdir) / "A_5_msa_bias.pt"
        assert tensor_path.exists()


def test_save_best_only():
    """Test save_best_only mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, save_best_only=True)

        tensor1 = torch.randn(10, 20)
        tensor2 = torch.randn(10, 20)

        # First save (should succeed - is best)
        saved1 = manager.save_checkpoint(
            iteration=0,
            run_id="A",
            loss=2.0,
            msa_bias=tensor1,
        )
        assert saved1 is True
        assert manager.best_loss == 2.0

        # Second save with worse loss (should not save)
        saved2 = manager.save_checkpoint(
            iteration=1,
            run_id="A",
            loss=3.0,
            msa_bias=tensor2,
        )
        assert saved2 is False
        assert manager.best_loss == 2.0

        # Third save with better loss (should save)
        saved3 = manager.save_checkpoint(
            iteration=2,
            run_id="A",
            loss=1.5,
            msa_bias=tensor2,
        )
        assert saved3 is True
        assert manager.best_loss == 1.5


def test_load_checkpoint():
    """Test loading a checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, save_best_only=False)

        # Create and save test tensor
        test_tensor = torch.randn(10, 20)
        manager.save_checkpoint(
            iteration=5,
            run_id="A",
            loss=1.23,
            msa_bias=test_tensor,
        )

        # Load checkpoint
        loaded = manager.load_checkpoint(
            run_id="A",
            iteration=5,
            tensor_keys=["msa_bias"],
        )

        assert "msa_bias" in loaded
        assert torch.allclose(loaded["msa_bias"], test_tensor)


def test_load_best():
    """Test loading best checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, save_best_only=False)

        # Save multiple checkpoints
        tensor1 = torch.randn(10, 20)
        tensor2 = torch.randn(10, 20)

        manager.save_checkpoint(0, "A", 2.0, msa_bias=tensor1)
        manager.save_checkpoint(1, "A", 1.5, msa_bias=tensor2)  # This is best
        manager.save_checkpoint(2, "A", 1.8, msa_bias=tensor1)

        # Load best
        loaded = manager.load_best(["msa_bias"])

        assert loaded is not None
        assert "msa_bias" in loaded
        assert torch.allclose(loaded["msa_bias"], tensor2)


def test_get_best_info():
    """Test getting best checkpoint info."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, save_best_only=False)

        # Save checkpoint
        test_tensor = torch.randn(5, 10)
        manager.save_checkpoint(10, "B", 0.95, msa_bias=test_tensor)

        # Get best info
        info = manager.get_best_info()

        assert info["iteration"] == 10
        assert info["run_id"] == "B"
        assert info["loss"] == 0.95


def test_multiple_tensors():
    """Test saving and loading multiple tensors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, save_best_only=False)

        # Create multiple tensors
        bias = torch.randn(10, 20)
        weights = torch.randn(10, 20)

        # Save checkpoint with multiple tensors
        manager.save_checkpoint(
            iteration=3,
            run_id="C",
            loss=1.11,
            msa_bias=bias,
            feat_weights=weights,
        )

        # Load both
        loaded = manager.load_checkpoint(
            "C", 3, ["msa_bias", "feat_weights"]
        )

        assert "msa_bias" in loaded
        assert "feat_weights" in loaded
        assert torch.allclose(loaded["msa_bias"], bias)
        assert torch.allclose(loaded["feat_weights"], weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
