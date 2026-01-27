"""Tests for metrics tracker."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from LossLab.refinement.metrics import MetricsTracker


def test_metrics_tracker_initialization():
    """Test metrics tracker initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = MetricsTracker(tmpdir, run_id="A", log_to_file=True)

        assert tracker.run_id == "A"
        assert tracker.output_dir == Path(tmpdir)
        assert tracker.log_file.exists()


def test_metrics_tracker_logging():
    """Test logging metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = MetricsTracker(tmpdir, run_id="A", log_to_file=False)

        # Log some metrics
        tracker.log(iteration=0, loss=1.5, plddt=0.85)
        tracker.log(iteration=1, loss=1.2, plddt=0.87)
        tracker.log(iteration=2, loss=1.0, plddt=0.90)

        # Check storage
        assert len(tracker.metrics["iteration"]) == 3
        assert len(tracker.metrics["loss"]) == 3
        assert len(tracker.metrics["plddt"]) == 3

        # Check values
        assert tracker.metrics["loss"][0] == 1.5
        assert tracker.metrics["plddt"][2] == 0.90


def test_metrics_tracker_tensor_conversion():
    """Test that tensors are converted to floats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = MetricsTracker(tmpdir, run_id="A", log_to_file=False)

        loss_tensor = torch.tensor(2.5)
        tracker.log(iteration=0, loss=loss_tensor)

        # Should be converted to float
        assert isinstance(tracker.metrics["loss"][0], float)
        assert tracker.metrics["loss"][0] == 2.5


def test_metrics_tracker_save():
    """Test saving metrics to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = MetricsTracker(tmpdir, run_id="A", log_to_file=False)

        # Log metrics
        for i in range(10):
            tracker.log(iteration=i, loss=1.0 / (i + 1))

        # Save
        tracker.save()

        # Check file exists
        save_path = Path(tmpdir) / "A_metrics.npz"
        assert save_path.exists()

        # Load and verify
        loaded = np.load(save_path)
        assert "iteration" in loaded
        assert "loss" in loaded
        assert len(loaded["iteration"]) == 10


def test_metrics_tracker_get_best():
    """Test finding best metric value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = MetricsTracker(tmpdir, run_id="A", log_to_file=False)

        # Log metrics with best at iteration 5
        losses = [2.0, 1.8, 1.5, 1.3, 1.1, 0.9, 1.0, 1.1, 1.2]
        for i, loss in enumerate(losses):
            tracker.log(iteration=i, loss=loss)

        # Find best (minimize)
        best_iter, best_value = tracker.get_best("loss", minimize=True)

        assert best_iter == 5
        assert best_value == 0.9


def test_metrics_tracker_summary():
    """Test summary generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = MetricsTracker(tmpdir, run_id="A", log_to_file=False)

        # Log some metrics
        for i in range(5):
            tracker.log(iteration=i, loss=5 - i, plddt=0.8 + i * 0.02)

        summary = tracker.summary()

        assert summary["total_iterations"] == 5
        assert summary["run_id"] == "A"
        assert "best_loss" in summary
        assert "final_loss" in summary
        assert summary["best_loss"] == 1.0
        assert summary["final_loss"] == 1.0


def test_metrics_tracker_file_logging():
    """Test CSV file logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = MetricsTracker(tmpdir, run_id="A", log_to_file=True)

        # Log metrics
        tracker.log(iteration=0, loss=1.5)
        tracker.log(iteration=1, loss=1.2)

        # Check file exists and has content
        assert tracker.log_file.exists()

        with open(tracker.log_file) as f:
            lines = f.readlines()

        # Header + 2 data lines
        assert len(lines) == 3
        assert "iteration" in lines[0]
        assert "loss" in lines[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
