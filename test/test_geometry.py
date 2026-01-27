"""Tests for geometry utilities."""

import numpy as np
import pytest
import torch

from LossLab.utils.geometry import (
    apply_rigid_body_transform,
    center_coordinates,
    compute_rmsd,
    kabsch_align,
)


def test_kabsch_align_identity():
    """Test that aligning identical coordinates gives identity transformation."""
    coords = torch.randn(100, 3)
    aligned = kabsch_align(coords, coords)

    # Should be nearly identical
    rmsd = compute_rmsd(coords, aligned)
    assert rmsd < 1e-5


def test_kabsch_align_rotation():
    """Test Kabsch alignment with known rotation."""
    # Create a set of coordinates
    original = torch.randn(50, 3)

    # Apply known rotation
    angle = np.pi / 4
    rotation = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    rotated = torch.matmul(original, rotation.T)

    # Align back
    aligned = kabsch_align(rotated, original)

    # Should be close to original
    rmsd = compute_rmsd(original, aligned)
    assert rmsd < 1e-4


def test_compute_rmsd_zero():
    """Test RMSD is zero for identical coordinates."""
    coords = torch.randn(100, 3)
    rmsd = compute_rmsd(coords, coords)
    assert rmsd < 1e-10


def test_compute_rmsd_known_value():
    """Test RMSD with known displacement."""
    coords1 = torch.zeros(10, 3)
    coords2 = torch.ones(10, 3)  # Displaced by 1.0 in each dimension

    rmsd = compute_rmsd(coords1, coords2)
    expected_rmsd = np.sqrt(3.0)  # sqrt(1^2 + 1^2 + 1^2)

    assert abs(rmsd - expected_rmsd) < 1e-5


def test_center_coordinates():
    """Test coordinate centering."""
    # Create coordinates with known centroid
    coords = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ])

    centered, centroid = center_coordinates(coords)

    # Check centroid is correct
    expected_centroid = torch.tensor([2.0, 3.0, 4.0])
    assert torch.allclose(centroid, expected_centroid)

    # Check centered coordinates have zero mean
    assert torch.allclose(centered.mean(dim=0), torch.zeros(3), atol=1e-6)


def test_apply_rigid_body_transform():
    """Test rigid body transformation."""
    coords = torch.randn(50, 3)

    # Identity rotation, zero translation
    rotation = torch.eye(3)
    translation = torch.zeros(3)

    transformed = apply_rigid_body_transform(coords, rotation, translation)

    assert torch.allclose(coords, transformed)


def test_kabsch_with_subset():
    """Test Kabsch alignment using subset of atoms."""
    coords1 = torch.randn(100, 3)
    coords2 = coords1.clone()
    coords2[:50] += 1.0  # Displace first 50 atoms

    # Align using only last 50 atoms (which are identical)
    indices = np.arange(50, 100)
    aligned = kabsch_align(coords2, coords1, indices, indices)

    # Last 50 should be very close
    rmsd_subset = compute_rmsd(aligned[50:], coords1[50:])
    assert rmsd_subset < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
