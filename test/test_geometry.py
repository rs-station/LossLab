"""Tests for geometry utilities."""

import numpy as np
import pytest
import torch

from losslab.utils.geometry import (
    apply_rigid_body_transform,
    center_coordinates,
    compute_rmsd,
    iterative_kabsch_alignment,
    kabsch_align,
    weighted_kabsch,
)


def _random_rotation_matrix(seed=0):
    rng = np.random.RandomState(seed)
    q = rng.randn(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


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
    rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

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


def test_weighted_kabsch_numpy():
    rng = np.random.RandomState(1)
    P = rng.randn(50, 3)
    R_true = _random_rotation_matrix(2)
    t_true = np.array([0.5, -0.2, 1.0])
    Q = (P @ R_true) + t_true

    R, _, P_aligned = weighted_kabsch(P, Q, weights=None, torch_backend=False)
    assert np.allclose(P_aligned, Q, atol=1e-6)
    assert np.allclose(R @ R_true.T, np.eye(3), atol=1e-6) or np.allclose(
        R, R_true, atol=1e-6
    )


def test_weighted_kabsch_torch():
    torch.manual_seed(1)
    P = torch.randn(30, 3)
    R_true = torch.tensor(_random_rotation_matrix(3), dtype=torch.float32)
    t_true = torch.tensor([-0.3, 0.8, 0.1], dtype=torch.float32)
    Q = (P @ R_true) + t_true

    _, _, P_aligned = weighted_kabsch(P, Q, weights=None, torch_backend=True)
    assert torch.allclose(P_aligned, Q, atol=1e-5)


def test_weighted_kabsch_outlier_resistance():
    rng = np.random.RandomState(7)
    P = rng.randn(80, 3)
    R_true = _random_rotation_matrix(8)
    t_true = np.array([0.1, -0.2, 0.3])
    Q = (P @ R_true) + t_true
    Q_noisy = Q.copy()
    Q_noisy[60:] += rng.normal(scale=1.0, size=(20, 3))

    w = np.ones(80)
    w[60:] = 0.1

    _, _, P_unweighted = weighted_kabsch(P, Q_noisy, weights=None, torch_backend=False)
    _, _, P_weighted = weighted_kabsch(P, Q_noisy, weights=w, torch_backend=False)

    rmsd_unweighted = np.sqrt(((P_unweighted[:60] - Q[:60]) ** 2).mean())
    rmsd_weighted = np.sqrt(((P_weighted[:60] - Q[:60]) ** 2).mean())
    assert rmsd_weighted < rmsd_unweighted


def test_iterative_kabsch_uniform_weights():
    rng = np.random.RandomState(9)
    P = rng.randn(40, 3)
    R_true = _random_rotation_matrix(10)
    t_true = np.array([0.2, 0.1, -0.3])
    Q = (P @ R_true) + t_true

    ones = np.ones(P.shape[0])
    _, _, P_none = iterative_kabsch_alignment(
        P, Q, weights=None, torch_backend=False, max_iters=3
    )
    _, _, P_ones = iterative_kabsch_alignment(
        P, Q, weights=ones, torch_backend=False, max_iters=3
    )

    assert np.allclose(P_none, Q, atol=1e-6)
    assert np.allclose(P_ones, Q, atol=1e-6)
    assert np.allclose(P_none, P_ones, atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
