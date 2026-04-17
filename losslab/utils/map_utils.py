"""Map utilities for electron density map operations."""

import gemmi
import numpy as np
import torch


def create_spherical_mask_for_grid(
    map_grid: gemmi.FloatGrid,
    position: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Create spherical boolean mask for a gemmi grid."""
    temp_mask = map_grid.clone()
    temp_mask.fill(0)
    temp_mask.set_points_around(
        gemmi.Position(position[0], position[1], position[2]),
        radius=radius,
        value=1,
    )
    temp_mask.symmetrize_max()
    return np.array(temp_mask, copy=False).astype(bool)


def normalize_map(
    map_grid: torch.Tensor,
    mask: torch.Tensor | None = None,
    method: str = "zscore",
) -> torch.Tensor:
    """Normalize electron density map.

    Args:
        map_grid: Input map [Dz, Dy, Dx]
        mask: Optional mask to compute statistics from
        method: Normalization method ('zscore', 'minmax')

    Returns:
        Normalized map
    """
    masked_values = map_grid[mask] if mask is not None else map_grid

    if method == "zscore":
        mean = masked_values.mean()
        std = masked_values.std()
        return (map_grid - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = masked_values.min()
        max_val = masked_values.max()
        return (map_grid - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_spherical_mask(
    grid_shape: tuple[int, int, int],
    center: np.ndarray | torch.Tensor,
    radius: float,
    voxel_size: tuple[float, float, float],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create spherical mask in map space.

    Args:
        grid_shape: Shape of the grid (nz, ny, nx)
        center: Center position in orthogonal space [3]
        radius: Mask radius in Angstroms
        voxel_size: Voxel dimensions (vz, vy, vx) in Angstroms
        device: PyTorch device

    Returns:
        Boolean mask [Dz, Dy, Dx]
    """
    nz, ny, nx = grid_shape
    vz, vy, vx = voxel_size

    if isinstance(center, np.ndarray):
        center = torch.tensor(center, device=device, dtype=torch.float32)

    # Create coordinate grids
    z = torch.arange(nz, device=device, dtype=torch.float32) * vz
    y = torch.arange(ny, device=device, dtype=torch.float32) * vy
    x = torch.arange(nx, device=device, dtype=torch.float32) * vx

    Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
    coords = torch.stack([X, Y, Z], dim=-1)

    # Compute distances from center
    distances = torch.norm(coords - center, dim=-1)

    # Create mask
    mask = distances <= radius

    return mask


def gaussian_smooth_3d(
    map_3d: torch.Tensor,
    sigma_angstrom: float,
    voxel_size: tuple[float, float, float],
) -> torch.Tensor:
    """Apply Gaussian smoothing using FFT.

    Args:
        map_3d: Input 3D map [Dz, Dy, Dx]
        sigma_angstrom: Standard deviation in Angstroms
        voxel_size: Voxel dimensions (vz, vy, vx) in Angstroms

    Returns:
        Smoothed map
    """
    device = map_3d.device
    nz, ny, nx = map_3d.shape

    # Convert sigma to voxels
    vz, vy, vx = voxel_size
    sigma_z = sigma_angstrom / vz
    sigma_y = sigma_angstrom / vy
    sigma_x = sigma_angstrom / vx

    # Create frequency grids
    kz = torch.fft.fftfreq(nz, d=1.0, device=device)
    ky = torch.fft.fftfreq(ny, d=1.0, device=device)
    kx = torch.fft.fftfreq(nx, d=1.0, device=device)

    KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")

    # Gaussian filter in Fourier space
    K2 = (KZ / sigma_z) ** 2 + (KY / sigma_y) ** 2 + (KX / sigma_x) ** 2
    gaussian_filter = torch.exp(-2 * (torch.pi**2) * K2)

    # Apply filter
    map_fft = torch.fft.fftn(map_3d)
    smoothed_fft = map_fft * gaussian_filter
    smoothed = torch.fft.ifftn(smoothed_fft).real

    return smoothed
