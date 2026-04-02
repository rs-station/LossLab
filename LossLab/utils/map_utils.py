"""Map utilities for electron density map operations."""

import gemmi
import numpy as np
import torch


def parse_pdb_coords(pdb_file: str) -> list[list[float]]:
    """Extract atomic coordinates from a PDB file."""
    coords: list[list[float]] = []
    structure = gemmi.read_structure(pdb_file)
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return coords


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


def denoise_and_mask_ccp4_map(
    input_ccp4: gemmi.Ccp4Map,
    ligand_coords: list[list[float]],
    high_res_limit: float = 1.8,
    mask_radius: float = 2.5,
    tv_denoise: bool = False,
) -> gemmi.Ccp4Map:
    """Optionally denoise a CCP4 map and mask ligand regions (in-memory)."""
    if tv_denoise:
        from meteor import tv
        from meteor.rsmap import Map

        rsmap = Map.from_ccp4_map(input_ccp4, high_resolution_limit=high_res_limit)
        denoised_map, _metadata = tv.tv_denoise_difference_map(rsmap, full_output=True)
        output_ccp4 = denoised_map.to_ccp4_map(map_sampling=3)
    else:
        output_ccp4 = input_ccp4

    grid = output_ccp4.grid

    combined_mask = np.zeros(grid.shape, dtype=bool)
    for coord in ligand_coords:
        mask = create_spherical_mask_for_grid(
            grid,
            np.array(coord),
            mask_radius,
        )
        combined_mask |= mask

    grid_array = np.array(grid, copy=False)
    grid_array[combined_mask] = 0
    return output_ccp4


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


def apply_mask(
    map_grid: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Apply binary mask to map.

    Args:
        map_grid: Input map [Dz, Dy, Dx]
        mask: Boolean mask [Dz, Dy, Dx]
        fill_value: Value to fill masked regions

    Returns:
        Masked map
    """
    masked_map = map_grid.clone()
    masked_map[~mask] = fill_value
    return masked_map


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


def save_map_as_ccp4(
    map_tensor: torch.Tensor,
    reference_map: gemmi.Ccp4Map,
    output_path: str,
) -> None:
    """Save torch tensor as CCP4 map file.

    Args:
        map_tensor: Map data as tensor [Dz, Dy, Dx]
        reference_map: Reference map for unit cell and space group info
        output_path: Output file path
    """
    # Convert to numpy
    map_np = map_tensor.detach().cpu().numpy().astype(np.float32)

    # Create new grid with reference properties
    grid = gemmi.FloatGrid(*map_np.shape)
    grid.set_unit_cell(reference_map.grid.unit_cell)
    grid.spacegroup = reference_map.grid.spacegroup

    # Copy data
    np.copyto(np.array(grid, copy=False), map_np)

    # Write CCP4 file
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(output_path)


def compute_local_correlation(
    map1: torch.Tensor,
    map2: torch.Tensor,
    window_size: int = 5,
) -> torch.Tensor:
    """Compute local correlation between two maps.

    Args:
        map1: First map [Dz, Dy, Dx]
        map2: Second map [Dz, Dy, Dx]
        window_size: Size of local window

    Returns:
        Local correlation map [Dz, Dy, Dx]
    """
    # Simple sliding window correlation
    # For production, consider using unfold or conv3d for efficiency
    pad = window_size // 2
    correlation_map = torch.zeros_like(map1)

    nz, ny, nx = map1.shape

    for z in range(pad, nz - pad):
        for y in range(pad, ny - pad):
            for x in range(pad, nx - pad):
                window1 = map1[
                    z - pad : z + pad + 1,
                    y - pad : y + pad + 1,
                    x - pad : x + pad + 1,
                ].flatten()
                window2 = map2[
                    z - pad : z + pad + 1,
                    y - pad : y + pad + 1,
                    x - pad : x + pad + 1,
                ].flatten()

                # Compute correlation
                mean1, mean2 = window1.mean(), window2.mean()
                std1, std2 = window1.std(), window2.std()

                if std1 > 1e-8 and std2 > 1e-8:
                    corr = torch.mean((window1 - mean1) * (window2 - mean2))
                    corr = corr / (std1 * std2)
                    correlation_map[z, y, x] = corr

    return correlation_map
