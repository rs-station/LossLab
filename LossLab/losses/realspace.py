"""Real-space loss functions for comparing model maps with target maps."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

import gemmi
import numpy as np
import torch
from geomloss import SamplesLoss
from loguru import logger

if TYPE_CHECKING:
    from SFC_Torch import PDBParser

from LossLab.losses.base import BaseLoss
from LossLab.utils.decorators import cached_property, gpu_memory_tracked, timed
from LossLab.utils.map_utils import (
    create_spherical_mask,
    create_spherical_mask_for_grid,
    gaussian_smooth_3d,
    normalize_map,
)


class RealSpaceLoss(BaseLoss):
    """Real-space loss comparing model-generated maps with target maps.

    This class handles various loss types (CC, L2, Sinkhorn) for
    comparing model coordinates against target experimental maps.

    Example:
        >>> loss_fn = RealSpaceLoss(
        ...     target_map=ccp4_map,
        ...     pdb_obj=pdb_parser,
        ...     loss_type="l2",
        ...     mask_center=np.array([10.0, 20.0, 15.0]),
        ...     mask_radius=15.0,
        ... )
        >>> loss = loss_fn.compute(coordinates, structure_factor_calc)
    """

    def __init__(
        self,
        target_map: gemmi.Ccp4Map,
        pdb_obj: PDBParser,
        device: torch.device | str = "cuda:0",
        loss_type: Literal["cc", "l2", "sinkhorn", "density_explained"] = "l2",
        mask_center: np.ndarray | None = None,
        mask_radius: float | None = None,
    ):
        """Initialize real-space loss.

        Args:
            target_map: Target CCP4 map
            pdb_obj: PDB parser object with structure information
            device: PyTorch device for computations
            loss_type: Type of loss function to use
            mask_center: Center of spherical mask in orthogonal coordinates
            mask_radius: Radius of spherical mask in Angstroms
        """
        super().__init__(device)

        self.pdb_obj = pdb_obj
        self.loss_type = loss_type
        self.target_ccp4_map = target_map
        self.mask_center = mask_center
        self.mask_radius = mask_radius

        # Extract map properties
        self.grid_shape = (target_map.grid.nu, target_map.grid.nv, target_map.grid.nw)
        self.unit_cell = target_map.grid.unit_cell
        self.voxel_size = (
            self.unit_cell.a / self.grid_shape[0],
            self.unit_cell.b / self.grid_shape[1],
            self.unit_cell.c / self.grid_shape[2],
        )

        # Load target map to device
        target_grid_np = np.array(target_map.grid, copy=False)
        self.target_map_grid = torch.tensor(
            target_grid_np, device=self.device, dtype=torch.float32
        ).clone()

        # Setup mask
        if mask_center is not None and mask_radius is not None:
            normalized_center = self._normalize_mask_center(mask_center)
            self.mask_center = normalized_center
            self.mask = self._build_mask(normalized_center, mask_radius)
            logger.info(
                "Created spherical mask: radius={}Å, voxels={}",
                mask_radius,
                int(self.mask.sum().item()),
            )
        else:
            self.mask = torch.ones_like(self.target_map_grid, dtype=torch.bool)

        # Alignment indices (use all atoms by default)
        self.alignment_indices = np.arange(len(pdb_obj.atom_pos))
        self._residue_ids = self._extract_residue_ids(pdb_obj)

    def set_pdb_obj(self, pdb_obj) -> None:
        self.pdb_obj = pdb_obj
        self.alignment_indices = np.arange(len(pdb_obj.atom_pos))
        self._residue_ids = self._extract_residue_ids(pdb_obj)

    def set_mask(
        self,
        mask_center: np.ndarray | None,
        mask_radius: float | None,
    ) -> None:
        self.mask_center = mask_center
        self.mask_radius = mask_radius
        if mask_center is not None and mask_radius is not None:
            normalized_center = self._normalize_mask_center(mask_center)
            self.mask_center = normalized_center
            self.mask = self._build_mask(normalized_center, mask_radius)
            logger.info(
                "Updated spherical mask: radius={}Å, voxels={}",
                mask_radius,
                int(self.mask.sum().item()),
            )
        else:
            self.mask = torch.ones_like(self.target_map_grid, dtype=torch.bool)
            logger.info("Updated spherical mask disabled; using full map")

        if hasattr(self, "_cached_normalized_target"):
            delattr(self, "_cached_normalized_target")

    def _build_mask(
        self,
        mask_center: np.ndarray | torch.Tensor,
        mask_radius: float,
    ) -> torch.Tensor:
        center_np = (
            mask_center.detach().cpu().numpy()
            if isinstance(mask_center, torch.Tensor)
            else np.asarray(mask_center, dtype=np.float32)
        )
        try:
            mask_np = create_spherical_mask_for_grid(
                self.target_ccp4_map.grid,
                center_np,
                float(mask_radius),
            )
            return torch.tensor(mask_np, device=self.device, dtype=torch.bool)
        except Exception as exc:
            logger.warning(
                "Failed grid-based mask; falling back to voxel mask: {}",
                exc,
            )
            mask = create_spherical_mask(
                self.grid_shape,
                center_np,
                float(mask_radius),
                self.voxel_size,
                self.device,
            )
            return mask

    def _normalize_mask_center(
        self,
        mask_center: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        center_np = (
            mask_center.detach().cpu().numpy()
            if isinstance(mask_center, torch.Tensor)
            else np.asarray(mask_center, dtype=np.float64)
        )
        try:
            pos = gemmi.Position(
                float(center_np[0]),
                float(center_np[1]),
                float(center_np[2]),
            )
            frac = self.unit_cell.fractionalize(pos)
            frac.x = frac.x % 1.0
            frac.y = frac.y % 1.0
            frac.z = frac.z % 1.0
            ortho = self.unit_cell.orthogonalize(frac)
            return np.array([ortho.x, ortho.y, ortho.z], dtype=np.float32)
        except Exception as exc:
            logger.warning("Failed to normalize mask center; using original: {}", exc)
            return np.asarray(center_np, dtype=np.float32)

    def _extract_residue_ids(self, pdb_obj) -> torch.Tensor | None:
        cra_name = getattr(pdb_obj, "cra_name", None)
        if cra_name is None:
            return None
        residue_ids = []
        for name in cra_name:
            match = re.search(r"-(\d+)-", str(name))
            if match is None:
                return None
            residue_ids.append(int(match.group(1)))
        return torch.tensor(residue_ids, device=self.device, dtype=torch.long)

    def _apply_residue_gradient_mask(
        self,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if self.mask_center is None or self.mask_radius is None:
            return coordinates
        if self._residue_ids is None:
            logger.warning("Residue ids unavailable; skipping gradient mask")
            return coordinates
        center = torch.as_tensor(
            self.mask_center,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        dists = torch.norm(coordinates - center, dim=-1)
        atom_in = dists <= float(self.mask_radius)

        unique_res, inverse = torch.unique(self._residue_ids, return_inverse=True)
        residue_keep = torch.zeros(
            unique_res.shape[0], device=coordinates.device, dtype=torch.bool
        )
        for idx in range(unique_res.shape[0]):
            residue_keep[idx] = atom_in[inverse == idx].any()

        atom_keep = residue_keep[inverse].to(coordinates.dtype)
        keep_mask = atom_keep.unsqueeze(-1)
        return coordinates * keep_mask + coordinates.detach() * (1 - keep_mask)

    @cached_property
    def normalized_target(self) -> torch.Tensor:
        """Get normalized target map (computed once and cached)."""
        return normalize_map(self.target_map_grid, self.mask, method="zscore")

    @timed
    @gpu_memory_tracked
    def model_to_map(
        self,
        coordinates: torch.Tensor,
        structure_factor_calc,
    ) -> torch.Tensor:
        """Generate normalized map from model coordinates.

        Args:
            coordinates: Atomic coordinates [N, 3]
            structure_factor_calc: Structure factor calculator object

        Returns:
            Normalized map grid [Dz, Dy, Dx]
        """
        # Calculate structure factors
        masked_coords = self._apply_residue_gradient_mask(coordinates)
        structure_factor_calc.calc_fprotein(masked_coords, Return=True)
        # Convert to real space via FFT
        from SFC_Torch.mask import reciprocal_grid
        from SFC_Torch.symmetry import expand_to_p1

        Hp1_array, Fp1_tensor = expand_to_p1(
            structure_factor_calc.space_group,
            structure_factor_calc.Hasu_array,
            structure_factor_calc.Fprotein_asu,
            dmin_mask=None,
            unitcell=structure_factor_calc.unit_cell,
            anomalous=structure_factor_calc.anomalous,
        )
        rs_grid = reciprocal_grid(Hp1_array, Fp1_tensor, structure_factor_calc.gridsize)

        map_grid = torch.real(torch.fft.fftn(rs_grid, dim=(-3, -2, -1)))

        # Normalize
        map_normalized = normalize_map(map_grid, self.mask, method="zscore")

        return map_normalized

    def compute(
        self,
        coordinates: torch.Tensor,
        structure_factor_calc: Any = None,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Compute loss value.

        Args:
            coordinates: Atomic coordinates [N, 3]
            structure_factor_calc: Structure factor calculator
            return_metadata: Whether to return additional metadata

        Returns:
            Loss value or tuple of (loss, metadata_dict)
        """
        result: torch.Tensor | tuple[torch.Tensor, dict]
        if self.loss_type == "cc":
            result = self._compute_cc_loss(coordinates, structure_factor_calc)
        elif self.loss_type == "l2":
            result = self._compute_l2_loss(
                coordinates, structure_factor_calc, return_metadata
            )
        elif self.loss_type == "sinkhorn":
            result = self._compute_sinkhorn_loss(coordinates, structure_factor_calc)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return result

    def _compute_cc_loss(
        self,
        coordinates: torch.Tensor,
        sfc,
    ) -> torch.Tensor:
        """Compute negative correlation coefficient loss."""
        model_map = self.model_to_map(coordinates, sfc)

        # Extract masked regions
        target_masked = self.normalized_target[self.mask]
        model_masked = model_map[self.mask]

        # Compute correlation
        correlation = torch.corrcoef(torch.stack([target_masked, model_masked]))[0, 1]

        return -correlation

    def _compute_l2_loss(
        self,
        coordinates: torch.Tensor,
        sfc,
        return_metadata: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Compute L2 loss on normalized maps."""
        model_map = self.model_to_map(coordinates, sfc)

        # Extract masked regions
        target_masked = self.normalized_target[self.mask]
        model_masked = model_map[self.mask]

        # Check sufficient voxels
        if target_masked.numel() < 50:
            loss = torch.tensor(1e6, device=self.device, dtype=torch.float32)
            if return_metadata:
                return loss, {"error": "insufficient_voxels"}
            return loss

        # Compute L2 loss
        l2_loss = torch.mean((target_masked - model_masked) ** 2)

        if return_metadata:
            # Compute RSCC map for B-factor estimation
            target_smoothed = gaussian_smooth_3d(
                self.normalized_target, sigma_angstrom=4.0, voxel_size=self.voxel_size
            )
            model_smoothed = gaussian_smooth_3d(
                model_map, sigma_angstrom=4.0, voxel_size=self.voxel_size
            )

            # Compute local correlation
            rscc_map = self._compute_rscc_map(target_smoothed, model_smoothed.detach())

            # Interpolate to atom positions
            atom_cc = self._interpolate_to_atoms(rscc_map, coordinates, sfc)

            # Convert CC to pseudo B-factors
            dmin = getattr(sfc, "dmin", 2.0)
            rscc_bfactors = self._cc_to_bfactor(atom_cc, dmin)

            # Metadata should only contain scalars for logging
            metadata = {
                "mean_cc": atom_cc.mean().item(),
                "std_cc": atom_cc.std().item(),
                "min_cc": atom_cc.min().item(),
                "max_cc": atom_cc.max().item(),
                "mean_bfactor": rscc_bfactors.mean().item(),
                "std_bfactor": rscc_bfactors.std().item(),
            }

            return l2_loss, metadata

        return l2_loss

    def _compute_sinkhorn_loss(
        self,
        coordinates: torch.Tensor,
        sfc,
        blurs: tuple[float, ...] = (3.0, 2.0, 1.0, 0.5),
    ) -> torch.Tensor:
        """Compute Sinkhorn (optimal transport) loss.

        Args:
            coordinates: Atomic coordinates
            sfc: Structure factor calculator
            blurs: Multiscale blur schedule in Angstroms

        Returns:
            Sinkhorn loss value
        """
        model_map = self.model_to_map(coordinates, sfc)

        # Create coordinate grid for voxels
        coords_3d = self._make_voxel_coordinates()

        # Get nonnegative masked densities
        if self.mask is not None:
            active = self.mask.reshape(-1)
        else:
            active = torch.ones(
                self.target_map_grid.numel(),
                dtype=torch.bool,
                device=self.device,
            )

        # Extract and normalize densities
        target_density = torch.clamp(self.target_map_grid.reshape(-1)[active], min=0.0)
        model_density = torch.clamp(model_map.reshape(-1)[active], min=0.0)
        coords = coords_3d[active]

        # Normalize to probability distributions
        eps = 1e-12
        target_density = target_density / (target_density.sum() + eps)
        model_density = model_density / (model_density.sum() + eps)

        # Compute Sinkhorn divergence over multiple scales
        total_loss = torch.tensor(0.0, device=self.device)
        for blur in blurs:
            sinkhorn = SamplesLoss(
                loss="sinkhorn",
                p=2,
                blur=blur,
                backend="multiscale",
                debias=True,
            )
            loss_val = sinkhorn(
                target_density.unsqueeze(-1) * coords,
                model_density.unsqueeze(-1) * coords,
            )
            total_loss += loss_val

        return total_loss / len(blurs)

    def _compute_rscc_map(
        self,
        map1: torch.Tensor,
        map2: torch.Tensor,
        window_radius: int = 3,
    ) -> torch.Tensor:
        """Compute real-space correlation coefficient map.

        Args:
            map1: First map [Dz, Dy, Dx]
            map2: Second map [Dz, Dy, Dx]
            window_radius: Radius of correlation window

        Returns:
            RSCC map [Dz, Dy, Dx]
        """
        # Use efficient convolution-based approach
        # For simplicity, returning a global estimate here
        # Production version should use sliding window
        corr = torch.corrcoef(torch.stack([map1.flatten(), map2.flatten()]))[0, 1]
        return torch.full_like(map1, corr.item())

    def _interpolate_to_atoms(
        self,
        map_grid: torch.Tensor,
        coordinates: torch.Tensor,
        sfc,
    ) -> torch.Tensor:
        """Interpolate map values to atomic positions.

        Args:
            map_grid: Map to interpolate [Dz, Dy, Dx]
            coordinates: Atomic coordinates [N, 3]
            sfc: Structure factor calculator (for fractional conversion)

        Returns:
            Interpolated values at atomic positions [N]
        """
        # Convert orthogonal to fractional coordinates
        # This is a simplified version - production should use proper conversion
        frac_coords = coordinates / torch.tensor(
            [self.unit_cell.a, self.unit_cell.b, self.unit_cell.c], device=self.device
        )

        # Convert to grid indices
        grid_indices = frac_coords * torch.tensor(self.grid_shape, device=self.device)

        # Trilinear interpolation (simplified)
        grid_indices_clamped = torch.clamp(
            grid_indices,
            torch.zeros(3, device=self.device),
            torch.tensor(self.grid_shape, device=self.device) - 1,
        )

        # Simple nearest neighbor for now
        indices = grid_indices_clamped.long()
        values = map_grid[indices[:, 0], indices[:, 1], indices[:, 2]]

        return values

    def _cc_to_bfactor(
        self,
        cc_values: torch.Tensor,
        dmin: float,
    ) -> torch.Tensor:
        """Convert correlation coefficients to pseudo B-factors.

        Args:
            cc_values: Correlation coefficient values [N]
            dmin: Minimum resolution in Angstroms

        Returns:
            Pseudo B-factors [N]
        """
        # Empirical conversion: B = -8π²d²ln(CC)
        cc_clamped = torch.clamp(cc_values, min=0.01, max=0.99)
        b_factors = -8.0 * (torch.pi**2) * (dmin**2) * torch.log(cc_clamped)
        return torch.clamp(b_factors, min=10.0, max=200.0)

    def _create_atomic_mask(
        self,
        coordinates: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        """Create boolean mask around atomic positions.

        Args:
            coordinates: Atomic coordinates [N, 3]
            radius: Mask radius in Angstroms

        Returns:
            Boolean mask [Dz, Dy, Dx]
        """
        # Create coordinate grid
        coords_3d = self._make_voxel_coordinates()

        # Initialize mask
        mask = torch.zeros(self.grid_shape, dtype=torch.bool, device=self.device)

        # Mark voxels within radius of any atom
        for atom_pos in coordinates:
            distances = torch.norm(coords_3d - atom_pos, dim=-1)
            mask = mask | (distances <= radius)

        return mask

    def _make_voxel_coordinates(self) -> torch.Tensor:
        """Create 3D coordinate array for all voxels.

        Returns:
            Coordinates [Dz, Dy, Dx, 3]
        """
        nz, ny, nx = self.grid_shape
        vz, vy, vx = self.voxel_size

        z = torch.arange(nz, device=self.device, dtype=torch.float32) * vz
        y = torch.arange(ny, device=self.device, dtype=torch.float32) * vy
        x = torch.arange(nx, device=self.device, dtype=torch.float32) * vx

        Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack([X, Y, Z], dim=-1)

        return coords
