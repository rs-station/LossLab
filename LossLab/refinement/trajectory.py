"""PDB trajectory writer for refinement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

# Optional import - ADD LOGGING HERE
try:
    import mdtraj as md
    MDTRAJ_AVAILABLE = True
    logger.info(f"mdtraj successfully imported, version: {md.__version__}")  # ← ADD THIS
except ImportError as e:
    MDTRAJ_AVAILABLE = False
    md = None
    logger.warning(f"mdtraj import failed: {e}")  # ← ADD THIS


class TrajectoryWriter:
    """Handles saving PDB trajectories during refinement."""
    
    def __init__(
        self,
        output_dir: Path,
        pdb_template_path: str | Path,
        save_interval: int = 10,
    ):
        """Initialize trajectory writer."""
        self.output_dir = Path(output_dir) / "trajectory"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pdb_template_path = Path(pdb_template_path)
        self.save_interval = save_interval
        self.save_individual = False  # Can be enabled to save individual frame PDBs
        self.frames: dict[str, list[dict]] = {}
        self.frame_metadata: list[dict] = []
        
        # ADD ALL THIS LOGGING
        logger.info(f"TrajectoryWriter initialization:")
        logger.info(f"  MDTRAJ_AVAILABLE: {MDTRAJ_AVAILABLE}")
        logger.info(f"  pdb_template_path: {self.pdb_template_path}")
        logger.info(f"  pdb_template_path exists: {self.pdb_template_path.exists()}")
        logger.info(f"  output_dir: {self.output_dir}")
        
        # Load mdtraj template if available
        if MDTRAJ_AVAILABLE:
            try:
                logger.info(f"Attempting to load mdtraj template...")
                self.mdtraj_template = md.load_pdb(str(self.pdb_template_path))
                self.topology = self.mdtraj_template.topology
                logger.info(f"✓ Successfully loaded template with {self.mdtraj_template.n_atoms} atoms")
            except Exception as e:
                logger.error(f"✗ Failed to load PDB template: {e}")
                logger.exception(e)
                self.mdtraj_template = None
                self.topology = None
        else:
            logger.warning("mdtraj not available - trajectory saving disabled")
            self.mdtraj_template = None
            self.topology = None
        
        logger.info(f"TrajectoryWriter initialized. mdtraj_template is: {self.mdtraj_template}")
    
    def save_frame(
        self,
        coordinates: torch.Tensor | np.ndarray,
        iteration: int,
        run_id: str,
        b_factors: torch.Tensor | np.ndarray | None = None,
        loss: float | None = None,
    ) -> None:
        """Save a single frame to trajectory.
        
        Args:
            coordinates: Atomic coordinates [N, 3] in Angstroms
            iteration: Current iteration number
            run_id: Current run identifier
            b_factors: Optional B-factors [N]
            loss: Optional loss value
        """
        if not MDTRAJ_AVAILABLE or self.topology is None:
            logger.debug("mdtraj not available, skipping frame save")
            return
            
        # Convert to numpy
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()
        if isinstance(b_factors, torch.Tensor):
            b_factors = b_factors.detach().cpu().numpy()
        
        # Validate atom count matches template
        n_coords = coordinates.shape[0]
        n_template = self.topology.n_atoms
        if n_coords != n_template:
            logger.warning(
                f"Atom count mismatch: coordinates have {n_coords} atoms, "
                f"template has {n_template} atoms. Truncating to {n_template}."
            )
            coordinates = coordinates[:n_template]
            if b_factors is not None:
                b_factors = b_factors[:n_template]
        
        # Store frame
        frame_data = {
            "coordinates": coordinates.copy(),
            "b_factors": b_factors.copy() if b_factors is not None else None,
            "iteration": iteration,
            "run_id": run_id,
            "loss": loss,
        }
        
        # Initialize list for this run_id if needed
        if run_id not in self.frames:
            self.frames[run_id] = []
        
        self.frames[run_id].append(frame_data)
        self.frame_metadata.append({
            "iteration": iteration,
            "run_id": run_id,
            "loss": loss,
        })
        
        # Save individual PDB if requested
        if self.save_individual and iteration % self.save_interval == 0:
            filename = f"{run_id}_{iteration:04d}.pdb"
            self._write_pdb(frame_data, self.output_dir / filename)
    
    def save_best(
        self,
        coordinates: torch.Tensor | np.ndarray,
        run_id: str,
        iteration: int,
        b_factors: torch.Tensor | np.ndarray | None = None,
        loss: float | None = None,
    ) -> None:
        """Save best model.
        
        Args:
            coordinates: Best atomic coordinates [N, 3] in Angstroms
            run_id: Run identifier
            iteration: Iteration number
            b_factors: Optional B-factors [N]
            loss: Loss value
        """
        if not MDTRAJ_AVAILABLE or self.topology is None:
            logger.debug("mdtraj not available, skipping best save")
            return
            
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()
        if isinstance(b_factors, torch.Tensor):
            b_factors = b_factors.detach().cpu().numpy()
        
        # Validate and truncate if needed
        n_coords = coordinates.shape[0]
        n_template = self.topology.n_atoms
        if n_coords != n_template:
            logger.warning(
                f"Atom count mismatch in save_best: coordinates have {n_coords} atoms, "
                f"template has {n_template} atoms. Truncating to {n_template}."
            )
            coordinates = coordinates[:n_template]
            if b_factors is not None:
                b_factors = b_factors[:n_template]
        
        frame_data = {
            "coordinates": coordinates,
            "b_factors": b_factors,
            "iteration": iteration,
            "run_id": run_id,
            "loss": loss,
        }
        
        filename = f"best_{run_id}_{iteration:04d}.pdb"
        self._write_pdb(frame_data, self.output_dir / filename)
        
        # Also save as "best.pdb" (overwrites each time)
        self._write_pdb(frame_data, self.output_dir / "best.pdb")
    
    def write_trajectory(self, filename: str = "trajectory.pdb") -> None:
        """Write all frames as multi-model PDB.
        
        Args:
            filename: Output filename
        """
        if not self.frames:
            return
            
        if not MDTRAJ_AVAILABLE or self.topology is None:
            logger.warning("mdtraj not available, cannot write trajectory")
            return
        
        output_path = self.output_dir / filename
        
        try:
            # Flatten all frames from all runs
            all_frames = []
            for run_id, run_frames in self.frames.items():
                all_frames.extend(run_frames)
            
            # Write multi-model PDB using mdtraj
            with PDBTrajectoryFile(str(output_path), mode="w") as traj_writer:
                for frame_data in all_frames:
                    coords_nm = frame_data["coordinates"] / 10.0  # Angstroms to nm
                    traj_writer.write(
                        coords_nm.reshape(-1, 3),
                        self.topology,
                        modelIndex=frame_data["iteration"]
                    )
            logger.info(f"Saved trajectory ({len(all_frames)} frames): {output_path}")
        except Exception as e:
            logger.error(f"Failed to write trajectory: {e}")
    
    def _write_pdb(self, frame_data: dict, output_path: Path) -> None:
        """Write single frame to PDB file using mdtraj.
        
        Args:
            frame_data: Frame data dictionary
            output_path: Output file path
        """
        if not MDTRAJ_AVAILABLE or self.topology is None:
            logger.warning("mdtraj not available, cannot write PDB")
            return
            
        try:
            # Convert coordinates to nm (mdtraj expects nm)
            coords_nm = frame_data["coordinates"] / 10.0  # Angstroms to nm
            coords_nm = coords_nm.reshape(1, -1, 3)  # Shape: (1, n_atoms, 3)
            
            # Create trajectory and save
            traj = md.Trajectory(coords_nm, self.topology)
            
            # Set B-factors if available
            if frame_data.get("b_factors") is not None:
                # mdtraj stores bfactors as topology attribute
                for atom, bfactor in zip(traj.topology.atoms, frame_data["b_factors"]):
                    atom.bfactor = float(bfactor)
            
            traj.save_pdb(str(output_path))
        except Exception as e:
            logger.error(f"Failed to write PDB to {output_path}: {e}")
    
    def clear(self) -> None:
        """Clear stored frames to free memory."""
        self.frames.clear()
        self.frame_metadata.clear()
        logger.debug("Cleared trajectory frames")
