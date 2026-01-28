"""PDB trajectory writer for refinement using mdtraj."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Optional import for mdtraj
try:
    import mdtraj as md
    from mdtraj.formats import PDBTrajectoryFile
    MDTRAJ_AVAILABLE = True
    logger.info(f"mdtraj successfully imported, version: {md.__version__}")
except ImportError as e:
    MDTRAJ_AVAILABLE = False
    md = None
    PDBTrajectoryFile = None
    logger.warning(f"mdtraj import failed: {e}")


class TrajectoryWriter:
    """Handles saving PDB trajectories during refinement using mdtraj."""
    
    def __init__(
        self,
        output_dir: Path,
        pdb_template_path: str | Path,
        save_interval: int = 10,
        wandb_logger = None,
    ):
        """Initialize trajectory writer.
        
        Args:
            output_dir: Output directory
            pdb_template_path: Path to PDB template file
            save_interval: Save interval for individual frames (not used with streaming)
            wandb_logger: Optional WandbLogger for real-time streaming
        """
        self.output_dir = Path(output_dir) / "trajectory"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pdb_template_path = Path(pdb_template_path)
        self.save_interval = save_interval
        self.wandb_logger = wandb_logger
        
        # Track writers per run_id
        self.traj_writers = {}
        self.mdtraj_template = None
        self.topology = None
        
        logger.info(f"TrajectoryWriter initialization:")
        logger.info(f"  MDTRAJ_AVAILABLE: {MDTRAJ_AVAILABLE}")
        logger.info(f"  pdb_template_path: {self.pdb_template_path}")
        logger.info(f"  output_dir: {self.output_dir}")
        
        # Load mdtraj template
        if MDTRAJ_AVAILABLE:
            if not self.pdb_template_path.exists():
                logger.error(f"PDB template file not found: {self.pdb_template_path}")
                return
                
            try:
                logger.info(f"📂 Loading mdtraj template from: {self.pdb_template_path}")
                self.mdtraj_template = md.load_pdb(str(self.pdb_template_path))
                self.topology = self.mdtraj_template.topology
                n_atoms = self.mdtraj_template.n_atoms
                logger.info(f"✓ Successfully loaded template with {n_atoms} atoms")
                logger.info(f"  Number of residues: {self.topology.n_residues}")
            except Exception as e:
                logger.error(f"✗ Failed to load PDB template: {e}")
                logger.exception(e)
                self.mdtraj_template = None
                self.topology = None
        else:
            logger.warning("mdtraj not available - trajectory saving disabled")
    
    def save_frame(
        self,
        coordinates: torch.Tensor | np.ndarray,
        iteration: int,
        run_id: str,
        b_factors: torch.Tensor | np.ndarray | None = None,
        loss: float | None = None,
    ) -> None:
        """Save a single frame to trajectory file.
        
        Args:
            coordinates: Atomic coordinates [N, 3] in Angstroms
            iteration: Current iteration number
            run_id: Current run identifier
            b_factors: Optional B-factors [N] (not used currently)
            loss: Optional loss value (not used currently)
        """
        if not MDTRAJ_AVAILABLE or self.topology is None:
            logger.debug("mdtraj not available, skipping frame save")
            return
        
        # Convert to numpy
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()
        
        # Validate atom count matches template
        n_coords = coordinates.shape[0]
        n_template = self.topology.n_atoms
        
        if n_coords != n_template:
            logger.error(
                f"❌ ATOM COUNT MISMATCH: coordinates have {n_coords} atoms, "
                f"but template has {n_template} atoms."
            )
            return
        
        # Initialize writer for this run_id if not exists
        if run_id not in self.traj_writers:
            traj_path = self.output_dir / f"{run_id}_refinement_trajectory.pdb"
            try:
                self.traj_writers[run_id] = PDBTrajectoryFile(str(traj_path), mode="w")
                logger.info(f"✓ Opened trajectory file: {traj_path}")
            except Exception as e:
                logger.error(f"Failed to open trajectory writer: {e}")
                return
        
        writer = self.traj_writers[run_id]
        
        # Write frame (PDBTrajectoryFile expects Angstroms, not nm)
        try:
            coords_angstrom = coordinates.reshape(-1, 3)
            writer.write(coords_angstrom, self.topology, modelIndex=iteration)
            
            # Flush to ensure data is written immediately
            if hasattr(writer, "_file") and hasattr(writer._file, "flush"):
                writer._file.flush()
            
            logger.debug(f"✓ Saved frame for run {run_id}, iteration {iteration}")
            
            # Real-time wandb logging
            if self.wandb_logger is not None and MDTRAJ_AVAILABLE:
                try:
                    # Create temp single-frame PDB for wandb
                    temp_frame_path = self.output_dir / f"_temp_{run_id}_frame.pdb"
                    coords_nm = coordinates.reshape(1, -1, 3) / 10.0  # mdtraj uses nm internally
                    temp_traj = md.Trajectory(coords_nm, self.topology)
                    temp_traj.save_pdb(str(temp_frame_path))
                    
                    # Log to wandb
                    import wandb
                    wandb.log({
                        f"trajectory_{run_id}_live": wandb.Molecule(str(temp_frame_path)),
                        "iteration": iteration,
                    })
                    
                    # Clean up
                    temp_frame_path.unlink()
                except Exception as wandb_err:
                    logger.debug(f"Could not stream frame to wandb: {wandb_err}")
                    logger.debug(f"Could not stream frame to wandb: {wandb_err}")
            
        except Exception as e:
            logger.error(f"Failed to write trajectory frame: {e}")
            logger.exception(e)
    
    def save_best(
        self,
        coordinates: np.ndarray | torch.Tensor,
        run_id: str,
        iteration: int,
        b_factors: np.ndarray | None = None,
    ) -> None:
        """Save a single best structure as a PDB file.
        
        Args:
            coordinates: Atomic coordinates in Angstroms, shape (n_atoms, 3)
            run_id: Unique identifier for this refinement run
            iteration: Iteration number when this best was found
            b_factors: Optional B-factors to write to PDB file
        """
        # Convert torch tensor to numpy if needed
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()
        
        # Validate atom count matches template
        n_coords = coordinates.shape[0]
        n_template = self.topology.n_atoms
        
        if n_coords != n_template:
            logger.error(
                f"❌ ATOM COUNT MISMATCH for best PDB: coordinates have {n_coords} atoms, "
                f"but template has {n_template} atoms."
            )
            return
        
        # Create output path
        best_path = self.output_dir / f"best_{run_id}_iter{iteration}.pdb"
        
        try:
            # Convert coordinates to nm (mdtraj uses nm)
            coords_nm = coordinates.reshape(1, -1, 3) / 10.0  # Shape: (1, n_atoms, 3)
            
            # Create a single-frame trajectory
            traj = md.Trajectory(coords_nm, self.topology)
            
            # Set B-factors if provided
            if b_factors is not None:
                if isinstance(b_factors, torch.Tensor):
                    b_factors = b_factors.detach().cpu().numpy()
                traj.bfactors = b_factors.reshape(-1, 1)  # Shape: (n_atoms, 1)
            
            # Save to PDB
            traj.save_pdb(str(best_path))
            logger.info(f"✓ Saved best structure to {best_path}")
            
        except Exception as e:
            logger.error(f"Failed to save best PDB: {e}")
            logger.exception(e)
    
    def close(self, run_id: str | None = None) -> None:
        """Close trajectory writer(s).
        
        Args:
            run_id: Specific run to close, or None to close all
        """
        if run_id is not None:
            if run_id in self.traj_writers:
                try:
                    self.traj_writers[run_id].close()
                    logger.info(f"✓ Closed trajectory writer for run {run_id}")
                except Exception as e:
                    logger.warning(f"Error closing trajectory writer for {run_id}: {e}")
                finally:
                    del self.traj_writers[run_id]
        else:
            # Close all writers
            for rid, writer in list(self.traj_writers.items()):
                try:
                    writer.close()
                    logger.info(f"✓ Closed trajectory writer for run {rid}")
                except Exception as e:
                    logger.warning(f"Error closing trajectory writer for {rid}: {e}")
            self.traj_writers.clear()
    
    def __del__(self):
        """Cleanup: close any open writers."""
        try:
            self.close()
        except Exception:
            pass
