"""Weights & Biases experiment tracking integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

# Optional import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandbLogger:
    """Weights & Biases experiment logger.
    
    Provides optional experiment tracking with wandb. If wandb is not
    installed, this class becomes a no-op.
    
    Example:
        >>> logger = WandbLogger(
        ...     project="protein-refinement",
        ...     name="x395_refinement",
        ...     config=config,
        ... )
        >>> 
        >>> # During training
        >>> logger.log({
        ...     "iteration": i,
        ...     "loss": loss,
        ...     "mean_cc": cc,
        ... })
        >>> 
        >>> # Save artifacts
        >>> logger.log_pdb("best.pdb")
        >>> logger.finish()
    """
    
    def __init__(
        self,
        project: str,
        entity: str | None = None,
        name: str | None = None,
        config: dict | Any | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        enabled: bool = True,
    ):
        """Initialize wandb logger.
        
        Args:
            project: W&B project name
            entity: W&B entity (team/organization name)
            name: Run name (auto-generated if None)
            config: Configuration dict or object to log
            tags: List of tags for this run
            notes: Markdown notes about this run
            enabled: Whether to actually log (useful for debugging)
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        
        if not WANDB_AVAILABLE:
            logger.warning(
                "wandb not installed. Install with: pip install wandb\n"
                "Experiment tracking disabled."
            )
            return
        
        if not self.enabled:
            logger.info("W&B logging disabled")
            return
        
        # Convert config to dict if needed
        if config is not None and hasattr(config, "__dict__"):
            config = {
                k: v for k, v in config.__dict__.items()
                if not k.startswith("_")
            }
        
        # Initialize wandb run
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
        )
        
        logger.info(f"W&B run initialized: {self.run.url}")
    
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (iteration)
        """
        if not self.enabled or self.run is None:
            return
        
        # Convert tensors to scalars
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                value = value.item() if value.size == 1 else value
            processed_metrics[key] = value
        
        wandb.log(processed_metrics, step=step)
    
    def log_pdb(
        self,
        pdb_path: str | Path,
        name: str | None = None,
    ) -> None:
        """Log PDB file as artifact.
        
        Args:
            pdb_path: Path to PDB file
            name: Optional artifact name
        """
        if not self.enabled or self.run is None:
            return
        
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            logger.warning(f"PDB file not found: {pdb_path}")
            return
        
        artifact_name = name or pdb_path.stem
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(str(pdb_path))
        self.run.log_artifact(artifact)
        
        logger.debug(f"Logged PDB artifact: {artifact_name}")
    
    def log_coordinates(
        self,
        coordinates: torch.Tensor | np.ndarray,
        name: str = "coordinates",
    ) -> None:
        """Log coordinates as artifact.
        
        Args:
            coordinates: Coordinates tensor [N, 3]
            name: Artifact name
        """
        if not self.enabled or self.run is None:
            return
        
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()
        
        # Save as numpy array
        artifact = wandb.Artifact(name, type="coordinates")
        with artifact.new_file(f"{name}.npy", mode="wb") as f:
            np.save(f, coordinates)
        self.run.log_artifact(artifact)
    
    def log_config_file(self, config_path: str | Path) -> None:
        """Log configuration file as artifact.
        
        Args:
            config_path: Path to config file (YAML, JSON, etc.)
        """
        if not self.enabled or self.run is None:
            return
        
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        artifact = wandb.Artifact("config", type="config")
        artifact.add_file(str(config_path))
        self.run.log_artifact(artifact)
    
    def watch_model(
        self,
        model: torch.nn.Module,
        log: str = "gradients",
        log_freq: int = 100,
    ) -> None:
        """Watch model for gradient/parameter tracking.
        
        Args:
            model: PyTorch model to watch
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency
        """
        if not self.enabled or self.run is None:
            return
        
        wandb.watch(model, log=log, log_freq=log_freq)
    
    def finish(self) -> None:
        """Finish wandb run."""
        if not self.enabled or self.run is None:
            return
        
        self.run.finish()
        logger.info("W&B run finished")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
    
    def log_artifact(
        self,
        file_path: str,
        name: str,
        artifact_type: str = "dataset",
    ) -> None:
        """Log a file as a wandb artifact."""
        if not self.enabled or self.run is None:
            logger.warning(f"Cannot log artifact {name} - wandb not enabled or run is None")
            return
        
        try:
            logger.info(f"Creating artifact: {name} (type: {artifact_type})")
            artifact = wandb.Artifact(name=name, type=artifact_type)
            logger.info(f"Adding file to artifact: {file_path}")
            artifact.add_file(file_path)
            logger.info(f"Logging artifact to wandb...")
            self.run.log_artifact(artifact)
            logger.info(f"Successfully logged artifact: {name}")
        except Exception as e:
            logger.error(f"Failed to log artifact {name}: {e}")
