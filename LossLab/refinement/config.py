"""Configuration for refinement runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml


@dataclass
class RefinementConfig:
    """Configuration for coordinate refinement.

    Example:
        >>> config = RefinementConfig(
        ...     num_iterations=100,
        ...     learning_rate_additive=1e-3,
        ...     output_dir="./output",
        ... )
        >>> config.to_yaml("config.yaml")
    """

    # Optimization parameters
    num_iterations: int = 100
    num_runs: int = 1
    learning_rate_additive: float = 1e-3
    learning_rate_multiplicative: float = 1e-3
    weight_decay: float | None = None
    optimizer_type: Literal["adam", "sgd", "lbfgs"] = "adam"

    # Model parameters
    init_recycling: int = 3
    device: str = "cuda:0"

    # Loss parameters
    loss_type: Literal["cc", "l2", "sinkhorn", "mse"] = "l2"
    plddt_weight: float = 0.0
    l2_weight: float = 0.0

    # Masking
    mask_center: np.ndarray | None = None
    mask_radius: float | None = None
    penalty_center: np.ndarray | None = None
    penalty_radius: float | None = None
    penalty_weight: float = 100.0

    # Domain/segment information
    domain_segments: list[tuple[int, int]] | None = None

    # Rigid body refinement
    use_rigid_body_refinement: bool = True
    rbr_learning_rate: float = 1e-3
    rbr_use_lbfgs: bool = True

    # Output settings
    output_dir: str | Path = "./output"
    run_note: str = "refinement"
    save_every_n_iterations: int = 50
    save_best_pdb: bool = True
    save_trajectory_pdb: bool = True
    save_trajectory_interval: int = 1
    save_model_maps: bool = False

    # Weights & Biases logging
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_tags: list[str] | None = None
    wandb_notes: str | None = None

    # Early stopping
    early_stopping_patience: int = 150
    early_stopping_min_delta: float = 0.0001

    # Logging
    verbose: bool = True
    log_metrics: bool = True

    # Alignment
    use_confidence_alignment_weights: bool = False

    # Starting parameters
    starting_bias_path: str | Path | None = None
    starting_weights_path: str | Path | None = None

    def __post_init__(self):
        """Validate and process configuration."""
        self.output_dir = Path(self.output_dir)

        if self.starting_bias_path:
            self.starting_bias_path = Path(self.starting_bias_path)
        if self.starting_weights_path:
            self.starting_weights_path = Path(self.starting_weights_path)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable dict
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
            else:
                config_dict[key] = value

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RefinementConfig":
        """Load configuration from YAML file.

        Args:
            path: Input file path

        Returns:
            RefinementConfig instance
        """
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        # Convert lists back to numpy arrays if needed
        if config_dict.get("mask_center") is not None:
            config_dict["mask_center"] = np.array(config_dict["mask_center"])
        if config_dict.get("penalty_center") is not None:
            config_dict["penalty_center"] = np.array(config_dict["penalty_center"])

        return cls(**config_dict)
