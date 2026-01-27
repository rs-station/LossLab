"""Checkpoint management for saving/loading refinement state."""

from pathlib import Path
from typing import Any

import torch
from loguru import logger


class CheckpointManager:
    """Manage checkpoints during refinement.

    Example:
        >>> manager = CheckpointManager(output_dir="./output")
        >>> manager.save_checkpoint(
        ...     iteration=100,
        ...     run_id="A",
        ...     loss=1.23,
        ...     msa_bias=bias_tensor,
        ...     feat_weights=weights_tensor,
        ... )
    """

    def __init__(
        self,
        output_dir: str | Path,
        save_best_only: bool = False,
    ):
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory to save checkpoints
            save_best_only: Only save when new best is achieved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only

        self.best_loss = float("inf")
        self.best_checkpoint_info = {}

    def save_checkpoint(
        self,
        iteration: int,
        run_id: str,
        loss: float,
        **state: Any,
    ) -> bool:
        """Save checkpoint with current state.

        Args:
            iteration: Current iteration
            run_id: Run identifier
            loss: Current loss value
            **state: Additional state to save (tensors, etc.)

        Returns:
            Whether checkpoint was saved
        """
        # Check if this is a new best
        is_best = loss < self.best_loss

        if self.save_best_only and not is_best:
            return False

        # Prepare checkpoint dictionary
        checkpoint = {
            "iteration": iteration,
            "run_id": run_id,
            "loss": loss,
        }
        checkpoint.update(state)

        # Save tensors separately for clarity
        tensor_state = {}
        other_state = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                tensor_state[key] = value.detach().cpu()
            else:
                other_state[key] = value

        # Save tensors
        if tensor_state:
            for key, tensor in tensor_state.items():
                tensor_path = self.output_dir / f"{run_id}_{iteration}_{key}.pt"
                torch.save(tensor, tensor_path)

        # Update best if needed
        if is_best:
            self.best_loss = loss
            self.best_checkpoint_info = {
                "iteration": iteration,
                "run_id": run_id,
                "loss": loss,
            }

            # Save best markers
            for key, tensor in tensor_state.items():
                best_path = self.output_dir / f"best_{key}_{run_id}_{iteration}.pt"
                torch.save(tensor, best_path)

            logger.info(
                f"New best checkpoint: loss={loss:.6f}, run={run_id}, iter={iteration}"
            )

        return True

    def load_checkpoint(
        self,
        run_id: str,
        iteration: int,
        tensor_keys: list[str],
    ) -> dict[str, torch.Tensor]:
        """Load checkpoint from disk.

        Args:
            run_id: Run identifier
            iteration: Iteration number
            tensor_keys: Keys of tensors to load

        Returns:
            Dictionary of loaded tensors
        """
        loaded = {}
        for key in tensor_keys:
            tensor_path = self.output_dir / f"{run_id}_{iteration}_{key}.pt"
            if tensor_path.exists():
                loaded[key] = torch.load(tensor_path)
            else:
                logger.warning(f"Checkpoint not found: {tensor_path}")

        return loaded

    def load_best(
        self,
        tensor_keys: list[str],
    ) -> dict[str, torch.Tensor] | None:
        """Load the best checkpoint.

        Args:
            tensor_keys: Keys of tensors to load

        Returns:
            Dictionary of loaded tensors or None if no best exists
        """
        if not self.best_checkpoint_info:
            logger.warning("No best checkpoint recorded")
            return None

        run_id = self.best_checkpoint_info["run_id"]
        iteration = self.best_checkpoint_info["iteration"]

        return self.load_checkpoint(run_id, iteration, tensor_keys)

    def get_best_info(self) -> dict[str, Any]:
        """Get information about best checkpoint.

        Returns:
            Dictionary with best checkpoint info
        """
        return self.best_checkpoint_info.copy()
