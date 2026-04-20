"""Refinement engine for coordinate optimization."""

import contextlib
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from tqdm import tqdm

from losslab.losses.base import BaseLoss
from losslab.refinement.checkpoint import CheckpointManager
from losslab.refinement.config import RefinementConfig
from losslab.refinement.metrics import MetricsTracker
from losslab.refinement.trajectory import TrajectoryWriter
from losslab.refinement.wandb_logger import WandbLogger
from losslab.utils.decorators import gpu_memory_tracked, timed
from losslab.utils.geometry import kabsch_align


class EarlyStopper:
    """Early stopping based on loss plateau."""

    def __init__(self, patience: int = 150, min_delta: float = 0.0001):
        """Initialize early stopper.

        Args:
            patience: Number of iterations to wait for improvement
            min_delta: Minimum change to count as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def should_stop(self, loss: float) -> bool:
        """Check if training should stop.

        Args:
            loss: Current loss value

        Returns:
            Whether to stop training
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class RefinementEngine:
    """High-level refinement engine for coordinate optimization.

    Encapsulates the entire refinement workflow.

    The engine is model-agnostic - it only cares about coordinates. All
    model-specific logic (features, recycling, etc.) should be handled
    inside the prediction callback.

    Example:
        >>> # User handles model and features
        >>> class MyPredictor:
        ...     def __init__(self, model, features):
        ...         self.model = model
        ...         self.features = features
        ...         self.state = None  # Model-specific state
        ...
        ...     def __call__(self):
        ...         # User's model logic
        ...         output = self.model(self.features, self.state)
        ...         self.state = output.get('state')
        ...         return output['coordinates']  # Return [N, 3] tensor
        >>>
        >>> # Initialize predictor with YOUR model and features
        >>> predictor = MyPredictor(my_model, my_features)
        >>>
        >>> # Initialize engine (only needs coordinates and loss)
        >>> engine = RefinementEngine(config, loss_fn, structure_factor_calc)
        >>>
        >>> # Run refinement - engine just calls predictor()
        >>> results = engine.run(
        ...     reference_coordinates=ref_coords,
        ...     prediction_callback=predictor,
        ... )
    """

    def __init__(
        self,
        config: RefinementConfig,
        loss_function: BaseLoss,
        structure_factor_calculator: Any,
        rbr_function: Callable | None = None,
        pdb_template: str | Path | None = None,
    ):
        """Initialize refinement engine.

        Args:
            config: Refinement configuration
            loss_function: Loss function instance
            structure_factor_calculator: Structure factor calculator
            rbr_function: Optional rigid body refinement function
            pdb_template: Optional PDB template for trajectory writing
        """
        self.config = config
        self.loss_fn = loss_function
        self.sfc = structure_factor_calculator
        self.rbr_fn = rbr_function
        self.pdb_template = pdb_template

        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.run_note
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.to_yaml(self.output_dir / "config.yaml")

        # Initialize tracking
        self.checkpoint_manager = CheckpointManager(
            self.output_dir, save_best_only=True
        )

        # Initialize trajectory writer if PDB template provided
        self.trajectory_writer = None
        if pdb_template is not None and (
            config.save_best_pdb or config.save_trajectory_pdb
        ):
            logger.info(f"Initializing TrajectoryWriter with template: {pdb_template}")
            logger.info(f"  Type: {type(pdb_template)}")
            logger.info(f"  save_best_pdb: {config.save_best_pdb}")
            logger.info(f"  save_trajectory_pdb: {config.save_trajectory_pdb}")

            # TrajectoryWriter will create its own PDBParser from the template file
            self.trajectory_writer = TrajectoryWriter(
                output_dir=self.output_dir,
                pdb_template_path=pdb_template,
                save_interval=config.save_trajectory_interval,
            )
        else:
            logger.warning(
                "TrajectoryWriter not initialized - no PDB template provided "
                "or saving disabled"
            )
            self.trajectory_writer = None

        # Initialize wandb logger if enabled
        self.wandb_logger: WandbLogger | None = None
        if config.use_wandb:
            self.wandb_logger = WandbLogger(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_name,
                config=config,
                tags=config.wandb_tags,
                notes=config.wandb_notes,
            )

        # Initialize trajectory writer after wandb logger for real-time streaming
        if config.save_trajectory_pdb and pdb_template is not None:
            logger.info(
                "Initializing TrajectoryWriter with real-time wandb streaming..."
            )

            # TrajectoryWriter will create its own PDBParser from the template file
            self.trajectory_writer = TrajectoryWriter(
                output_dir=self.output_dir,
                pdb_template_path=pdb_template,
                save_interval=config.save_trajectory_interval,
                # Pass wandb logger for real-time streaming.
                wandb_logger=self.wandb_logger,
            )
        else:
            logger.warning(
                "TrajectoryWriter not initialized - no PDB template provided "
                "or saving disabled"
            )
            self.trajectory_writer = None

        # Global best tracking
        self.global_best_loss = float("inf")
        self.global_best_state: dict[str, Any] = {}

        logger.info("Initialized RefinementEngine")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(
            f"Config: {config.num_runs} runs × {config.num_iterations} iterations"
        )

    @timed
    @gpu_memory_tracked
    def run(
        self,
        reference_coordinates: torch.Tensor,
        prediction_callback: Callable[[], torch.Tensor],
        optimizer: torch.optim.Optimizer | None = None,
        save_pdb_callback: Callable | None = None,
        best_state_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """Run complete refinement process.

        Args:
            reference_coordinates: Reference structure coordinates [N, 3]
            prediction_callback: Callable that returns atomic data [N, 4]
                where columns are [x, y, z, confidence/bfactor].
                All model-specific logic (features, state, recycling)
                should be handled inside this callback.
                The callback is called with no arguments: callback()
            optimizer: Optional pre-configured optimizer. If None, engine will not
                      perform gradient-based optimization (useful for inference-only).
            save_pdb_callback: Optional function to save PDB files,
                              signature: callback(coordinates, path)

        Returns:
            Dictionary containing:
                - best_loss: Best loss value achieved
                - best_run: Run ID of best result
                - best_iteration: Iteration of best result
                - best_coordinates: Best coordinates
        """

        logger.info("=" * 60)
        logger.info("Starting refinement")
        logger.info("=" * 60)

        for run_idx in range(self.config.num_runs):
            run_id = uuid.uuid4()
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Run {run_id} ({run_idx + 1}/{self.config.num_runs})")
            logger.info(f"{'=' * 60}")

            run_results = self._run_single_refinement(
                run_id=run_id,
                reference_coordinates=reference_coordinates,
                prediction_callback=prediction_callback,
                optimizer=optimizer,
                save_pdb_callback=save_pdb_callback,
                best_state_callback=best_state_callback,
            )

            # Update global best
            if run_results["best_loss"] < self.global_best_loss:
                self.global_best_loss = run_results["best_loss"]
                self.global_best_state = {
                    "run_id": run_id,
                    "iteration": run_results["best_iteration"],
                    "loss": run_results["best_loss"],
                    "coordinates": run_results["best_coordinates"],
                }

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("REFINEMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best loss: {self.global_best_loss:.6f}")
        logger.info(f"Best run: {self.global_best_state['run_id']}")
        logger.info(f"Best iteration: {self.global_best_state['iteration']}")

        # Save final best
        self._save_final_best(save_pdb_callback)

        # Close trajectory writers BEFORE logging to wandb
        if self.trajectory_writer is not None:
            logger.info("Closing trajectory writers...")
            self.trajectory_writer.close()

        # Finish wandb run
        if self.wandb_logger is not None:
            logger.info("Logging final results to W&B...")
            # Log final best artifacts
            if self.config.save_best_pdb:
                best_pdb_path = self.output_dir / "trajectory" / "best.pdb"
                logger.info(
                    "Checking for best PDB: %s, exists: %s",
                    best_pdb_path,
                    best_pdb_path.exists(),
                )
                if best_pdb_path.exists():
                    self.wandb_logger.log_pdb(best_pdb_path, "best_model")
                    # Log as interactive 3D structure
                    logger.info("Logging final best as 3D molecule...")
                    self.wandb_logger.log_molecule_3d(
                        best_pdb_path,
                        caption=f"Best Model (Loss: {self.global_best_loss:.4f})",
                    )

            # Log final trajectory animation for interactive playback
            if self.config.save_trajectory_pdb:
                trajectory_dir = self.output_dir / "trajectory"
                if trajectory_dir.exists():
                    for traj_file in trajectory_dir.glob("*_refinement_trajectory.pdb"):
                        logger.info(f"Creating trajectory animation: {traj_file.name}")
                        self.wandb_logger.log_trajectory_3d(traj_file, max_frames=50)

            self.wandb_logger.log_config_file(self.output_dir / "config.yaml")
            logger.info("Finishing W&B run...")
            self.wandb_logger.finish()

        return self.global_best_state

    def _process_coordinates(
        self,
        coordinates: torch.Tensor,
        reference_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Process coordinates through alignment and optional RBR.

        Args:
            coordinates: Raw predicted coordinates [N, 3]
            reference_coordinates: Reference coordinates [N, 3]

        Returns:
            Refined coordinates [N, 3]
        """
        # Align to reference
        indices_moving = getattr(self, "alignment_indices_moving", None)
        indices_reference = getattr(self, "alignment_indices_reference", None)
        if indices_moving is None:
            indices_moving = getattr(self.loss_fn, "alignment_indices_moving", None)
        if indices_reference is None:
            indices_reference = getattr(
                self.loss_fn, "alignment_indices_reference", None
            )
        aligned_coords = kabsch_align(
            coordinates,
            reference_coordinates,
            indices_moving=indices_moving,
            indices_reference=indices_reference,
        )

        # Apply rigid body refinement if enabled
        if self.config.use_rigid_body_refinement and self.rbr_fn is not None:
            refined_coords, _ = self.rbr_fn(
                aligned_coords,
                self.loss_fn,
                self.sfc,
                domain_segs=self.config.domain_segments,
                lbfgs=self.config.rbr_use_lbfgs,
                lbfgs_lr=self.config.rbr_learning_rate,
            )
            return refined_coords

        return aligned_coords

    def _compute_loss_with_metadata(
        self,
        coordinates: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss and extract metadata.

        Args:
            coordinates: Coordinates to compute loss on [N, 3]
            confidence: Per-atom confidence values [N]

        Returns:
            Tuple of (loss, metadata_dict)
        """
        loss = self.loss_fn.compute(coordinates)
        metadata: dict[str, float] = {"mean_confidence": confidence.mean().item()}
        return loss, metadata

    def _optimize(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer | None,
        iteration: int,
    ) -> None:
        """Perform optimization step if optimizer is provided.

        Args:
            loss: Loss tensor to backpropagate
            optimizer: Optional optimizer
            iteration: Current iteration number (for logging)
        """
        if optimizer is None:
            return

        if not loss.requires_grad:
            logger.warning(f"Iteration {iteration}: Loss has no gradients")
            return
        loss.backward()
        optimizer.step()

    def _update_best_and_save(
        self,
        loss: torch.Tensor,
        refined_coords: torch.Tensor,
        iteration: int,
        run_id: uuid.UUID,
        run_best_loss: float,
        run_best_state: dict,
        save_pdb_callback: Callable | None,
        confidence: torch.Tensor | None = None,
        best_state_callback: Callable | None = None,
    ) -> tuple[float, dict]:
        """Update best state and save checkpoints/PDFs.

        Args:
            loss: Current loss value
            refined_coords: Current refined coordinates
            iteration: Current iteration
            run_id: Current run identifier
            run_best_loss: Best loss so far in this run
            run_best_state: Best state so far in this run
            save_pdb_callback: Optional callback to save PDB files
            confidence: Optional confidence/B-factor values

        Returns:
            Tuple of (updated_best_loss, updated_best_state)
        """
        loss_value = loss.item()

        # Check if this is a new best
        if loss_value < run_best_loss:
            run_best_loss = loss_value
            run_best_state = {
                "iteration": iteration,
                "loss": loss_value,
                "coordinates": refined_coords.detach().cpu().clone(),
            }

            # Skip coordinate snapshot checkpoints to avoid .pt files

            # Save best PDB structure
            if self.trajectory_writer is not None:
                self.trajectory_writer.save_best(
                    coordinates=refined_coords,
                    run_id=str(run_id),
                    iteration=iteration,
                    b_factors=confidence,
                )

            if best_state_callback is not None and loss_value < self.global_best_loss:
                best_state_callback(
                    run_id=run_id,
                    iteration=iteration,
                    loss=loss_value,
                )

        # Save trajectory frame
        if self.trajectory_writer is not None:
            self.trajectory_writer.save_frame(
                coordinates=refined_coords,
                iteration=iteration,
                run_id=str(run_id),
                b_factors=confidence,
                loss=loss_value,
            )

        # Periodic PDB saving via callback (legacy support)
        if (
            save_pdb_callback is not None
            and iteration % self.config.save_every_n_iterations == 0
        ):
            pdb_path = self.output_dir / f"{run_id}_{iteration}_refined.pdb"
            save_pdb_callback(refined_coords, pdb_path)

        return run_best_loss, run_best_state

    def _run_single_refinement(
        self,
        run_id: uuid.UUID,
        reference_coordinates: torch.Tensor,
        prediction_callback: Callable[[], torch.Tensor],
        optimizer: torch.optim.Optimizer | None,
        save_pdb_callback: Callable | None,
        best_state_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """Run single refinement iteration.

        Args:
            run_id: Run identifier
            reference_coordinates: Reference coordinates
            prediction_callback: Prediction function (no arguments)
            optimizer: Optional optimizer
            save_pdb_callback: PDB save function

        Returns:
            Dictionary with run results
        """
        # Initialize tracking
        metrics = MetricsTracker(
            self.output_dir, str(run_id), log_to_file=self.config.log_metrics
        )
        early_stopper = EarlyStopper(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )

        # Track best for this run
        run_best_loss = float("inf")
        run_best_state = {
            "iteration": 0,
            "loss": float("inf"),
            "coordinates": None,
        }

        # Progress bar
        progress = tqdm(
            range(self.config.num_iterations),
            desc=f"Run {run_id}",
        )

        # Main refinement loop
        for iteration in progress:
            if optimizer is not None:
                optimizer.zero_grad()

            if torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    torch.cuda.reset_peak_memory_stats()

            # Process coordinates through pipeline
            prediction = prediction_callback()
            coordinates, confidence = prediction[:, :3], prediction[:, 3]
            refined_coords = self._process_coordinates(
                coordinates, reference_coordinates
            )

            # Compute loss with metadata
            loss, metadata = self._compute_loss_with_metadata(
                refined_coords, confidence
            )
            # Optimization step
            self._optimize(loss, optimizer, iteration)

            # Track metrics
            metrics.log(
                iteration=iteration,
                loss=loss.item(),
                **metadata,
            )

            # Log to wandb
            if self.wandb_logger is not None:
                wandb_metrics = {
                    f"{run_id}/loss": loss.item(),
                    f"{run_id}/iteration": iteration,
                    "optimization_loss": loss.item(),
                    "iteration": iteration,
                }
                wandb_metrics.update({f"{run_id}/{k}": v for k, v in metadata.items()})
                self.wandb_logger.log(wandb_metrics, step=iteration)

            # Update progress bar
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                memory=f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}G",
            )

            # Update best and save checkpoints
            run_best_loss, run_best_state = self._update_best_and_save(
                loss=loss,
                refined_coords=refined_coords,
                iteration=iteration,
                run_id=run_id,
                run_best_loss=run_best_loss,
                run_best_state=run_best_state,
                save_pdb_callback=save_pdb_callback,
                confidence=confidence,
                best_state_callback=best_state_callback,
            )

            # Early stopping check
            if early_stopper.should_stop(loss.item()):
                logger.info(f"Early stopping at iteration {iteration}")
                break

            # Clear cache periodically
            if iteration % 10 == 0:
                torch.cuda.empty_cache()

        # Save metrics
        metrics.save()

        # Return results with proper keys
        return {
            "best_loss": run_best_loss,
            "best_iteration": run_best_state["iteration"],
            "best_coordinates": run_best_state["coordinates"],
        }

    def _save_final_best(self, save_pdb_callback: Callable | None = None) -> None:
        """Save final best results.

        Args:
            save_pdb_callback: Optional PDB save function
        """
        if not self.global_best_state:
            logger.warning("No best state to save")
            return

        # Log trajectory files to wandb if enabled
        if self.wandb_logger:
            trajectory_dir = self.output_dir / "trajectory"
            if trajectory_dir.exists():
                for traj_file in trajectory_dir.glob("*_refinement_trajectory.pdb"):
                    run_id = traj_file.stem.split("_")[0]  # Extract A, B, C
                    self.wandb_logger.log_artifact(
                        str(traj_file),
                        name=f"trajectory_{run_id}",
                        artifact_type="trajectory",
                    )
                    logger.info(f"Logged trajectory to W&B: {traj_file.name}")

        # Save coordinates as tensor
        torch.save(
            self.global_best_state["coordinates"],
            self.output_dir / "final_best_coordinates.pt",
        )

        # Save summary
        import json

        summary = {
            "best_loss": self.global_best_loss,
            "best_run": self.global_best_state["run_id"],
            "best_iteration": self.global_best_state["iteration"],
        }

        with open(self.output_dir / "refinement_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved final best results to {self.output_dir}")
