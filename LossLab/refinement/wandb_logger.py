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
            config = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}

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

    def log_molecule_3d(
        self,
        pdb_path: str | Path,
        caption: str | None = None,
        step: int | None = None,
    ) -> None:
        """Log 3D molecular structure for interactive visualization in W&B.

        Args:
            pdb_path: Path to PDB file
            caption: Optional caption for the visualization
            step: Optional step number (iteration)
        """
        if not self.enabled or self.run is None:
            return

        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            logger.warning(f"PDB file not found: {pdb_path}")
            return

        try:
            logger.info(f"Logging 3D molecule from {pdb_path}")

            # Create Molecule3D object - pass file path, not content!
            molecule = wandb.Molecule(str(pdb_path))
            logger.info("Created wandb.Molecule object")

            # Log with optional caption
            log_dict = {"molecule_3d": molecule}
            if caption:
                log_dict["caption"] = caption

            logger.info(f"Logging to W&B with step={step}")
            wandb.log(log_dict, step=step)
            logger.info(f"✓ Successfully logged 3D molecule: {pdb_path.name}")
        except Exception as e:
            logger.error(f"Failed to log 3D molecule {pdb_path}: {e}")
            logger.exception(e)

    def log_trajectory_3d(
        self,
        trajectory_path: str | Path,
        max_frames: int = 100,
    ) -> None:
        """Log multi-model PDB trajectory with 3D animated visualization.

        Creates both a table view and an animated 3D viewer using 3Dmol.js.

        Args:
            trajectory_path: Path to multi-model PDB file
            max_frames: Maximum number of frames to log (to avoid UI overload)
        """
        if not self.enabled or self.run is None:
            return

        trajectory_path = Path(trajectory_path)
        if not trajectory_path.exists():
            logger.warning(f"Trajectory file not found: {trajectory_path}")
            return

        try:
            import mdtraj as md

            logger.info(f"Logging 3D trajectory from {trajectory_path}")

            # Load trajectory
            traj = md.load(str(trajectory_path))
            n_frames = min(traj.n_frames, max_frames)

            # Create animated HTML viewer with 3Dmol.js
            html_path = (
                trajectory_path.parent / f"{trajectory_path.stem}_animation.html"
            )
            self._create_3dmol_animation(trajectory_path, html_path, n_frames)

            # Log the animated HTML viewer
            wandb.log({
                f"trajectory_animation_{trajectory_path.stem}": wandb.Html(
                    str(html_path)
                )
            })

            logger.info(
                "✓ Logged animated 3D trajectory with %s frames: %s",
                n_frames,
                trajectory_path.name,
            )

        except ImportError:
            logger.warning("mdtraj not available, cannot load trajectory")
        except Exception as e:
            logger.warning(f"Failed to log 3D trajectory {trajectory_path}: {e}")
            logger.exception(e)

    def _create_3dmol_animation(
        self,
        pdb_path: Path,
        output_html: Path,
        n_frames: int,
    ) -> None:
        """Create an HTML file with 3Dmol.js animation of trajectory.

        Args:
            pdb_path: Path to multi-model PDB file
            output_html: Path for output HTML file
            n_frames: Number of frames in trajectory
        """
        # Read the PDB file content
        with open(pdb_path) as f:
            pdb_content = f.read()

        # Escape for JavaScript
        pdb_content_js = (
            pdb_content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        )

        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"    <title>Trajectory Animation - {pdb_path.stem}</title>",
            '    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>',
            "    <style>",
            "        body { margin: 0; padding: 5px; font-family: Arial, sans-serif;",
            "            background: white; }",
            "        #container { width: 100%; max-width: 500px; height: 350px;",
            "            position: relative; margin: 0 auto; border: 1px solid #ddd;",
            "            box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            "        #controls { text-align: center; margin: 10px; }",
            "        button { margin: 2px; padding: 6px 12px; font-size: 12px;",
            "            cursor: pointer; border: 1px solid #ccc; background: #f5f5f5;",
            "            border-radius: 3px; }",
            "        button:hover { background: #e0e0e0; }",
            "        #frameInfo { margin: 8px; font-size: 13px; font-weight: bold;",
            "            text-align: center; }",
            "        h2 { text-align: center; margin: 8px 0; font-size: 16px; }",
            "        label { font-size: 12px; }",
            "        #speed { vertical-align: middle; width: 100px; }",
            "    </style>",
            "</head>",
            "<body>",
            f'    <h2 style="text-align: center;">Trajectory: {pdb_path.stem}</h2>',
            f'    <div id="frameInfo">Frame: 0 / {n_frames - 1}</div>',
            '    <div id="container"></div>',
            '    <div id="controls">',
            '        <button onclick="playAnimation()">▶ Play</button>',
            '        <button onclick="pauseAnimation()">⏸ Pause</button>',
            '        <button onclick="resetAnimation()">⏮ Reset</button>',
            '        <button onclick="prevFrame()">◀ Prev</button>',
            '        <button onclick="nextFrame()">▶ Next</button>',
            "        <label>Speed: ",
            '            <input type="range" id="speed" min="100"',
            '                max="2000" value="500" step="100">',
            '            <span id="speedLabel">500ms</span>',
            "        </label>",
            "    </div>",
            "",
            "    <script>",
            "        let viewer = null;",
            "        let currentFrame = 0;",
            "        let isPlaying = false;",
            "        let animationInterval = null;",
            "        let animationSpeed = 500;",
            f"        let numFrames = {n_frames};",
            "",
            '        viewer = $3Dmol.createViewer("container", {',
            '            backgroundColor: "white"',
            "        });",
            "",
            f"        const pdbData = `{pdb_content_js}`;",
            '        viewer.addModelsAsFrames(pdbData, "pdb");',
            "",
            '        viewer.setStyle({}, {cartoon: {color: "spectrum"}});',
            "        viewer.zoomTo();",
            '        viewer.animate({loop: "forward", reps: 0});',
            "        viewer.stopAnimate();",
            "        viewer.render();",
            "",
            "        function updateFrameDisplay() {",
            "            const info = document.getElementById('frameInfo');",
            "            info.textContent = `Frame: ${currentFrame} / ` +",
            "                `${numFrames - 1}`;",
            "        }",
            "",
            "        function showFrame(frameNum) {",
            "            currentFrame = frameNum % numFrames;",
            "            viewer.setFrame(currentFrame);",
            "            viewer.render();",
            "            updateFrameDisplay();",
            "        }",
            "",
            "        function nextFrame() {",
            "            showFrame(currentFrame + 1);",
            "        }",
            "",
            "        function prevFrame() {",
            "            showFrame(currentFrame - 1 + numFrames);",
            "        }",
            "",
            "        function playAnimation() {",
            "            if (isPlaying) return;",
            "            isPlaying = true;",
            "            animationInterval = setInterval(() => {",
            "                nextFrame();",
            "            }, animationSpeed);",
            "        }",
            "",
            "        function pauseAnimation() {",
            "            isPlaying = false;",
            "            if (animationInterval) {",
            "                clearInterval(animationInterval);",
            "                animationInterval = null;",
            "            }",
            "        }",
            "",
            "        function resetAnimation() {",
            "            pauseAnimation();",
            "            showFrame(0);",
            "        }",
            "",
            "        document.getElementById('speed').addEventListener(",
            "            'input',",
            "            function(e) {",
            "                animationSpeed = parseInt(e.target.value);",
            "                const label = document.getElementById('speedLabel');",
            "                label.textContent = animationSpeed + 'ms';",
            "                if (isPlaying) {",
            "                    pauseAnimation();",
            "                    playAnimation();",
            "                }",
            "            }",
            "        );",
            "",
            "        updateFrameDisplay();",
            "    </script>",
            "</body>",
            "</html>",
        ]
        html_content = "\n".join(html_lines)

        with open(output_html, "w") as f:
            f.write(html_content)

        logger.info(f"Created 3Dmol.js animation HTML: {output_html}")

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
            logger.warning(
                f"Cannot log artifact {name} - wandb not enabled or run is None"
            )
            return

        try:
            logger.info(f"Creating artifact: {name} (type: {artifact_type})")
            artifact = wandb.Artifact(name=name, type=artifact_type)
            logger.info(f"Adding file to artifact: {file_path}")
            artifact.add_file(file_path)
            logger.info("Logging artifact to wandb...")
            self.run.log_artifact(artifact)
            logger.info(f"Successfully logged artifact: {name}")
        except Exception as e:
            logger.error(f"Failed to log artifact {name}: {e}")
