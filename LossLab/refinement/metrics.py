"""Metrics tracking for refinement runs."""

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger


class MetricsTracker:
    """Track and log metrics during refinement.

    Example:
        >>> tracker = MetricsTracker(output_dir="./output", run_id="A")
        >>> tracker.log(iteration=0, loss=1.23, mean_plddt=0.85)
        >>> tracker.save()
    """

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str = "A",
        log_to_file: bool = True,
    ):
        """Initialize metrics tracker.

        Args:
            output_dir: Directory to save metrics
            run_id: Identifier for this run
            log_to_file: Whether to write metrics to file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.log_to_file = log_to_file

        # Storage
        self.metrics: dict[str, list] = defaultdict(list)
        self.iteration_times: list[float] = []
        self.start_time = time.time()

        # Initialize log file
        if self.log_to_file:
            self.log_file = self.output_dir / f"{run_id}_metrics.csv"
            self._init_log_file()

    def _init_log_file(self) -> None:
        """Initialize CSV log file with header."""
        with open(self.log_file, "w") as f:
            f.write("iteration,loss,time_sec,memory_gb\n")

    def log(
        self,
        iteration: int,
        **metrics: float | torch.Tensor,
    ) -> None:
        """Log metrics for current iteration.

        Args:
            iteration: Current iteration number
            **metrics: Metric name-value pairs
        """
        # Convert tensors to floats
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = value.item()
            else:
                processed_metrics[key] = value

        # Store
        self.metrics["iteration"].append(iteration)
        for key, value in processed_metrics.items():
            self.metrics[key].append(value)

        # Memory tracking
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            self.metrics["memory_gb"].append(memory_gb)

        # Time tracking
        elapsed = time.time() - self.start_time
        self.metrics["time_sec"].append(elapsed)

        # Write to file
        if self.log_to_file:
            with open(self.log_file, "a") as f:
                # Build CSV line
                values = [str(iteration)]
                if "loss" in processed_metrics:
                    values.append(f"{processed_metrics['loss']:.6f}")
                else:
                    values.append("NA")
                values.append(f"{elapsed:.3f}")
                if torch.cuda.is_available():
                    values.append(f"{memory_gb:.3f}")
                else:
                    values.append("NA")
                f.write(",".join(values) + "\n")

        # Log to console if verbose
        log_str = f"Iter {iteration}: "
        log_str += ", ".join(f"{k}={v:.4f}" for k, v in processed_metrics.items())
        logger.debug(log_str)

    def save(self, filename: str | None = None) -> None:
        """Save all metrics to numpy file.

        Args:
            filename: Output filename (default: {run_id}_metrics.npz)
        """
        if filename is None:
            filename = f"{self.run_id}_metrics.npz"

        output_path = self.output_dir / filename

        # Convert to numpy arrays
        np_metrics = {key: np.array(values) for key, values in self.metrics.items()}

        np.savez(output_path, **np_metrics)
        logger.info(f"Saved metrics to {output_path}")

    def get_best(
        self, metric: str = "loss", minimize: bool = True
    ) -> tuple[int, float]:
        """Get best iteration for a given metric.

        Args:
            metric: Metric name to evaluate
            minimize: Whether lower is better

        Returns:
            Tuple of (best_iteration, best_value)
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric '{metric}' not found")

        values = self.metrics[metric]
        best_idx = np.argmin(values) if minimize else np.argmax(values)

        best_iter = self.metrics["iteration"][best_idx]
        best_value = values[best_idx]

        return best_iter, best_value

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics.

        Returns:
            Dictionary of summary statistics
        """
        summary_dict = {
            "total_iterations": len(self.metrics.get("iteration", [])),
            "total_time_sec": time.time() - self.start_time,
            "run_id": self.run_id,
        }

        # Add best values for key metrics
        for metric in ["loss"]:
            if metric in self.metrics:
                best_iter, best_val = self.get_best(metric, minimize=True)
                summary_dict[f"best_{metric}"] = best_val
                summary_dict[f"best_{metric}_iteration"] = best_iter

        # Add final values
        for key, values in self.metrics.items():
            if values and key != "iteration":
                summary_dict[f"final_{key}"] = values[-1]

        return summary_dict
