"""Decorators for common patterns in coordinate refinement."""

import functools
import time
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger


def gpu_memory_tracked(func: Callable) -> Callable:
    """Track GPU memory usage before and after function call.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that logs memory usage
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / 1024**3

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            logger.debug(
                f"{func.__name__}: Memory before={mem_before:.2f}GB, "
                f"after={mem_after:.2f}GB, peak={mem_peak:.2f}GB"
            )

        return result

    return wrapper


def timed(func: Callable) -> Callable:
    """Time function execution.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that logs execution time
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__}: {elapsed:.3f}s")
        return result

    return wrapper


def validate_shapes(*expected_shapes):
    """Validate tensor shapes match expected dimensions.

    Args:
        expected_shapes: Tuples of (arg_index, expected_shape) or
                        (arg_name, expected_shape) for kwargs

    Example:
        @validate_shapes((0, (None, 3)), (1, (None, 3)))
        def align_coords(coords1, coords2):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for spec in expected_shapes:
                if isinstance(spec[0], int):
                    # Positional argument
                    tensor = args[spec[0]]
                    expected = spec[1]
                else:
                    # Keyword argument
                    tensor = kwargs.get(spec[0])
                    expected = spec[1]

                if tensor is not None and isinstance(tensor, torch.Tensor):
                    actual_shape = tensor.shape
                    for i, exp_dim in enumerate(expected):
                        if exp_dim is not None and actual_shape[i] != exp_dim:
                            raise ValueError(
                                f"Shape mismatch in {func.__name__}: "
                                f"expected {expected}, got {actual_shape}"
                            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def cached_property(func: Callable) -> property:
    """Cache property result after first computation.

    Args:
        func: Property function to wrap

    Returns:
        Cached property
    """
    attr_name = f"_cached_{func.__name__}"

    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return property(wrapper)


def handle_oom(fallback_value: Any = None):
    """Handle CUDA out-of-memory errors gracefully.

    Args:
        fallback_value: Value to return on OOM error

    Returns:
        Decorated function that catches OOM
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"OOM in {func.__name__}")
                    if torch.cuda.is_available():
                        mem_gb = torch.cuda.memory_allocated() / 1024**3
                        logger.error(f"Allocated: {mem_gb:.2f}GB")
                        torch.cuda.empty_cache()
                    return fallback_value
                raise

        return wrapper

    return decorator


def batch_operation(batch_size: int = 100):
    """Process operations in batches to reduce memory usage.

    Args:
        batch_size: Number of items to process per batch

    Returns:
        Decorated function that processes in batches
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(data: torch.Tensor, *args, **kwargs):
            if len(data) <= batch_size:
                return func(data, *args, **kwargs)

            results = []
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                batch_result = func(batch, *args, **kwargs)
                results.append(batch_result)

            return torch.cat(results, dim=0)

        return wrapper

    return decorator
